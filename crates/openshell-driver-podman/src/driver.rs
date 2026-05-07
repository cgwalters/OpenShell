// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Podman compute driver.

use crate::client::{PodmanApiError, PodmanClient};
use crate::config::PodmanComputeConfig;
use crate::container::{self, LABEL_MANAGED_FILTER, LABEL_ROLE, LABEL_SANDBOX_ID, ROLE_AGENT};
use crate::watcher::{
    self, WatchStream, driver_sandbox_from_inspect, driver_sandbox_from_list_entry,
};
use openshell_core::ComputeDriverError;
use openshell_core::proto::compute::v1::{DriverSandbox, GetCapabilitiesResponse};
use tracing::{info, warn};

impl From<PodmanApiError> for ComputeDriverError {
    fn from(value: PodmanApiError) -> Self {
        match value {
            PodmanApiError::Conflict(_) => Self::AlreadyExists,
            PodmanApiError::NotFound(msg) => Self::Message(format!("not found: {msg}")),
            other => Self::Message(other.to_string()),
        }
    }
}

/// Podman compute driver managing sandbox containers via the Podman REST API.
#[derive(Clone)]
pub struct PodmanComputeDriver {
    client: PodmanClient,
    config: PodmanComputeConfig,
    /// The host's IP on the bridge network. Sandbox containers use this to
    /// reach the gateway server when no explicit gRPC endpoint is configured.
    network_gateway_ip: Option<String>,
}

impl std::fmt::Debug for PodmanComputeDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PodmanComputeDriver")
            .field("socket_path", &self.config.socket_path)
            .field("default_image", &self.config.default_image)
            .field("network_name", &self.config.network_name)
            .finish()
    }
}

/// Construct and validate a container name from a sandbox name.
///
/// Combines the prefix with the sandbox name and validates the result
/// against Podman's naming rules before any resources are created.
fn validated_container_name(sandbox_name: &str) -> Result<String, ComputeDriverError> {
    let name = container::container_name(sandbox_name);
    crate::client::validate_name(&name)
        .map_err(|e| ComputeDriverError::Precondition(e.to_string()))?;
    Ok(name)
}

impl PodmanComputeDriver {
    /// Create a new driver, verifying the Podman socket is reachable.
    pub async fn new(mut config: PodmanComputeConfig) -> Result<Self, PodmanApiError> {
        if !config.socket_path.exists() {
            if cfg!(target_os = "macos") {
                warn!(
                    path = %config.socket_path.display(),
                    "Podman socket not found; is podman machine running? \
                     Try `podman machine start` or set OPENSHELL_PODMAN_SOCKET to override."
                );
            } else {
                warn!(
                    path = %config.socket_path.display(),
                    "Podman socket not found; is the Podman service running? \
                     Set OPENSHELL_PODMAN_SOCKET or XDG_RUNTIME_DIR to override."
                );
            }
        }

        // Validate TLS configuration before connecting.  Partial configs
        // (e.g. CA set but cert/key missing) are rejected early so operators
        // get a clear error instead of a silent fallback to plaintext HTTP.
        config.validate_tls_config()?;

        let client = PodmanClient::new(config.socket_path.clone());

        // Verify connectivity.
        client.ping().await?;

        // Verify cgroups v2, detect rootless mode, and log system info.
        match client.system_info().await {
            Ok(info) => {
                if info.host.cgroup_version != "v2" {
                    return Err(PodmanApiError::Connection(format!(
                        "cgroups v2 is required; detected cgroups '{}'. \
                         Ensure your host uses a unified cgroup hierarchy \
                         (systemd.unified_cgroup_hierarchy=1).",
                        info.host.cgroup_version
                    )));
                }
                info!(
                    cgroup_version = %info.host.cgroup_version,
                    network_backend = %info.host.network_backend,
                    rootless = info.host.security.rootless,
                    "Connected to Podman"
                );
            }
            Err(e) => {
                return Err(PodmanApiError::Connection(format!(
                    "failed to query Podman system info: {e}"
                )));
            }
        }

        // Rootless pre-flight: warn if subuid/subgid ranges look missing.
        // Not a hard error because some systems configure these via LDAP or
        // other mechanisms that /etc/subuid does not reflect.
        if nix::unistd::getuid().as_raw() != 0 {
            check_subuid_range();
        }

        // Ensure the bridge network exists.
        client.ensure_network(&config.network_name).await?;
        let network_gateway_ip = client
            .network_gateway_ip(&config.network_name)
            .await
            .unwrap_or(None);
        info!(
            network = %config.network_name,
            gateway_ip = ?network_gateway_ip,
            "Bridge network ready"
        );

        // Auto-detect the gRPC callback endpoint when not explicitly
        // configured. Sandbox containers use host.containers.internal
        // (injected via hostadd with host-gateway in the container spec)
        // to reach the gateway server on the host. The scheme is
        // determined by whether TLS client certs are configured: when
        // all three TLS paths are set, the endpoint uses https so the
        // supervisor connects with mTLS.
        if config.grpc_endpoint.is_empty() {
            let scheme = if config.tls_enabled() {
                "https"
            } else {
                "http"
            };
            config.grpc_endpoint = format!(
                "{scheme}://host.containers.internal:{}",
                config.gateway_port
            );
            info!(
                grpc_endpoint = %config.grpc_endpoint,
                tls = config.tls_enabled(),
                "Auto-detected gRPC endpoint"
            );
        }

        // Discover host DNS resolvers for proxy sidecar containers.
        // The proxy runs on an internal network and needs real upstream DNS
        // servers to resolve external hostnames.
        if config.host_dns_servers.is_empty() {
            config.host_dns_servers = crate::config::discover_host_dns_servers();
            if !config.host_dns_servers.is_empty() {
                info!(
                    dns_servers = ?config.host_dns_servers,
                    "Discovered host DNS resolvers for proxy containers"
                );
            }
        }

        Ok(Self {
            client,
            config,
            network_gateway_ip,
        })
    }

    /// The host's IP on the bridge network, if available.
    ///
    /// Used by the server to auto-detect the gRPC callback endpoint when
    /// no explicit `--grpc-endpoint` is configured.
    #[must_use]
    pub fn network_gateway_ip(&self) -> Option<&str> {
        self.network_gateway_ip.as_deref()
    }

    /// Report driver capabilities.
    pub fn capabilities(&self) -> Result<GetCapabilitiesResponse, ComputeDriverError> {
        let supports_gpu = Self::has_gpu_capacity();
        Ok(GetCapabilitiesResponse {
            driver_name: "podman".to_string(),
            driver_version: openshell_core::VERSION.to_string(),
            default_image: self.config.default_image.clone(),
            supports_gpu,
            gpu_count: 0,
        })
    }

    #[must_use]
    pub fn default_image(&self) -> &str {
        &self.config.default_image
    }

    /// Check whether GPU devices are available via CDI.
    ///
    /// The Podman system info response doesn't directly list CDI devices in all
    /// versions. As a heuristic, check if the NVIDIA device node exists (this
    /// works for both rootful and rootless).
    fn has_gpu_capacity() -> bool {
        std::path::Path::new("/dev/nvidia0").exists()
    }

    /// Validate a sandbox before creation.
    pub fn validate_sandbox_create(
        &self,
        sandbox: &DriverSandbox,
    ) -> Result<(), ComputeDriverError> {
        let gpu_requested = sandbox.spec.as_ref().is_some_and(|s| s.gpu);
        if gpu_requested && !Self::has_gpu_capacity() {
            return Err(ComputeDriverError::Precondition(
                "GPU sandbox requested, but no NVIDIA GPU devices are available.".to_string(),
            ));
        }
        Ok(())
    }

    /// Create a sandbox: either a single container (nested/passthrough mode)
    /// or the 3-resource sidecar topology (internal network + proxy + agent).
    pub async fn create_sandbox(&self, sandbox: &DriverSandbox) -> Result<(), ComputeDriverError> {
        if sandbox.name.is_empty() {
            return Err(ComputeDriverError::Precondition(
                "sandbox name is required".into(),
            ));
        }
        if sandbox.id.is_empty() {
            return Err(ComputeDriverError::Precondition(
                "sandbox id is required".into(),
            ));
        }

        if container::is_nested_mode(sandbox) {
            return self.create_sandbox_passthrough(sandbox).await;
        }

        self.create_sandbox_sidecar(sandbox).await
    }

    /// Nested/passthrough mode: single container with supervisor injected
    /// for SSH/exec, but no inner sandboxing or sidecar proxy.
    async fn create_sandbox_passthrough(
        &self,
        sandbox: &DriverSandbox,
    ) -> Result<(), ComputeDriverError> {
        let name = validated_container_name(&sandbox.name)?;
        let vol_name = container::volume_name(&sandbox.id);
        let sec_name = container::secret_name(&sandbox.id);

        info!(
            sandbox_id = %sandbox.id,
            sandbox_name = %sandbox.name,
            container = %name,
            mode = "nested",
            "Creating sandbox in nested mode (supervisor with no enforcement)"
        );

        // 1a. Pull the supervisor image if needed. The supervisor binary
        //     is shipped in a standalone OCI image and mounted into sandbox
        //     containers via Podman's type=image mount. Using "missing"
        //     policy so the image is only pulled once and then cached.
        info!(
            image = %self.config.supervisor_image,
            policy = "missing",
            "Ensuring supervisor image"
        );
        self.client
            .pull_image(&self.config.supervisor_image, "missing")
            .await
            .map_err(ComputeDriverError::from)?;

        // 1b. Pull the sandbox image.
        let image = container::resolve_image(sandbox, &self.config);
        if image.is_empty() {
            return Err(ComputeDriverError::Precondition(
                "no sandbox image configured: set --sandbox-image on the server \
                 or provide an image in the sandbox template"
                    .to_string(),
            ));
        }
        let pull_policy = self.config.image_pull_policy.as_str();
        info!(image = %image, policy = %pull_policy, "Ensuring sandbox image");
        self.client
            .pull_image(image, pull_policy)
            .await
            .map_err(ComputeDriverError::from)?;

        // 2. Create the SSH handshake secret via the Podman secrets API
        //    so it is not exposed in `podman inspect` output.
        self.client
            .create_secret(&sec_name, self.config.ssh_handshake_secret.as_bytes())
            .await
            .map_err(ComputeDriverError::from)?;

        // 3. Create workspace volume.
        if let Err(e) = self.client.create_volume(&vol_name).await {
            let _ = self.client.remove_secret(&sec_name).await;
            return Err(ComputeDriverError::from(e));
        }

        // 4. Create container.
        let spec = container::build_container_spec(sandbox, &self.config);
        match self.client.create_container(&spec).await {
            Ok(_) => {}
            Err(PodmanApiError::Conflict(_)) => {
                let _ = self.client.remove_volume(&vol_name).await;
                let _ = self.client.remove_secret(&sec_name).await;
                return Err(ComputeDriverError::AlreadyExists);
            }
            Err(e) => {
                let _ = self.client.remove_volume(&vol_name).await;
                let _ = self.client.remove_secret(&sec_name).await;
                return Err(ComputeDriverError::from(e));
            }
        }

        // 5. Start container.
        if let Err(e) = self.client.start_container(&name).await {
            warn!(
                sandbox_name = %sandbox.name,
                error = %e,
                "Failed to start container; cleaning up"
            );
            let _ = self.client.remove_container(&name).await;
            let _ = self.client.remove_volume(&vol_name).await;
            let _ = self.client.remove_secret(&sec_name).await;
            return Err(ComputeDriverError::from(e));
        }

        info!(
            sandbox_id = %sandbox.id,
            sandbox_name = %sandbox.name,
            "Sandbox container started (nested mode)"
        );

        Ok(())
    }

    /// Sidecar mode: create internal network, proxy sidecar, and agent container.
    async fn create_sandbox_sidecar(
        &self,
        sandbox: &DriverSandbox,
    ) -> Result<(), ComputeDriverError> {
        let agent_name = validated_container_name(&sandbox.name)?;
        let proxy_name = container::proxy_container_name(&sandbox.name);
        crate::client::validate_name(&proxy_name)
            .map_err(|e| ComputeDriverError::Precondition(e.to_string()))?;

        let internal_net = container::internal_network_name(&sandbox.id);
        let tls_vol = container::tls_volume_name(&sandbox.id);
        let workspace_vol = container::volume_name(&sandbox.id);
        let sec_name = container::secret_name(&sandbox.id);

        info!(
            sandbox_id = %sandbox.id,
            sandbox_name = %sandbox.name,
            "Creating sidecar sandbox"
        );

        // 1. Pull images (supervisor + sandbox).
        self.client
            .pull_image(&self.config.supervisor_image, "missing")
            .await
            .map_err(ComputeDriverError::from)?;

        let image = container::resolve_image(sandbox, &self.config);
        if image.is_empty() {
            return Err(ComputeDriverError::Precondition(
                "no sandbox image configured: set --sandbox-image on the server \
                 or provide an image in the sandbox template"
                    .to_string(),
            ));
        }
        self.client
            .pull_image(image, self.config.image_pull_policy.as_str())
            .await
            .map_err(ComputeDriverError::from)?;

        // 2. Create internal network.
        self.client
            .ensure_internal_network(&internal_net)
            .await
            .map_err(ComputeDriverError::from)?;

        // 3. Create SSH handshake secret.
        if let Err(e) = self
            .client
            .create_secret(&sec_name, self.config.ssh_handshake_secret.as_bytes())
            .await
        {
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 4. Create workspace volume.
        if let Err(e) = self.client.create_volume(&workspace_vol).await {
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 5. Create TLS volume (proxy writes CA + .ready; agent reads).
        if let Err(e) = self.client.create_volume(&tls_vol).await {
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 6. Create proxy sidecar container on internal network.
        let host_dns = self.config.host_dns_servers.clone();
        let proxy_spec =
            container::build_proxy_sidecar_spec(sandbox, &self.config, &internal_net, &host_dns);
        if let Err(e) = self.client.create_container(&proxy_spec).await {
            let _ = self.client.remove_volume(&tls_vol).await;
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 7. Connect proxy to bridge network (gives it internet access).
        if let Err(e) = self
            .client
            .network_connect(&self.config.network_name, &proxy_name)
            .await
        {
            let _ = self.client.remove_container(&proxy_name).await;
            let _ = self.client.remove_volume(&tls_vol).await;
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 8. Start proxy sidecar.
        if let Err(e) = self.client.start_container(&proxy_name).await {
            let _ = self.client.remove_container(&proxy_name).await;
            let _ = self.client.remove_volume(&tls_vol).await;
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 9. Get proxy IP on internal network (needed for agent's HTTP_PROXY env).
        //    Roll back all resources if the proxy has no IP (crash, API timeout, etc.).
        let proxy_ip = match self.client.container_ip(&proxy_name, &internal_net).await {
            Ok(Some(ip)) => ip,
            result => {
                let err_msg = match result {
                    Ok(None) => "proxy container has no IP on internal network".to_string(),
                    Err(e) => format!("failed to get proxy container IP: {e}"),
                    Ok(Some(_)) => unreachable!(),
                };
                warn!(sandbox_name = %sandbox.name, error = %err_msg, "Rolling back sidecar sandbox");
                let _ = self.client.stop_container(&proxy_name, 5).await;
                let _ = self.client.remove_container(&proxy_name).await;
                let _ = self.client.remove_volume(&tls_vol).await;
                let _ = self.client.remove_volume(&workspace_vol).await;
                let _ = self.client.remove_secret(&sec_name).await;
                let _ = self.client.remove_network(&internal_net).await;
                return Err(ComputeDriverError::Message(err_msg));
            }
        };

        // 10. Create agent container on internal network only.
        let agent_spec = container::build_agent_container_spec(
            sandbox,
            &self.config,
            &internal_net,
            &proxy_ip,
            &tls_vol,
        );
        if let Err(e) = self.client.create_container(&agent_spec).await {
            let _ = self.client.stop_container(&proxy_name, 5).await;
            let _ = self.client.remove_container(&proxy_name).await;
            let _ = self.client.remove_volume(&tls_vol).await;
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        // 11. Start agent container.
        if let Err(e) = self.client.start_container(&agent_name).await {
            let _ = self.client.remove_container(&agent_name).await;
            let _ = self.client.stop_container(&proxy_name, 5).await;
            let _ = self.client.remove_container(&proxy_name).await;
            let _ = self.client.remove_volume(&tls_vol).await;
            let _ = self.client.remove_volume(&workspace_vol).await;
            let _ = self.client.remove_secret(&sec_name).await;
            let _ = self.client.remove_network(&internal_net).await;
            return Err(ComputeDriverError::from(e));
        }

        info!(
            sandbox_id = %sandbox.id,
            sandbox_name = %sandbox.name,
            proxy_ip = %proxy_ip,
            "Sidecar sandbox created"
        );

        Ok(())
    }

    /// Stop a sandbox: agent container first, then proxy sidecar (best-effort).
    pub async fn stop_sandbox(&self, sandbox_name: &str) -> Result<(), ComputeDriverError> {
        let agent_name = validated_container_name(sandbox_name)?;
        let proxy_name = container::proxy_container_name(sandbox_name);

        info!(sandbox_name = %sandbox_name, container = %agent_name, "Stopping sandbox");

        // Stop agent first (it depends on proxy).
        self.client
            .stop_container(&agent_name, self.config.stop_timeout_secs)
            .await
            .map_err(ComputeDriverError::from)?;

        // Stop proxy (best-effort — may not exist in passthrough mode).
        let _ = self.client.stop_container(&proxy_name, 5).await;

        Ok(())
    }

    /// Delete a sandbox and all its resources (agent, proxy, volumes, secret, network).
    pub async fn delete_sandbox(
        &self,
        sandbox_id: &str,
        sandbox_name: &str,
    ) -> Result<bool, ComputeDriverError> {
        if sandbox_id.is_empty() {
            return Err(ComputeDriverError::Precondition(
                "sandbox id is required".into(),
            ));
        }
        let agent_name = validated_container_name(sandbox_name)?;
        let proxy_name = container::proxy_container_name(sandbox_name);
        let internal_net = container::internal_network_name(sandbox_id);
        let tls_vol = container::tls_volume_name(sandbox_id);
        let workspace_vol = container::volume_name(sandbox_id);
        let sec_name = container::secret_name(sandbox_id);

        info!(
            sandbox_id = %sandbox_id,
            sandbox_name = %sandbox_name,
            "Deleting sandbox"
        );

        // Use the request's stable sandbox ID as the source of truth for
        // cleanup. Inspect is only used as a best-effort cross-check so
        // cleanup still works if the container is already gone or mislabeled.
        match self.client.inspect_container(&agent_name).await {
            Ok(inspect) => match inspect.config.labels.get(LABEL_SANDBOX_ID) {
                Some(label_id) if label_id != sandbox_id => {
                    warn!(
                        sandbox_id = %sandbox_id,
                        sandbox_name = %sandbox_name,
                        container = %agent_name,
                        label_sandbox_id = %label_id,
                        "Container label sandbox ID did not match delete request; cleaning up using request sandbox_id"
                    );
                }
                None => {
                    warn!(
                        sandbox_id = %sandbox_id,
                        sandbox_name = %sandbox_name,
                        container = %agent_name,
                        "Container missing '{}' label; cleaning up using request sandbox_id",
                        LABEL_SANDBOX_ID,
                    );
                }
                Some(_) => {}
            },
            Err(PodmanApiError::NotFound(_)) => {}
            Err(e) => return Err(ComputeDriverError::from(e)),
        }

        // Stop and remove agent container.
        let _ = self
            .client
            .stop_container(&agent_name, self.config.stop_timeout_secs)
            .await;

        let agent_existed = match self.client.remove_container(&agent_name).await {
            Ok(()) => true,
            Err(PodmanApiError::NotFound(_)) => false,
            Err(e) => return Err(ComputeDriverError::from(e)),
        };

        // Stop and remove proxy sidecar (best-effort).
        let _ = self.client.stop_container(&proxy_name, 5).await;
        let _ = self.client.remove_container(&proxy_name).await;

        // Remove volumes, secret, and network (all best-effort, log warnings).
        for (what, result) in [
            ("TLS volume", self.client.remove_volume(&tls_vol).await),
            (
                "workspace volume",
                self.client.remove_volume(&workspace_vol).await,
            ),
            ("secret", self.client.remove_secret(&sec_name).await),
            ("network", self.client.remove_network(&internal_net).await),
        ] {
            if let Err(e) = result {
                warn!(
                    sandbox_id = %sandbox_id,
                    sandbox_name = %sandbox_name,
                    resource = what,
                    error = %e,
                    "Failed to remove {what}"
                );
            }
        }

        Ok(agent_existed)
    }

    /// Check whether a sandbox container exists.
    pub async fn sandbox_exists(&self, sandbox_name: &str) -> Result<bool, ComputeDriverError> {
        let name = container::container_name(sandbox_name);
        match self.client.inspect_container(&name).await {
            Ok(_) => Ok(true),
            Err(PodmanApiError::NotFound(_)) => Ok(false),
            Err(e) => Err(ComputeDriverError::from(e)),
        }
    }

    /// Fetch a single sandbox by name.
    pub async fn get_sandbox(
        &self,
        sandbox_name: &str,
    ) -> Result<Option<DriverSandbox>, ComputeDriverError> {
        let name = container::container_name(sandbox_name);
        match self.client.inspect_container(&name).await {
            Ok(inspect) => Ok(driver_sandbox_from_inspect(&inspect)),
            Err(PodmanApiError::NotFound(_)) => Ok(None),
            Err(e) => Err(ComputeDriverError::from(e)),
        }
    }

    /// List all managed sandboxes.
    ///
    /// Only returns agent containers (not proxy sidecars). Containers without
    /// a role label (pre-sidecar containers) are included for backwards compat.
    /// Running containers are inspected for health check status; non-running
    /// containers are built directly from the list entry data.
    pub async fn list_sandboxes(&self) -> Result<Vec<DriverSandbox>, ComputeDriverError> {
        let entries = self
            .client
            .list_containers(LABEL_MANAGED_FILTER)
            .await
            .map_err(ComputeDriverError::from)?;

        // Filter to agent containers only. Containers without a role label
        // (created before the sidecar architecture) are included.
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|e| {
                e.labels
                    .get(LABEL_ROLE)
                    .map_or(true, |r| r == ROLE_AGENT)
            })
            .collect();

        let mut sandboxes = Vec::with_capacity(entries.len());
        for entry in &entries {
            if entry.state == "running" {
                // Running containers need inspect for health check status.
                match self.client.inspect_container(&entry.id).await {
                    Ok(inspect) => {
                        if let Some(sandbox) = driver_sandbox_from_inspect(&inspect) {
                            sandboxes.push(sandbox);
                            continue;
                        }
                    }
                    Err(e) => {
                        let name = entry.names.first().cloned().unwrap_or_default();
                        warn!(
                            container = %name,
                            error = %e,
                            "Failed to inspect running container during list, falling back to list entry"
                        );
                    }
                }
            }
            // Non-running containers (or inspect fallback): build from list data.
            if let Some(sandbox) = driver_sandbox_from_list_entry(entry) {
                sandboxes.push(sandbox);
            }
        }

        sandboxes.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
        Ok(sandboxes)
    }

    /// Start watching all managed sandbox containers.
    pub async fn watch_sandboxes(&self) -> Result<WatchStream, ComputeDriverError> {
        watcher::start_watch(self.client.clone())
            .await
            .map_err(ComputeDriverError::from)
    }
}

#[cfg(test)]
impl PodmanComputeDriver {
    pub(crate) fn for_tests(config: PodmanComputeConfig) -> Self {
        let client = PodmanClient::new(config.socket_path.clone());
        Self {
            client,
            config,
            network_gateway_ip: None,
        }
    }
}

/// Check whether the current user has subuid/subgid ranges configured.
///
/// Rootless Podman requires entries in `/etc/subuid` and `/etc/subgid` for
/// the running user. If missing, container creation fails with an obscure
/// error. This pre-flight check emits a warning to guide operators.
fn check_subuid_range() {
    let uid = nix::unistd::getuid().as_raw();
    let username = nix::unistd::User::from_uid(nix::unistd::Uid::from_raw(uid))
        .ok()
        .flatten()
        .map(|u| u.name);

    let has_range = |path: &str| -> bool {
        let Ok(content) = std::fs::read_to_string(path) else {
            return false;
        };
        let uid_str = uid.to_string();
        content.lines().any(|line| {
            let Some(entry) = line.split(':').next() else {
                return false;
            };
            entry == uid_str || username.as_deref() == Some(entry)
        })
    };

    if !has_range("/etc/subuid") || !has_range("/etc/subgid") {
        let user_display = username.as_deref().map_or_else(
            || format!("UID {uid}"),
            |name| format!("{name} (UID {uid})"),
        );
        warn!(
            user = %user_display,
            "Rootless Podman detected but no /etc/subuid or /etc/subgid entry found. \
             Container creation may fail. Add entries with: \
             sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(whoami)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::Full;
    use hyper::body::Bytes;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::net::UnixListener;

    #[test]
    fn podman_driver_error_from_conflict() {
        let err = ComputeDriverError::from(PodmanApiError::Conflict("exists".into()));
        assert!(matches!(err, ComputeDriverError::AlreadyExists));
    }

    #[test]
    fn podman_driver_error_from_not_found() {
        let err = ComputeDriverError::from(PodmanApiError::NotFound("gone".into()));
        assert!(matches!(err, ComputeDriverError::Message(_)));
    }

    // ── grpc_endpoint auto-detection ───────────────────────────────────
    //
    // PodmanComputeDriver::new() fills grpc_endpoint when it is empty.
    // The scheme (http vs https) depends on whether TLS client certs are
    // configured. These tests simulate the auto-detection logic.

    #[test]
    fn grpc_endpoint_http_without_tls() {
        let mut cfg = PodmanComputeConfig {
            gateway_port: 8081,
            ..PodmanComputeConfig::default()
        };
        if cfg.grpc_endpoint.is_empty() {
            let scheme = if cfg.tls_enabled() { "https" } else { "http" };
            cfg.grpc_endpoint = format!("{scheme}://host.containers.internal:{}", cfg.gateway_port);
        }
        assert_eq!(cfg.grpc_endpoint, "http://host.containers.internal:8081");
    }

    #[test]
    fn grpc_endpoint_https_with_tls() {
        let mut cfg = PodmanComputeConfig {
            gateway_port: 8080,
            guest_tls_ca: Some(PathBuf::from("/tls/ca.crt")),
            guest_tls_cert: Some(PathBuf::from("/tls/tls.crt")),
            guest_tls_key: Some(PathBuf::from("/tls/tls.key")),
            ..PodmanComputeConfig::default()
        };
        if cfg.grpc_endpoint.is_empty() {
            let scheme = if cfg.tls_enabled() { "https" } else { "http" };
            cfg.grpc_endpoint = format!("{scheme}://host.containers.internal:{}", cfg.gateway_port);
        }
        assert_eq!(cfg.grpc_endpoint, "https://host.containers.internal:8080");
    }

    #[test]
    fn partial_tls_config_returns_error() {
        let cfg = PodmanComputeConfig {
            gateway_port: 8080,
            guest_tls_ca: Some(PathBuf::from("/tls/ca.crt")),
            // guest_tls_cert and guest_tls_key not set — incomplete TLS config.
            ..PodmanComputeConfig::default()
        };
        assert!(!cfg.tls_enabled());
        let err = cfg
            .validate_tls_config()
            .expect_err("partial TLS config should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("OPENSHELL_PODMAN_TLS_CERT"),
            "error should name the missing cert: {msg}"
        );
        assert!(
            msg.contains("OPENSHELL_PODMAN_TLS_KEY"),
            "error should name the missing key: {msg}"
        );
    }

    #[test]
    fn explicit_grpc_endpoint_takes_precedence() {
        let mut cfg = PodmanComputeConfig {
            grpc_endpoint: "https://gateway.internal:9000".to_string(),
            gateway_port: 8081,
            ..PodmanComputeConfig::default()
        };
        if cfg.grpc_endpoint.is_empty() {
            let scheme = if cfg.tls_enabled() { "https" } else { "http" };
            cfg.grpc_endpoint = format!("{scheme}://host.containers.internal:{}", cfg.gateway_port);
        }
        assert_eq!(cfg.grpc_endpoint, "https://gateway.internal:9000");
    }

    #[derive(Clone)]
    struct StubResponse {
        status: StatusCode,
        body: String,
    }

    impl StubResponse {
        fn new(status: StatusCode, body: impl Into<String>) -> Self {
            Self {
                status,
                body: body.into(),
            }
        }
    }

    fn unique_socket_path(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        PathBuf::from(format!(
            "/tmp/openshell-podman-{test_name}-{}-{nanos}.sock",
            std::process::id()
        ))
    }

    fn test_driver(socket_path: PathBuf) -> PodmanComputeDriver {
        let config = PodmanComputeConfig {
            socket_path,
            stop_timeout_secs: 10,
            ..PodmanComputeConfig::default()
        };
        PodmanComputeDriver::for_tests(config)
    }

    fn api_path(path: &str) -> String {
        format!("/v5.0.0{path}")
    }

    fn spawn_podman_stub(
        test_name: &str,
        responses: Vec<StubResponse>,
    ) -> (
        PathBuf,
        Arc<Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let socket_path = unique_socket_path(test_name);
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let request_log = Arc::new(Mutex::new(Vec::new()));
        let response_queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let expected = response_queue
            .lock()
            .expect("response queue lock should not be poisoned")
            .len();
        let socket_path_for_task = socket_path.clone();
        let log_for_task = request_log.clone();
        let queue_for_task = response_queue;
        let handle = tokio::spawn(async move {
            for _ in 0..expected {
                let (stream, _) = listener.accept().await.expect("test stub should accept");
                let log = log_for_task.clone();
                let queue = queue_for_task.clone();
                let result = http1::Builder::new()
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(move |req| {
                            let log = log.clone();
                            let queue = queue.clone();
                            async move {
                                let path = req.uri().path_and_query().map_or_else(
                                    || req.uri().path().to_string(),
                                    |pq| pq.as_str().to_string(),
                                );
                                log.lock()
                                    .expect("request log lock should not be poisoned")
                                    .push(format!("{} {}", req.method(), path));
                                let response = queue
                                    .lock()
                                    .expect("response queue lock should not be poisoned")
                                    .pop_front()
                                    .expect("stub response should exist");
                                Ok::<_, Infallible>(
                                    Response::builder()
                                        .status(response.status)
                                        .body(Full::new(Bytes::from(response.body)))
                                        .expect("stub response should build"),
                                )
                            }
                        }),
                    )
                    .await;
                // The one-shot test client can close the Unix socket after the
                // response, which Hyper reports as a shutdown error. Let the
                // request log assertions below decide whether the stub served
                // the expected API calls.
                let _ = result;
            }
            let _ = std::fs::remove_file(&socket_path_for_task);
        });
        (socket_path, request_log, handle)
    }

    #[tokio::test]
    async fn delete_sandbox_cleans_up_with_request_id_when_container_is_already_gone() {
        let sandbox_id = "sandbox-123";
        let sandbox_name = "demo";
        let agent_name = container::container_name(sandbox_name);
        let proxy_name = container::proxy_container_name(sandbox_name);
        let tls_vol_name = container::tls_volume_name(sandbox_id);
        let workspace_vol_name = container::volume_name(sandbox_id);
        let secret_name = container::secret_name(sandbox_id);
        let internal_net = container::internal_network_name(sandbox_id);
        let (socket_path, request_log, handle) = spawn_podman_stub(
            "delete-not-found",
            vec![
                // 1. inspect agent → not found
                StubResponse::new(StatusCode::NOT_FOUND, r#"{"message":"gone"}"#),
                // 2. stop agent → not found
                StubResponse::new(StatusCode::NOT_FOUND, r#"{"message":"gone"}"#),
                // 3. remove agent → not found
                StubResponse::new(StatusCode::NOT_FOUND, r#"{"message":"gone"}"#),
                // 4. stop proxy → not found
                StubResponse::new(StatusCode::NOT_FOUND, r#"{"message":"gone"}"#),
                // 5. remove proxy → not found
                StubResponse::new(StatusCode::NOT_FOUND, r#"{"message":"gone"}"#),
                // 6. remove TLS volume
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 7. remove workspace volume
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 8. remove secret
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 9. remove network
                StubResponse::new(StatusCode::NO_CONTENT, ""),
            ],
        );
        let driver = test_driver(socket_path.clone());

        let deleted = driver
            .delete_sandbox(sandbox_id, sandbox_name)
            .await
            .expect("delete should succeed");

        assert!(!deleted, "missing container should report deleted=false");
        handle.await.expect("stub task should finish");
        let requests = request_log
            .lock()
            .expect("request log lock should not be poisoned")
            .clone();
        assert_eq!(
            requests,
            vec![
                format!(
                    "GET {}",
                    api_path(&format!("/libpod/containers/{agent_name}/json"))
                ),
                format!(
                    "POST {}",
                    api_path(&format!(
                        "/libpod/containers/{agent_name}/stop?timeout=10"
                    ))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!(
                        "/libpod/containers/{agent_name}?force=true&v=true"
                    ))
                ),
                format!(
                    "POST {}",
                    api_path(&format!("/libpod/containers/{proxy_name}/stop?timeout=5"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!(
                        "/libpod/containers/{proxy_name}?force=true&v=true"
                    ))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/volumes/{tls_vol_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/volumes/{workspace_vol_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/secrets/{secret_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/networks/{internal_net}"))
                ),
            ]
        );
        let _ = std::fs::remove_file(socket_path);
    }

    #[tokio::test]
    async fn delete_sandbox_uses_request_id_when_container_label_disagrees() {
        let sandbox_id = "sandbox-request-id";
        let sandbox_name = "demo";
        let agent_name = container::container_name(sandbox_name);
        let _proxy_name = container::proxy_container_name(sandbox_name);
        let tls_vol_name = container::tls_volume_name(sandbox_id);
        let workspace_vol_name = container::volume_name(sandbox_id);
        let secret_name = container::secret_name(sandbox_id);
        let internal_net = container::internal_network_name(sandbox_id);
        let inspect_body = serde_json::json!({
            "Id": "container-id",
            "Name": format!("/{agent_name}"),
            "State": {
                "Status": "running",
                "Running": true
            },
            "Config": {
                "Labels": {
                    LABEL_SANDBOX_ID: "sandbox-label-id"
                }
            }
        })
        .to_string();
        let (socket_path, request_log, handle) = spawn_podman_stub(
            "delete-mismatch",
            vec![
                // 1. inspect agent → label mismatch
                StubResponse::new(StatusCode::OK, inspect_body),
                // 2. stop agent
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 3. remove agent
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 4. stop proxy
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 5. remove proxy
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 6. remove TLS volume
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 7. remove workspace volume
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 8. remove secret
                StubResponse::new(StatusCode::NO_CONTENT, ""),
                // 9. remove network
                StubResponse::new(StatusCode::NO_CONTENT, ""),
            ],
        );
        let driver = test_driver(socket_path.clone());

        let deleted = driver
            .delete_sandbox(sandbox_id, sandbox_name)
            .await
            .expect("delete should succeed");

        assert!(deleted, "existing container should report deleted=true");
        handle.await.expect("stub task should finish");
        let requests = request_log
            .lock()
            .expect("request log lock should not be poisoned")
            .clone();
        // Verify the cleanup resources use the request sandbox_id.
        assert_eq!(
            requests[5..],
            [
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/volumes/{tls_vol_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/volumes/{workspace_vol_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/secrets/{secret_name}"))
                ),
                format!(
                    "DELETE {}",
                    api_path(&format!("/libpod/networks/{internal_net}"))
                ),
            ]
        );
        let _ = std::fs::remove_file(socket_path);
    }
}

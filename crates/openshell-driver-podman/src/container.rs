// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Container spec construction for the Podman driver.

use crate::config::PodmanComputeConfig;
use openshell_core::config::CDI_GPU_DEVICE_ALL;
use openshell_core::proto::compute::v1::DriverSandbox;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Returns `true` when `SELinux` is enabled (enforcing or permissive).
///
/// Checks whether selinuxfs is mounted, matching Podman's own detection
/// logic. Bind-mount relabeling (the `z` mount option) is needed in both
/// enforcing and permissive modes: enforcing blocks access outright, while
/// permissive floods the audit log with AVC denials that mask real issues.
///
/// On non-`SELinux` systems (Ubuntu, macOS, Alpine) the directory does not
/// exist and this returns `false`, leaving mount options unchanged.
#[cfg(target_os = "linux")]
fn is_selinux_enabled() -> bool {
    std::path::Path::new("/sys/fs/selinux").is_dir()
}

#[cfg(not(target_os = "linux"))]
fn is_selinux_enabled() -> bool {
    false
}

/// Label key for the sandbox ID.
pub const LABEL_SANDBOX_ID: &str = "openshell.sandbox-id";
/// Label key for the sandbox name.
pub const LABEL_SANDBOX_NAME: &str = "openshell.sandbox-name";
/// Label key for the sandbox namespace.
pub const LABEL_SANDBOX_NAMESPACE: &str = "openshell.sandbox-namespace";
/// Label applied to all managed containers.
pub const LABEL_MANAGED: &str = "openshell.managed";
/// Label filter string for list/event queries.
pub const LABEL_MANAGED_FILTER: &str = "openshell.managed=true";
/// Label key for the container role (proxy or agent) in the sidecar architecture.
pub const LABEL_ROLE: &str = "openshell.role";
/// Role value for the proxy sidecar container.
pub const ROLE_PROXY: &str = "proxy";
/// Role value for the agent container.
pub const ROLE_AGENT: &str = "agent";

/// Container name prefix to avoid collisions with user containers.
const CONTAINER_PREFIX: &str = "openshell-sandbox-";

/// Proxy sidecar container name prefix.
const PROXY_PREFIX: &str = "openshell-proxy-";

/// Volume name prefix.
const VOLUME_PREFIX: &str = "openshell-sandbox-";

/// Container-side mount paths for client TLS materials.
const TLS_CA_MOUNT_PATH: &str = "/etc/openshell/tls/client/ca.crt";
const TLS_CERT_MOUNT_PATH: &str = "/etc/openshell/tls/client/tls.crt";
const TLS_KEY_MOUNT_PATH: &str = "/etc/openshell/tls/client/tls.key";

/// Build a Podman container name from the sandbox name (used for the agent container).
#[must_use]
pub fn container_name(sandbox_name: &str) -> String {
    format!("{CONTAINER_PREFIX}{sandbox_name}")
}

/// Build the proxy sidecar container name from the sandbox name.
#[must_use]
pub fn proxy_container_name(sandbox_name: &str) -> String {
    format!("{PROXY_PREFIX}{sandbox_name}")
}

/// Build the workspace volume name from the sandbox ID.
#[must_use]
pub fn volume_name(sandbox_id: &str) -> String {
    format!("{VOLUME_PREFIX}{sandbox_id}-workspace")
}

/// Build the TLS volume name for mTLS material shared between proxy and agent.
#[must_use]
pub fn tls_volume_name(sandbox_id: &str) -> String {
    format!("openshell-tls-{sandbox_id}")
}

/// Build the internal network name for sandbox isolation.
#[must_use]
pub fn internal_network_name(sandbox_id: &str) -> String {
    format!("openshell-sbx-{sandbox_id}")
}

/// Podman secret name prefix.
const SECRET_PREFIX: &str = "openshell-handshake-";

/// Build the Podman secret name for a sandbox's SSH handshake secret.
#[must_use]
pub fn secret_name(sandbox_id: &str) -> String {
    format!("{SECRET_PREFIX}{sandbox_id}")
}

/// Truncate a container ID to 12 characters (standard short form).
#[must_use]
pub fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}

// ---------------------------------------------------------------------------
// Typed container spec structs for the Podman libpod create API.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ContainerSpec {
    name: String,
    image: String,
    labels: BTreeMap<String, String>,
    env: BTreeMap<String, String>,
    volumes: Vec<NamedVolume>,
    image_volumes: Vec<ImageVolume>,
    hostname: String,
    /// Overrides the image's ENTRYPOINT. In Podman's libpod API, `command`
    /// only overrides CMD (appended as args to the entrypoint). We must set
    /// `entrypoint` explicitly so the supervisor binary runs directly,
    /// regardless of what ENTRYPOINT the sandbox image defines.
    entrypoint: Vec<String>,
    command: Vec<String>,
    user: String,
    cap_drop: Vec<String>,
    cap_add: Vec<String>,
    no_new_privileges: bool,
    seccomp_profile_path: String,
    image_pull_policy: String,
    healthconfig: HealthConfig,
    resource_limits: ResourceLimits,
    /// Env-type secrets: map of `ENV_VAR_NAME → secret_name`.
    /// Podman's libpod `SpecGenerator` uses `secret_env` (a flat map) for
    /// environment-variable injection, distinct from `secrets` which only
    /// handles file-mounted secrets under `/run/secrets/`.
    secret_env: BTreeMap<String, String>,
    stop_timeout: u32,
    /// Extra /etc/hosts entries. Used to inject `host.containers.internal`
    /// via Podman's `host-gateway` magic so sandbox containers can reach
    /// the gateway server running on the host in rootless mode.
    hostadd: Vec<String>,
    netns: NetNS,
    networks: BTreeMap<String, NetworkAttachment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    devices: Option<Vec<LinuxDevice>>,
    /// Extra mounts for the libpod `SpecGenerator` (e.g. tmpfs entries).
    mounts: Vec<Mount>,
    /// Port mappings from host to container. Using `host_port=0` requests an
    /// ephemeral port, readable back from the inspect response.
    portmappings: Vec<PortMapping>,
    /// Custom DNS servers for the container. Maps to Podman's `dns_server`
    /// field in the libpod `SpecGenerator`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    dns_server: Vec<String>,
    /// Paths to unmask in /proc and /sys for nested-container support.
    ///
    /// Podman's libpod `SpecGenerator` `unmask` field accepts a list of paths
    /// to expose inside the container that are ordinarily masked/read-only
    /// for security. Required for nested podman to read `/proc/*` and manage
    /// cgroups under `/sys/fs/cgroup`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unmask: Vec<String>,
    /// `SELinux` process label options.
    ///
    /// The libpod `SpecGenerator` `selinux_opts` field corresponds to the
    /// CLI `--security-opt label=<value>` flag. Setting `["disable"]` is
    /// equivalent to `--security-opt label=disable`, which disables `SELinux`
    /// confinement for the container — necessary for nested containers to
    /// bind-mount host paths and manage their own overlayfs layers.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    selinux_opts: Vec<String>,
}

/// A port mapping entry for the libpod `SpecGenerator`.
#[derive(Serialize)]
struct PortMapping {
    host_port: u16,
    container_port: u16,
    protocol: String,
}

/// A mount entry for the libpod container create API `mounts` field.
///
/// Unlike `volumes` (named Podman volumes) or `image_volumes` (OCI image
/// mounts resolved at the libpod layer), these mounts are passed to the
/// libpod `SpecGenerator` and support arbitrary mount types (e.g. tmpfs).
/// Field names must be lowercase to match the libpod JSON schema.
#[derive(Serialize)]
struct Mount {
    #[serde(rename = "type")]
    kind: String,
    source: String,
    destination: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    options: Vec<String>,
}

/// A Podman image volume for the libpod container create API.
///
/// Image volumes mount an OCI image's filesystem into a container without
/// running it. Podman resolves these at the libpod layer before generating
/// the OCI runtime spec, unlike `mounts` which are passed directly to the
/// OCI runtime (crun/runc).
#[derive(Serialize)]
struct ImageVolume {
    source: String,
    destination: String,
    rw: bool,
}

#[derive(Serialize)]
struct NamedVolume {
    name: String,
    dest: String,
    options: Vec<String>,
}

#[derive(Serialize)]
struct HealthConfig {
    test: Vec<String>,
    #[serde(rename = "Interval")]
    interval: u64,
    #[serde(rename = "Timeout")]
    timeout: u64,
    #[serde(rename = "Retries")]
    retries: u32,
    #[serde(rename = "StartPeriod")]
    start_period: u64,
}

#[derive(Serialize)]
struct ResourceLimits {
    cpu: CpuLimits,
    memory: MemoryLimits,
}

#[derive(Serialize)]
struct CpuLimits {
    quota: u64,
    period: u64,
}

#[derive(Serialize)]
struct MemoryLimits {
    limit: u64,
}

#[derive(Serialize)]
struct NetNS {
    nsmode: String,
}

#[derive(Serialize)]
struct NetworkAttachment {
    /// Static IP address for this network attachment.
    ///
    /// Note: static IPs on multi-network containers require `podman network
    /// connect --ip` at runtime; this field is reserved for future use with
    /// single-network containers or post-connect workflows.
    #[serde(skip_serializing_if = "Option::is_none")]
    static_ip: Option<String>,
}

#[derive(Serialize)]
struct LinuxDevice {
    path: String,
}

/// Default limits: 2 CPU cores (200000µs quota / 100000µs period), 4 GiB memory.
const DEFAULT_CPU_QUOTA: u64 = 200_000;
const DEFAULT_CPU_PERIOD: u64 = 100_000;
const DEFAULT_MEMORY_LIMIT: u64 = 4_294_967_296; // 4 GiB

/// Resolve the OCI image reference for a sandbox, using the template image
/// if provided, otherwise the driver's default image.
#[must_use]
pub fn resolve_image<'a>(sandbox: &'a DriverSandbox, config: &'a PodmanComputeConfig) -> &'a str {
    let spec = sandbox.spec.as_ref();
    let template = spec.and_then(|s| s.template.as_ref());
    template
        .map(|t| t.image.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(&config.default_image)
}

/// Merge environment variables from user spec/template with required driver vars.
///
/// User-supplied vars are inserted first so that the required driver
/// vars always win -- preventing spec/template overrides of security-
/// critical values like `OPENSHELL_ENDPOINT` or `OPENSHELL_SANDBOX_ID`.
fn build_env(
    sandbox: &DriverSandbox,
    config: &PodmanComputeConfig,
    image: &str,
) -> BTreeMap<String, String> {
    let spec = sandbox.spec.as_ref();
    let template = spec.and_then(|s| s.template.as_ref());

    let mut env: BTreeMap<String, String> = BTreeMap::new();

    // 1. User-supplied environment (lowest priority).
    if let Some(s) = spec {
        if !s.log_level.is_empty() {
            env.insert("OPENSHELL_LOG_LEVEL".into(), s.log_level.clone());
        }
        for (k, v) in &s.environment {
            env.insert(k.clone(), v.clone());
        }
    }
    if let Some(t) = template {
        for (k, v) in &t.environment {
            env.insert(k.clone(), v.clone());
        }
    }

    // 2. Required driver vars (highest priority -- always overwrite).
    env.insert("OPENSHELL_SANDBOX".into(), sandbox.name.clone());
    env.insert("OPENSHELL_SANDBOX_ID".into(), sandbox.id.clone());
    env.insert("OPENSHELL_ENDPOINT".into(), config.grpc_endpoint.clone());
    env.insert(
        "OPENSHELL_SSH_SOCKET_PATH".into(),
        config.sandbox_ssh_socket_path.clone(),
    );
    // NOTE: The SSH handshake secret is injected via a Podman secret
    // (see the "secrets" field below) rather than a plaintext env var.
    // This prevents exposure through `podman inspect`.
    env.insert(
        "OPENSHELL_SSH_HANDSHAKE_SKEW_SECS".into(),
        config.ssh_handshake_skew_secs.to_string(),
    );
    env.insert("OPENSHELL_CONTAINER_IMAGE".into(), image.to_string());
    env.insert("OPENSHELL_SANDBOX_COMMAND".into(), "sleep infinity".into());

    // 3. TLS client cert paths (when mTLS is enabled). These point to
    //    the container-side mount paths where the cert files are
    //    bind-mounted from the host.
    if config.tls_enabled() {
        env.insert("OPENSHELL_TLS_CA".into(), TLS_CA_MOUNT_PATH.into());
        env.insert("OPENSHELL_TLS_CERT".into(), TLS_CERT_MOUNT_PATH.into());
        env.insert("OPENSHELL_TLS_KEY".into(), TLS_KEY_MOUNT_PATH.into());
    }

    env
}

/// Merge labels from the sandbox template with required managed labels.
///
/// User-supplied labels are inserted first so that the managed labels
/// always win -- preventing template overrides of internal tracking labels.
fn build_labels(sandbox: &DriverSandbox) -> BTreeMap<String, String> {
    let template = sandbox.spec.as_ref().and_then(|s| s.template.as_ref());

    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    if let Some(t) = template {
        for (k, v) in &t.labels {
            labels.insert(k.clone(), v.clone());
        }
    }
    // Managed labels (highest priority -- always overwrite).
    labels.insert(LABEL_SANDBOX_ID.into(), sandbox.id.clone());
    labels.insert(LABEL_SANDBOX_NAME.into(), sandbox.name.clone());
    labels.insert(LABEL_SANDBOX_NAMESPACE.into(), sandbox.namespace.clone());
    labels.insert(LABEL_MANAGED.into(), "true".into());

    labels
}

/// Parse resource limits from the sandbox template, falling back to defaults.
fn build_resource_limits(sandbox: &DriverSandbox) -> ResourceLimits {
    let resources = sandbox
        .spec
        .as_ref()
        .and_then(|s| s.template.as_ref())
        .and_then(|t| t.resources.as_ref());

    let cpu_micros = resources
        .filter(|r| !r.cpu_limit.is_empty())
        .and_then(|r| parse_cpu_to_microseconds(&r.cpu_limit))
        .unwrap_or(DEFAULT_CPU_QUOTA);

    let mem_bytes = resources
        .filter(|r| !r.memory_limit.is_empty())
        .and_then(|r| parse_memory_to_bytes(&r.memory_limit))
        .unwrap_or(DEFAULT_MEMORY_LIMIT);

    ResourceLimits {
        cpu: CpuLimits {
            quota: cpu_micros,
            period: DEFAULT_CPU_PERIOD,
        },
        memory: MemoryLimits { limit: mem_bytes },
    }
}

/// Build CDI GPU device list if GPU is requested.
fn build_devices(sandbox: &DriverSandbox) -> Option<Vec<LinuxDevice>> {
    if sandbox.spec.as_ref().is_some_and(|s| s.gpu) {
        Some(vec![LinuxDevice {
            path: CDI_GPU_DEVICE_ALL.into(),
        }])
    } else {
        None
    }
}

/// Determine whether the sandbox should be created in nested (passthrough) mode.
///
/// Nested mode runs the supervisor for SSH/exec but skips all security
/// enforcement (Landlock, seccomp, netns, privilege dropping). The container
/// runs with elevated capabilities needed for nested containerization.
///
/// Returns `true` when the template's `mode` field is set to
/// `SANDBOX_MODE_NESTED` (2), indicating the sandbox is intended for
/// inner-container workloads (e.g. running Podman inside the sandbox).
pub(crate) fn is_nested_mode(sandbox: &DriverSandbox) -> bool {
    sandbox
        .spec
        .as_ref()
        .and_then(|s| s.template.as_ref())
        .is_some_and(|t| t.mode == 2)
}

/// Build the Podman container creation JSON spec.
///
/// This function only handles passthrough (nested) mode. For supervised mode,
/// callers must use [`build_proxy_sidecar_spec`] and [`build_agent_container_spec`]
/// directly to produce the two-container sidecar topology.
///
/// # Panics
///
/// Panics if called for a non-nested sandbox. Supervised mode requires the
/// sidecar API (`build_proxy_sidecar_spec` + `build_agent_container_spec`).
#[must_use]
pub fn build_container_spec(sandbox: &DriverSandbox, config: &PodmanComputeConfig) -> Value {
    assert!(
        is_nested_mode(sandbox),
        "use build_proxy_sidecar_spec + build_agent_container_spec for supervised mode"
    );
    build_container_spec_passthrough(sandbox, config)
}

/// Build a nested container spec: supervisor injected for SSH/exec, but no
/// inner sandboxing (no Landlock, no seccomp, no netns, no privilege dropping).
///
/// The container runs with the kernel permissions needed for nested
/// containerization (`unmask`, `selinux_opts`, `no_new_privileges=false`).
/// The supervisor binary is side-loaded and provides SSH access, but tells
/// the sandbox runtime to skip all security enforcement via `OPENSHELL_MODE=nested`.
fn build_container_spec_passthrough(
    sandbox: &DriverSandbox,
    config: &PodmanComputeConfig,
) -> Value {
    let image = resolve_image(sandbox, config);
    let name = container_name(&sandbox.name);
    let vol = volume_name(&sandbox.id);

    // Use the full supervisor env but add OPENSHELL_MODE=nested to signal
    // the supervisor to skip security enforcement.
    let mut env = build_env(sandbox, config, image);
    env.insert("OPENSHELL_MODE".into(), "nested".into());

    // Inject GOOGLE_APPLICATION_CREDENTIALS for ADC support.
    if config.adc_host_path.is_some() {
        env.entry("GOOGLE_APPLICATION_CREDENTIALS".into())
            .or_insert_with(|| "/run/gcloud/adc.json".into());
    }

    let labels = build_labels(sandbox);
    let resource_limits = build_resource_limits(sandbox);
    let devices = build_devices(sandbox);

    let mut networks = BTreeMap::new();
    networks.insert(config.network_name.clone(), NetworkAttachment { static_ip: None });

    // Build mounts: TLS bind-mounts (if configured) + ADC bind-mount (if configured)
    // + tmpfs for /run/netns (needed by the supervisor for network namespace operations).
    let mut mounts = Vec::new();
    if let Some(host_path) = &config.adc_host_path {
        mounts.push(Mount {
            kind: "bind".into(),
            source: host_path.to_string_lossy().into_owned(),
            destination: "/run/gcloud/adc.json".into(),
            options: vec!["ro".into(), "rbind".into()],
        });
    }
    mounts.push(Mount {
        kind: "tmpfs".into(),
        source: "tmpfs".into(),
        destination: "/run/netns".into(),
        options: vec!["rw".into(), "nosuid".into(), "nodev".into()],
    });
    // Bind-mount client TLS materials into the container when mTLS is enabled.
    if let (Some(ca), Some(cert), Some(key)) = (
        &config.guest_tls_ca,
        &config.guest_tls_cert,
        &config.guest_tls_key,
    ) {
        let mut ro = vec!["ro".into(), "rbind".into()];
        if is_selinux_enabled() {
            ro.push("z".into());
        }
        mounts.push(Mount {
            kind: "bind".into(),
            source: ca.display().to_string(),
            destination: TLS_CA_MOUNT_PATH.into(),
            options: ro.clone(),
        });
        mounts.push(Mount {
            kind: "bind".into(),
            source: cert.display().to_string(),
            destination: TLS_CERT_MOUNT_PATH.into(),
            options: ro.clone(),
        });
        mounts.push(Mount {
            kind: "bind".into(),
            source: key.display().to_string(),
            destination: TLS_KEY_MOUNT_PATH.into(),
            options: ro,
        });
    }

    let container_spec = ContainerSpec {
        name,
        image: image.to_string(),
        labels,
        env,
        volumes: vec![NamedVolume {
            name: vol,
            dest: "/sandbox".into(),
            options: vec!["rw".into()],
        }],
        // Side-load the supervisor binary, same as supervised mode.
        image_volumes: vec![ImageVolume {
            source: config.supervisor_image.clone(),
            destination: "/opt/openshell/bin".into(),
            rw: false,
        }],
        hostname: format!("sandbox-{}", sandbox.name),
        // Run the supervisor as the entrypoint so SSH/exec are available.
        entrypoint: vec!["/opt/openshell/bin/openshell-sandbox".into()],
        command: vec![],
        // Run as root so nested container tools (podman, buildah) have the
        // permissions needed to manage storage, create namespaces, and call
        // newuidmap/newgidmap for rootless inner containers.
        user: "0:0".into(),
        // Minimal capability drop for nested mode. We keep SYS_ADMIN and
        // NET_ADMIN for nested podman and drop only clearly unnecessary caps.
        cap_drop: vec!["NET_BIND_SERVICE".into(), "NET_RAW".into()],
        cap_add: vec![
            // Required for nested container runtimes (unshare, mount, etc.).
            "SYS_ADMIN".into(),
            // Required for inner network namespace creation.
            "NET_ADMIN".into(),
        ],
        // Must be false: nested container runtimes need privilege transitions
        // (e.g. newuidmap/newgidmap for rootless inner containers).
        no_new_privileges: false,
        // Outer seccomp must be unconfined so the inner runtime can install
        // its own seccomp filter and use mount/clone/unshare syscalls.
        seccomp_profile_path: "unconfined".into(),
        image_pull_policy: config.image_pull_policy.as_str().to_string(),
        // Health check: supervisor SSH socket or port ready.
        healthconfig: HealthConfig {
            test: vec![
                "CMD-SHELL".into(),
                format!(
                    "test -e /var/run/openshell-ssh-ready || test -S {} || ss -tlnp | grep -q :{}",
                    config.sandbox_ssh_socket_path, config.ssh_port
                ),
            ],
            interval: 3_000_000_000,
            timeout: 2_000_000_000,
            retries: 10,
            start_period: 5_000_000_000,
        },
        resource_limits,
        // Inject the SSH handshake secret via Podman's secret_env map.
        secret_env: BTreeMap::from([(
            "OPENSHELL_SSH_HANDSHAKE_SECRET".into(),
            secret_name(&sandbox.id),
        )]),
        stop_timeout: config.stop_timeout_secs,
        hostadd: vec![
            "host.containers.internal:host-gateway".into(),
            "host.openshell.internal:host-gateway".into(),
        ],
        netns: NetNS {
            nsmode: "bridge".to_string(),
        },
        networks,
        // Add /dev/fuse so inner rootless Podman can use fuse-overlayfs as
        // its storage driver. Without this, rootless containers inside the
        // sandbox fall back to vfs (slow/large) or fail entirely on kernels
        // that don't support native overlayfs in user namespaces.
        devices: {
            let mut devs = devices.unwrap_or_default();
            devs.push(LinuxDevice {
                path: "/dev/fuse".into(),
            });
            Some(devs)
        },
        mounts,
        // Publish the SSH port with host_port=0 to get an ephemeral host port.
        portmappings: vec![PortMapping {
            host_port: 0,
            container_port: config.ssh_port,
            protocol: "tcp".into(),
        }],
        dns_server: vec![],
        // Unmask /proc/* and /sys/fs/cgroup so the inner container runtime
        // can read process info and manage cgroups. Without these, Podman
        // inside the container fails when trying to inspect or create containers.
        unmask: vec!["/proc/*".into(), "/sys/fs/cgroup".into()],
        // Disable SELinux confinement for the container. Inner container
        // runtimes need to create bind mounts and manage overlayfs layers;
        // SELinux label enforcement blocks these on SELinux-enabled hosts.
        selinux_opts: vec!["disable".into()],
    };

    serde_json::to_value(container_spec).expect("ContainerSpec serialization cannot fail")
}

/// Proxy sidecar resource limits: 1 CPU core, 512 MiB memory.
const PROXY_CPU_QUOTA: u64 = 100_000;
const PROXY_MEMORY_LIMIT: u64 = 536_870_912; // 512 MiB

/// Build the proxy sidecar container spec for the sidecar architecture.
///
/// The proxy container runs the supervisor in `proxy` mode, handling all
/// network enforcement (egress filtering, mTLS termination). It is created
/// on the internal network first; the driver connects it to the bridge
/// network via `network_connect` after creation.
#[must_use]
pub fn build_proxy_sidecar_spec(
    sandbox: &DriverSandbox,
    config: &PodmanComputeConfig,
    internal_network: &str,
    host_dns_servers: &[String],
) -> Value {
    let name = proxy_container_name(&sandbox.name);
    let tls_vol = tls_volume_name(&sandbox.id);

    let mut labels = build_labels(sandbox);
    labels.insert(LABEL_ROLE.into(), ROLE_PROXY.into());

    // Proxy env: minimal set — the proxy doesn't run user workloads.
    let mut env: BTreeMap<String, String> = BTreeMap::new();

    // Propagate log level from spec if set.
    if let Some(spec) = sandbox.spec.as_ref() {
        if !spec.log_level.is_empty() {
            env.insert("OPENSHELL_LOG_LEVEL".into(), spec.log_level.clone());
        }
    }

    env.insert("OPENSHELL_MODE".into(), "proxy".into());
    env.insert("OPENSHELL_ENDPOINT".into(), config.grpc_endpoint.clone());
    env.insert("OPENSHELL_SANDBOX_ID".into(), sandbox.id.clone());
    env.insert("OPENSHELL_SANDBOX".into(), sandbox.name.clone());
    env.insert(
        "OPENSHELL_SSH_HANDSHAKE_SKEW_SECS".into(),
        config.ssh_handshake_skew_secs.to_string(),
    );

    // Internal network only (bridge added later via network_connect).
    #[allow(clippy::zero_sized_map_values)]
    let mut networks = BTreeMap::new();
    networks.insert(
        internal_network.to_string(),
        NetworkAttachment { static_ip: None },
    );

    // The proxy sidecar uses the same sandbox base image as the agent so it has
    // a working dynamic linker, CA certs, and common tools (curl for health checks).
    // The supervisor binary is sideloaded from the supervisor OCI image via
    // image_volumes, same as the agent container.
    let proxy_base_image = resolve_image(sandbox, config);

    let container_spec = ContainerSpec {
        name,
        image: proxy_base_image.to_string(),
        labels,
        env,
        volumes: vec![NamedVolume {
            name: tls_vol,
            dest: "/openshell-tls".into(),
            options: vec!["rw".into()],
        }],
        // Side-load the supervisor binary from the supervisor OCI image.
        image_volumes: vec![ImageVolume {
            source: config.supervisor_image.clone(),
            destination: "/opt/openshell/bin".into(),
            rw: false,
        }],
        hostname: format!("proxy-{}", sandbox.name),
        entrypoint: vec!["/opt/openshell/bin/openshell-sandbox".into()],
        command: vec![],
        user: "0:0".into(),
        // Minimal capability set for the proxy: no /proc scanning needed.
        cap_drop: vec![
            "DAC_OVERRIDE".into(),
            "FSETID".into(),
            "KILL".into(),
            "NET_BIND_SERVICE".into(),
            "NET_RAW".into(),
            "SETFCAP".into(),
            "SETPCAP".into(),
            "SYS_CHROOT".into(),
        ],
        cap_add: vec![
            // Potential seccomp/Landlock in future, namespace creation.
            "SYS_ADMIN".into(),
            // Network configuration for proxy operations.
            "NET_ADMIN".into(),
            // No SYS_PTRACE: proxy does not scan /proc.
            // No DAC_READ_SEARCH: proxy does not read /proc/<pid>/fd/.
        ],
        no_new_privileges: true,
        seccomp_profile_path: "unconfined".into(),
        image_pull_policy: config.image_pull_policy.as_str().to_string(),
        // Health: check for the TLS ready sentinel written by the proxy after
        // mTLS material is generated.
        healthconfig: HealthConfig {
            test: vec![
                "CMD-SHELL".into(),
                "test -e /openshell-tls/.ready".into(),
            ],
            interval: 3_000_000_000,
            timeout: 2_000_000_000,
            retries: 10,
            start_period: 5_000_000_000,
        },
        resource_limits: ResourceLimits {
            cpu: CpuLimits {
                quota: PROXY_CPU_QUOTA,
                period: DEFAULT_CPU_PERIOD,
            },
            memory: MemoryLimits {
                limit: PROXY_MEMORY_LIMIT,
            },
        },
        // SSH handshake secret for proxy-side authentication.
        secret_env: BTreeMap::from([(
            "OPENSHELL_SSH_HANDSHAKE_SECRET".into(),
            secret_name(&sandbox.id),
        )]),
        stop_timeout: config.stop_timeout_secs,
        hostadd: vec!["host.containers.internal:host-gateway".into()],
        netns: NetNS {
            nsmode: "bridge".to_string(),
        },
        networks,
        devices: None,
        mounts: vec![],
        // No port mappings: proxy is only reachable from the internal network.
        portmappings: vec![],
        dns_server: host_dns_servers.to_vec(),
        unmask: vec![],
        selinux_opts: vec![],
    };

    serde_json::to_value(container_spec).expect("ContainerSpec serialization cannot fail")
}

/// Build the agent container spec for the sidecar architecture.
///
/// The agent container runs the user sandbox image with the supervisor
/// side-loaded. It connects only to the internal network (no bridge) and
/// routes all external traffic through the proxy sidecar via HTTP_PROXY.
#[must_use]
pub fn build_agent_container_spec(
    sandbox: &DriverSandbox,
    config: &PodmanComputeConfig,
    internal_network: &str,
    proxy_ip: &str,
    tls_volume: &str,
) -> Value {
    let image = resolve_image(sandbox, config);
    let name = container_name(&sandbox.name);
    let workspace_vol = volume_name(&sandbox.id);

    let mut env = build_env(sandbox, config, image);

    // Sidecar mode: supervisor skips network enforcement (proxy handles it).
    env.insert("OPENSHELL_PROXY_MODE".into(), "sidecar".into());

    // Route gRPC traffic through the proxy's TCP forwarder.
    env.insert(
        "OPENSHELL_ENDPOINT".into(),
        format!("http://{proxy_ip}:8081"),
    );

    // HTTP proxy env vars for all outbound traffic.
    let proxy_url = format!("http://{proxy_ip}:3128");
    env.insert("HTTP_PROXY".into(), proxy_url.clone());
    env.insert("HTTPS_PROXY".into(), proxy_url);
    // Include the proxy IP in NO_PROXY so gRPC traffic to the TCP forwarder
    // (on proxy_ip:8081) is NOT double-proxied through the L7 proxy on :3128.
    let no_proxy = format!("127.0.0.1,localhost,::1,{proxy_ip}");
    env.insert("NO_PROXY".into(), no_proxy.clone());
    env.insert("no_proxy".into(), no_proxy);

    // CA trust env vars pointing to the proxy's CA certificate on the shared
    // TLS volume. The proxy writes openshell-ca.pem (just the CA) and
    // ca-bundle.pem (system CAs + proxy CA combined).
    let ca_bundle = "/openshell-tls/ca-bundle.pem";
    let ca_cert = "/openshell-tls/openshell-ca.pem";
    env.insert("SSL_CERT_FILE".into(), ca_bundle.into());
    env.insert("NODE_EXTRA_CA_CERTS".into(), ca_cert.into());
    env.insert("CURL_CA_BUNDLE".into(), ca_bundle.into());
    env.insert("GIT_SSL_CAINFO".into(), ca_bundle.into());
    env.insert("REQUESTS_CA_BUNDLE".into(), ca_bundle.into());

    // Block SSH to force HTTPS usage.
    env.insert(
        "GIT_SSH_COMMAND".into(),
        "echo 'SSH disabled — use HTTPS' && exit 1".into(),
    );

    let mut labels = build_labels(sandbox);
    labels.insert(LABEL_ROLE.into(), ROLE_AGENT.into());

    let resource_limits = build_resource_limits(sandbox);
    let devices = build_devices(sandbox);

    // Internal network only — no bridge connection (this is the isolation boundary).
    #[allow(clippy::zero_sized_map_values)]
    let mut networks = BTreeMap::new();
    networks.insert(
        internal_network.to_string(),
        NetworkAttachment { static_ip: None },
    );

    let container_spec = ContainerSpec {
        name,
        image: image.to_string(),
        labels,
        env,
        volumes: vec![
            NamedVolume {
                name: workspace_vol,
                dest: "/sandbox".into(),
                options: vec!["rw".into()],
            },
            NamedVolume {
                name: tls_volume.to_string(),
                dest: "/openshell-tls".into(),
                // Read-only: only the proxy writes TLS material.
                options: vec!["ro".into()],
            },
        ],
        // Side-load the supervisor binary from the supervisor OCI image.
        image_volumes: vec![ImageVolume {
            source: config.supervisor_image.clone(),
            destination: "/opt/openshell/bin".into(),
            rw: false,
        }],
        hostname: format!("sandbox-{}", sandbox.name),
        entrypoint: vec!["/opt/openshell/bin/openshell-sandbox".into()],
        command: vec![],
        user: "0:0".into(),
        // Reduced capability set: no SYS_PTRACE or DAC_READ_SEARCH needed
        // in sidecar mode (no /proc scanning — proxy handles process identity).
        cap_drop: vec![
            "DAC_OVERRIDE".into(),
            "FSETID".into(),
            "KILL".into(),
            "NET_BIND_SERVICE".into(),
            "NET_RAW".into(),
            "SETFCAP".into(),
            "SETPCAP".into(),
            "SYS_CHROOT".into(),
        ],
        cap_add: vec![
            // Namespace creation, Landlock setup.
            "SYS_ADMIN".into(),
            // Network configuration (if needed for local netns).
            "NET_ADMIN".into(),
            // Kernel log reading for bypass-detection diagnostics.
            "SYSLOG".into(),
            // No SYS_PTRACE: not needed in sidecar mode.
            // No DAC_READ_SEARCH: not needed in sidecar mode.
        ],
        no_new_privileges: true,
        seccomp_profile_path: "unconfined".into(),
        image_pull_policy: config.image_pull_policy.as_str().to_string(),
        healthconfig: HealthConfig {
            test: vec![
                "CMD-SHELL".into(),
                format!(
                    "test -e /var/run/openshell-ssh-ready || test -S {} || ss -tlnp | grep -q :{}",
                    config.sandbox_ssh_socket_path, config.ssh_port
                ),
            ],
            interval: 3_000_000_000,
            timeout: 2_000_000_000,
            retries: 10,
            start_period: 5_000_000_000,
        },
        resource_limits,
        // SSH handshake secret for agent-side authentication.
        secret_env: BTreeMap::from([(
            "OPENSHELL_SSH_HANDSHAKE_SECRET".into(),
            secret_name(&sandbox.id),
        )]),
        stop_timeout: config.stop_timeout_secs,
        // Inject stable host aliases into /etc/hosts so sandbox containers can
        // reach services on the host. `host.openshell.internal` is the driver-
        // neutral alias used by policies and e2e tests.
        hostadd: vec![
            "host.containers.internal:host-gateway".into(),
            "host.openshell.internal:host-gateway".into(),
        ],
        netns: NetNS {
            nsmode: "bridge".to_string(),
        },
        networks,
        devices,
        // Mount a tmpfs at /run/netns so the sandbox supervisor can create
        // named network namespaces via `ip netns add`. The `ip` command requires
        // /run/netns to exist and be bind-mountable; in rootless Podman this
        // directory does not exist on the host, so the mkdir inside the container
        // fails with EPERM. A private tmpfs gives the supervisor its own writable
        // /run/netns without needing host filesystem access.
        mounts: {
            let mut m = vec![Mount {
                kind: "tmpfs".into(),
                source: "tmpfs".into(),
                destination: "/run/netns".into(),
                options: vec!["rw".into(), "nosuid".into(), "nodev".into()],
            }];
            // Bind-mount client TLS materials into the container when mTLS
            // is enabled. The supervisor reads these via OPENSHELL_TLS_CA,
            // OPENSHELL_TLS_CERT, and OPENSHELL_TLS_KEY env vars (set in
            // build_env above) to establish an mTLS connection back to the
            // gateway.
            if let (Some(ca), Some(cert), Some(key)) = (
                &config.guest_tls_ca,
                &config.guest_tls_cert,
                &config.guest_tls_key,
            ) {
                let mut ro = vec!["ro".into(), "rbind".into()];
                // On SELinux-enabled systems (Fedora, RHEL), bind-mounted
                // files need the shared relabel option so the container
                // process can read them through the SELinux MAC policy.
                if is_selinux_enabled() {
                    ro.push("z".into());
                }
                m.push(Mount {
                    kind: "bind".into(),
                    source: ca.display().to_string(),
                    destination: TLS_CA_MOUNT_PATH.into(),
                    options: ro.clone(),
                });
                m.push(Mount {
                    kind: "bind".into(),
                    source: cert.display().to_string(),
                    destination: TLS_CERT_MOUNT_PATH.into(),
                    options: ro.clone(),
                });
                m.push(Mount {
                    kind: "bind".into(),
                    source: key.display().to_string(),
                    destination: TLS_KEY_MOUNT_PATH.into(),
                    options: ro,
                });
            }
            m
        },
        // Publish the SSH port with host_port=0 to get an ephemeral host port.
        // In rootless Podman the bridge network (10.89.x.x) is not routable from
        // the host, so we must use the published host port on 127.0.0.1 instead.
        portmappings: vec![PortMapping {
            host_port: 0,
            container_port: config.ssh_port,
            protocol: "tcp".into(),
        }],
        // No DNS override: internal DNS resolves container names; external
        // DNS is irrelevant since all traffic goes through the proxy.
        dns_server: vec![],
        unmask: vec![],
        selinux_opts: vec![],
    };

    serde_json::to_value(container_spec).expect("ContainerSpec serialization cannot fail")
}

/// Parse a Kubernetes-style CPU quantity to cgroup quota microseconds
/// (for a 100ms period).
///
/// Examples: `"500m"` → 50000, `"2"` → 200000, `"0.5"` → 50000.
fn parse_cpu_to_microseconds(quantity: &str) -> Option<u64> {
    let micros = if let Some(millis_str) = quantity.strip_suffix('m') {
        let millis: u64 = millis_str.parse().ok()?;
        // quota = millis * period / 1000
        millis.checked_mul(100)?
    } else {
        let cores: f64 = quantity.parse().ok()?;
        if cores <= 0.0 || cores.is_nan() || cores.is_infinite() {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let val = (cores * 100_000.0) as u64;
        val
    };
    // A quota of 0 microseconds is invalid — treat as no limit.
    if micros == 0 { None } else { Some(micros) }
}

/// Parse a Kubernetes-style memory quantity to bytes.
///
/// Supports: `Ki`, `Mi`, `Gi`, `Ti` (binary) and `k`, `M`, `G`, `T`
/// (decimal), as well as plain byte values.
fn parse_memory_to_bytes(quantity: &str) -> Option<u64> {
    let suffixes: &[(&str, u64)] = &[
        ("Ti", 1024 * 1024 * 1024 * 1024),
        ("Gi", 1024 * 1024 * 1024),
        ("Mi", 1024 * 1024),
        ("Ki", 1024),
        ("T", 1_000_000_000_000),
        ("G", 1_000_000_000),
        ("M", 1_000_000),
        ("k", 1_000),
    ];

    for (suffix, multiplier) in suffixes {
        if let Some(num_str) = quantity.strip_suffix(suffix) {
            let num: u64 = num_str.parse().ok()?;
            return num.checked_mul(*multiplier);
        }
    }

    // Plain bytes.
    quantity.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_millicore() {
        assert_eq!(parse_cpu_to_microseconds("500m"), Some(50_000));
        assert_eq!(parse_cpu_to_microseconds("1000m"), Some(100_000));
        assert_eq!(parse_cpu_to_microseconds("250m"), Some(25_000));
    }

    #[test]
    fn parse_cpu_whole_cores() {
        assert_eq!(parse_cpu_to_microseconds("1"), Some(100_000));
        assert_eq!(parse_cpu_to_microseconds("2"), Some(200_000));
        assert_eq!(parse_cpu_to_microseconds("0.5"), Some(50_000));
    }

    #[test]
    fn parse_memory_binary_suffixes() {
        assert_eq!(parse_memory_to_bytes("256Mi"), Some(256 * 1024 * 1024));
        assert_eq!(parse_memory_to_bytes("4Gi"), Some(4 * 1024 * 1024 * 1024));
        assert_eq!(parse_memory_to_bytes("1Ki"), Some(1024));
    }

    #[test]
    fn parse_memory_decimal_suffixes() {
        assert_eq!(parse_memory_to_bytes("1G"), Some(1_000_000_000));
        assert_eq!(parse_memory_to_bytes("500M"), Some(500_000_000));
    }

    #[test]
    fn parse_memory_plain_bytes() {
        assert_eq!(parse_memory_to_bytes("1048576"), Some(1_048_576));
    }

    #[test]
    fn container_name_is_prefixed() {
        assert_eq!(container_name("my-sandbox"), "openshell-sandbox-my-sandbox");
    }

    #[test]
    fn proxy_container_name_is_prefixed() {
        assert_eq!(
            proxy_container_name("my-sandbox"),
            "openshell-proxy-my-sandbox"
        );
    }

    #[test]
    fn volume_name_uses_id() {
        assert_eq!(
            volume_name("abc-123"),
            "openshell-sandbox-abc-123-workspace"
        );
    }

    #[test]
    fn tls_volume_name_uses_id() {
        assert_eq!(tls_volume_name("abc-123"), "openshell-tls-abc-123");
    }

    #[test]
    fn internal_network_name_uses_id() {
        assert_eq!(
            internal_network_name("abc-123"),
            "openshell-sbx-abc-123"
        );
    }

    #[test]
    fn secret_name_uses_id() {
        assert_eq!(secret_name("abc-123"), "openshell-handshake-abc-123");
    }

    #[test]
    fn short_id_truncates() {
        assert_eq!(short_id("abc123def456789"), "abc123def456");
        assert_eq!(short_id("short"), "short");
    }

    // --- Proxy sidecar spec tests ---

    #[test]
    fn proxy_spec_has_minimal_capabilities() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        let added: Vec<&str> = spec["cap_add"]
            .as_array()
            .expect("cap_add should be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(added.contains(&"SYS_ADMIN"), "proxy missing SYS_ADMIN");
        assert!(added.contains(&"NET_ADMIN"), "proxy missing NET_ADMIN");
        // Proxy must NOT have SYS_PTRACE or DAC_READ_SEARCH.
        assert!(
            !added.contains(&"SYS_PTRACE"),
            "proxy must not have SYS_PTRACE"
        );
        assert!(
            !added.contains(&"DAC_READ_SEARCH"),
            "proxy must not have DAC_READ_SEARCH"
        );
    }

    #[test]
    fn proxy_spec_uses_secret_env() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        // Secret must NOT appear in plaintext env.
        let env_map = spec["env"].as_object().expect("env should be an object");
        assert!(
            !env_map.contains_key("OPENSHELL_SSH_HANDSHAKE_SECRET"),
            "handshake secret should not be in plaintext env"
        );

        let secret_env = spec["secret_env"]
            .as_object()
            .expect("secret_env should be an object");
        assert!(
            secret_env.contains_key("OPENSHELL_SSH_HANDSHAKE_SECRET"),
            "proxy secret_env should map OPENSHELL_SSH_HANDSHAKE_SECRET"
        );
        assert_eq!(
            secret_env["OPENSHELL_SSH_HANDSHAKE_SECRET"].as_str(),
            Some("openshell-handshake-test-id"),
        );
    }

    #[test]
    fn proxy_spec_has_role_label() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        let labels = spec["labels"]
            .as_object()
            .expect("labels should be an object");
        assert_eq!(
            labels.get(LABEL_ROLE).and_then(|v| v.as_str()),
            Some(ROLE_PROXY),
            "proxy should have role=proxy label"
        );
    }

    #[test]
    fn proxy_spec_has_tls_volume_rw() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        let volumes = spec["volumes"]
            .as_array()
            .expect("volumes should be an array");
        let tls_vol = volumes
            .iter()
            .find(|v| v["dest"].as_str() == Some("/openshell-tls"))
            .expect("proxy should mount TLS volume");
        let options: Vec<&str> = tls_vol["options"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(options.contains(&"rw"), "proxy TLS volume should be rw");
    }

    #[test]
    fn proxy_spec_sets_dns_servers() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let dns = vec!["8.8.8.8".to_string(), "1.1.1.1".to_string()];
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &dns);

        let dns_server: Vec<&str> = spec["dns_server"]
            .as_array()
            .expect("dns_server should be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(dns_server, vec!["8.8.8.8", "1.1.1.1"]);
    }

    #[test]
    fn proxy_spec_has_no_port_mappings() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        // portmappings should be empty (skip_serializing_if = "Vec::is_empty").
        assert!(
            spec.get("portmappings").is_none()
                || spec["portmappings"]
                    .as_array()
                    .is_none_or(|a| a.is_empty()),
            "proxy should not expose any ports"
        );
    }

    #[test]
    fn proxy_spec_healthcheck_checks_tls_ready() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_proxy_sidecar_spec(&sandbox, &config, "openshell-sbx-test-id", &[]);

        let test_cmd = spec["healthconfig"]["test"]
            .as_array()
            .expect("healthcheck test should be an array");
        let command = test_cmd
            .get(1)
            .and_then(|v| v.as_str())
            .expect("healthcheck should include shell command");
        assert!(
            command.contains("/openshell-tls/.ready"),
            "proxy healthcheck should check for TLS ready sentinel"
        );
    }

    // --- Agent container spec tests ---

    #[test]
    fn agent_spec_has_reduced_capabilities() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let added: Vec<&str> = spec["cap_add"]
            .as_array()
            .expect("cap_add should be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(added.contains(&"SYS_ADMIN"), "agent missing SYS_ADMIN");
        assert!(added.contains(&"NET_ADMIN"), "agent missing NET_ADMIN");
        assert!(added.contains(&"SYSLOG"), "agent missing SYSLOG");
        // Agent must NOT have SYS_PTRACE or DAC_READ_SEARCH in sidecar mode.
        assert!(
            !added.contains(&"SYS_PTRACE"),
            "agent must not have SYS_PTRACE in sidecar mode"
        );
        assert!(
            !added.contains(&"DAC_READ_SEARCH"),
            "agent must not have DAC_READ_SEARCH in sidecar mode"
        );

        // Verify essential caps are not dropped.
        let dropped: Vec<&str> = spec["cap_drop"]
            .as_array()
            .expect("cap_drop should be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(!dropped.contains(&"SETUID"), "SETUID must not be dropped");
        assert!(!dropped.contains(&"SETGID"), "SETGID must not be dropped");
        assert!(!dropped.contains(&"CHOWN"), "CHOWN must not be dropped");
        assert!(!dropped.contains(&"FOWNER"), "FOWNER must not be dropped");
        assert!(
            !dropped.contains(&"ALL"),
            "must not use cap_drop:ALL in rootless Podman"
        );
    }

    #[test]
    fn agent_spec_uses_secret_env_not_plaintext() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");
        assert!(
            !env_map.contains_key("OPENSHELL_SSH_HANDSHAKE_SECRET"),
            "handshake secret should not be in plaintext env"
        );

        let secret_env = spec["secret_env"]
            .as_object()
            .expect("secret_env should be an object");
        assert!(
            secret_env.contains_key("OPENSHELL_SSH_HANDSHAKE_SECRET"),
            "agent secret_env should map OPENSHELL_SSH_HANDSHAKE_SECRET"
        );
        assert_eq!(
            secret_env["OPENSHELL_SSH_HANDSHAKE_SECRET"].as_str(),
            Some("openshell-handshake-test-id"),
        );
    }

    #[test]
    fn agent_spec_sets_sandbox_name_in_env() {
        let sandbox = test_sandbox("test-id", "my-sandbox");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");
        assert_eq!(
            env_map.get("OPENSHELL_SANDBOX").and_then(|v| v.as_str()),
            Some("my-sandbox"),
        );
    }

    #[test]
    fn agent_spec_sets_ssh_socket_path_in_env() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");
        assert_eq!(
            env_map
                .get("OPENSHELL_SSH_SOCKET_PATH")
                .and_then(|v| v.as_str()),
            Some("/run/openshell/test-ssh.sock"),
        );
    }

    #[test]
    fn agent_spec_healthcheck_accepts_supervisor_socket() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let healthcheck = spec["healthconfig"]["test"]
            .as_array()
            .expect("healthcheck test should be an array");
        let command = healthcheck
            .get(1)
            .and_then(|v| v.as_str())
            .expect("healthcheck should include shell command");
        assert!(
            command.contains("test -S /run/openshell/test-ssh.sock"),
            "healthcheck should consider the supervisor Unix socket ready"
        );
    }

    #[test]
    fn agent_spec_required_vars_cannot_be_overridden() {
        use openshell_core::proto::compute::v1::{DriverSandboxSpec, DriverSandboxTemplate};

        let mut sandbox = test_sandbox("test-id", "legit-name");
        let mut env_overrides = std::collections::HashMap::new();
        env_overrides.insert(
            "OPENSHELL_ENDPOINT".to_string(),
            "http://evil.example.com".to_string(),
        );
        env_overrides.insert("OPENSHELL_SANDBOX_ID".to_string(), "spoofed-id".to_string());
        env_overrides.insert(
            "OPENSHELL_SSH_SOCKET_PATH".to_string(),
            "/tmp/evil.sock".to_string(),
        );
        sandbox.spec = Some(DriverSandboxSpec {
            environment: env_overrides,
            template: Some(DriverSandboxTemplate::default()),
            ..Default::default()
        });

        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");

        // OPENSHELL_ENDPOINT is overwritten by the sidecar proxy address.
        assert_eq!(
            env_map.get("OPENSHELL_ENDPOINT").and_then(|v| v.as_str()),
            Some("http://10.89.0.2:8081"),
            "OPENSHELL_ENDPOINT must point to proxy, not be overridden by user env"
        );
        assert_eq!(
            env_map.get("OPENSHELL_SANDBOX_ID").and_then(|v| v.as_str()),
            Some("test-id"),
            "OPENSHELL_SANDBOX_ID must not be overridden by user env"
        );
        assert_eq!(
            env_map
                .get("OPENSHELL_SSH_SOCKET_PATH")
                .and_then(|v| v.as_str()),
            Some("/run/openshell/test-ssh.sock"),
            "OPENSHELL_SSH_SOCKET_PATH must not be overridden by user env"
        );
    }

    #[test]
    fn agent_spec_required_labels_cannot_be_overridden() {
        use openshell_core::proto::compute::v1::{DriverSandboxSpec, DriverSandboxTemplate};

        let mut sandbox = test_sandbox("real-id", "real-name");
        sandbox.namespace = "real-namespace".to_string();
        let mut label_overrides = std::collections::HashMap::new();
        label_overrides.insert("openshell.sandbox-id".to_string(), "spoofed-id".to_string());
        label_overrides.insert(
            "openshell.sandbox-name".to_string(),
            "spoofed-name".to_string(),
        );
        label_overrides.insert(
            "openshell.sandbox-namespace".to_string(),
            "spoofed-namespace".to_string(),
        );
        sandbox.spec = Some(DriverSandboxSpec {
            template: Some(DriverSandboxTemplate {
                labels: label_overrides,
                ..Default::default()
            }),
            ..Default::default()
        });

        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-real-id",
            "10.89.0.2",
            "openshell-tls-real-id",
        );

        let labels = spec["labels"]
            .as_object()
            .expect("labels should be an object");
        assert_eq!(
            labels.get("openshell.sandbox-id").and_then(|v| v.as_str()),
            Some("real-id"),
            "openshell.sandbox-id must not be overridden by template labels"
        );
        assert_eq!(
            labels
                .get("openshell.sandbox-name")
                .and_then(|v| v.as_str()),
            Some("real-name"),
            "openshell.sandbox-name must not be overridden by template labels"
        );
        assert_eq!(
            labels
                .get("openshell.sandbox-namespace")
                .and_then(|v| v.as_str()),
            Some("real-namespace"),
            "openshell.sandbox-namespace must not be overridden by template labels"
        );
    }

    #[test]
    fn container_spec_injects_host_aliases() {
        let sandbox = test_nested_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_container_spec(&sandbox, &config);

        let hostadd: Vec<&str> = spec["hostadd"]
            .as_array()
            .expect("hostadd should be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();

        assert!(
            hostadd.contains(&"host.containers.internal:host-gateway"),
            "missing Podman host alias"
        );
        assert!(
            hostadd.contains(&"host.openshell.internal:host-gateway"),
            "missing OpenShell stable host alias"
        );
        assert!(
            !hostadd.contains(&"host.docker.internal:host-gateway"),
            "Podman should not inject Docker's host alias"
        );
    }

    #[test]
    fn agent_spec_has_role_label() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let labels = spec["labels"]
            .as_object()
            .expect("labels should be an object");
        assert_eq!(
            labels.get(LABEL_ROLE).and_then(|v| v.as_str()),
            Some(ROLE_AGENT),
            "agent should have role=agent label"
        );
    }

    #[test]
    fn agent_spec_has_proxy_env_vars() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");

        assert_eq!(
            env_map.get("HTTP_PROXY").and_then(|v| v.as_str()),
            Some("http://10.89.0.2:3128"),
            "HTTP_PROXY should point to proxy sidecar"
        );
        assert_eq!(
            env_map.get("HTTPS_PROXY").and_then(|v| v.as_str()),
            Some("http://10.89.0.2:3128"),
            "HTTPS_PROXY should point to proxy sidecar"
        );
        assert_eq!(
            env_map.get("NO_PROXY").and_then(|v| v.as_str()),
            Some("127.0.0.1,localhost,::1,10.89.0.2"),
            "NO_PROXY must include proxy IP to avoid double-proxying gRPC traffic"
        );
        assert_eq!(
            env_map.get("no_proxy").and_then(|v| v.as_str()),
            Some("127.0.0.1,localhost,::1,10.89.0.2"),
            "no_proxy must include proxy IP to avoid double-proxying gRPC traffic"
        );
        assert_eq!(
            env_map.get("OPENSHELL_PROXY_MODE").and_then(|v| v.as_str()),
            Some("sidecar"),
            "OPENSHELL_PROXY_MODE should be 'sidecar'"
        );
        assert_eq!(
            env_map.get("OPENSHELL_ENDPOINT").and_then(|v| v.as_str()),
            Some("http://10.89.0.2:8081"),
            "OPENSHELL_ENDPOINT should route through proxy"
        );
    }

    #[test]
    fn agent_spec_has_ca_trust_env_vars() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");
        let ca_bundle = "/openshell-tls/ca-bundle.pem";
        let ca_cert = "/openshell-tls/openshell-ca.pem";
        // Bundle-based env vars should point to the combined CA bundle.
        for var in &["SSL_CERT_FILE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "REQUESTS_CA_BUNDLE"] {
            assert_eq!(
                env_map.get(*var).and_then(|v| v.as_str()),
                Some(ca_bundle),
                "{var} should point to {ca_bundle}"
            );
        }
        // Node.js needs just the proxy CA cert (not the full bundle).
        assert_eq!(
            env_map.get("NODE_EXTRA_CA_CERTS").and_then(|v| v.as_str()),
            Some(ca_cert),
            "NODE_EXTRA_CA_CERTS should point to {ca_cert}"
        );
    }

    #[test]
    fn agent_spec_blocks_git_ssh() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let env_map = spec["env"].as_object().expect("env should be an object");
        let git_ssh = env_map
            .get("GIT_SSH_COMMAND")
            .and_then(|v| v.as_str())
            .expect("GIT_SSH_COMMAND should be set");
        assert!(
            git_ssh.contains("exit 1"),
            "GIT_SSH_COMMAND should block SSH"
        );
    }

    #[test]
    fn agent_spec_internal_network_only() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let internal_net = "openshell-sbx-test-id";
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            internal_net,
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let networks = spec["networks"]
            .as_object()
            .expect("networks should be an object");
        assert!(
            networks.contains_key(internal_net),
            "agent should be on internal network"
        );
        assert_eq!(
            networks.len(),
            1,
            "agent should ONLY be on internal network (no bridge)"
        );
    }

    #[test]
    fn agent_spec_has_tls_volume_ro() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let tls_vol = "openshell-tls-test-id";
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            tls_vol,
        );

        let volumes = spec["volumes"]
            .as_array()
            .expect("volumes should be an array");
        let tls = volumes
            .iter()
            .find(|v| v["dest"].as_str() == Some("/openshell-tls"))
            .expect("agent should mount TLS volume");
        let options: Vec<&str> = tls["options"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(options.contains(&"ro"), "agent TLS volume should be ro");
        assert_eq!(
            tls["name"].as_str(),
            Some(tls_vol),
            "TLS volume name should match"
        );
    }

    #[test]
    fn agent_spec_includes_supervisor_image_volume() {
        let sandbox = test_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_agent_container_spec(
            &sandbox,
            &config,
            "openshell-sbx-test-id",
            "10.89.0.2",
            "openshell-tls-test-id",
        );

        let image_volumes = spec["image_volumes"]
            .as_array()
            .expect("image_volumes should be an array");
        assert_eq!(
            image_volumes.len(),
            1,
            "should have exactly one image volume"
        );
        assert_eq!(
            image_volumes[0]["source"].as_str(),
            Some("openshell/supervisor:latest"),
        );
        assert_eq!(
            image_volumes[0]["destination"].as_str(),
            Some("/opt/openshell/bin"),
        );
        assert_eq!(image_volumes[0]["rw"].as_bool(), Some(false));
    }

    #[test]
    fn parse_cpu_negative_returns_none() {
        assert_eq!(parse_cpu_to_microseconds("-1"), None);
        assert_eq!(parse_cpu_to_microseconds("-500m"), None);
    }

    #[test]
    fn parse_cpu_zero_returns_none() {
        assert_eq!(parse_cpu_to_microseconds("0m"), None);
        assert_eq!(parse_cpu_to_microseconds("0"), None);
    }

    fn test_sandbox(id: &str, name: &str) -> DriverSandbox {
        DriverSandbox {
            id: id.to_string(),
            name: name.to_string(),
            namespace: String::new(),
            spec: None,
            status: None,
        }
    }

    fn test_config() -> PodmanComputeConfig {
        PodmanComputeConfig {
            socket_path: std::path::PathBuf::from("/tmp/test.sock"),
            default_image: "test-image:latest".to_string(),
            grpc_endpoint: "http://localhost:50051".to_string(),
            sandbox_ssh_socket_path: "/run/openshell/test-ssh.sock".to_string(),
            ssh_handshake_secret: "test-secret-value".to_string(),
            ..PodmanComputeConfig::default()
        }
    }

    /// Helper to create a nested sandbox for test purposes.
    fn test_nested_sandbox(id: &str, name: &str) -> DriverSandbox {
        use openshell_core::proto::compute::v1::{DriverSandboxSpec, DriverSandboxTemplate};
        let mut sandbox = test_sandbox(id, name);
        sandbox.spec = Some(DriverSandboxSpec {
            template: Some(DriverSandboxTemplate {
                mode: 2, // SANDBOX_MODE_NESTED
                ..Default::default()
            }),
            ..Default::default()
        });
        sandbox
    }

    #[test]
    fn container_spec_includes_supervisor_image_volume() {
        let sandbox = test_nested_sandbox("test-id", "test-name");
        let config = test_config();
        let spec = build_container_spec(&sandbox, &config);

        let image_volumes = spec["image_volumes"]
            .as_array()
            .expect("image_volumes should be an array");
        assert_eq!(
            image_volumes.len(),
            1,
            "should have exactly one image volume"
        );

        let vol = &image_volumes[0];
        assert_eq!(
            vol["source"].as_str(),
            Some("openshell/supervisor:latest"),
            "image volume source should be the supervisor image"
        );
        assert_eq!(
            vol["destination"].as_str(),
            Some("/opt/openshell/bin"),
            "image volume destination should be /opt/openshell/bin"
        );
        assert_eq!(
            vol["rw"].as_bool(),
            Some(false),
            "image volume should be read-only"
        );
    }

    #[test]
    fn container_spec_includes_tls_mounts_when_configured() {
        let sandbox = test_nested_sandbox("tls-id", "tls-name");
        let mut config = test_config();
        config.guest_tls_ca = Some(std::path::PathBuf::from("/host/ca.crt"));
        config.guest_tls_cert = Some(std::path::PathBuf::from("/host/tls.crt"));
        config.guest_tls_key = Some(std::path::PathBuf::from("/host/tls.key"));

        let spec = build_container_spec(&sandbox, &config);

        // Verify TLS env vars are set.
        let env_map = spec["env"].as_object().expect("env should be an object");
        assert_eq!(
            env_map.get("OPENSHELL_TLS_CA").and_then(|v| v.as_str()),
            Some("/etc/openshell/tls/client/ca.crt"),
        );
        assert_eq!(
            env_map.get("OPENSHELL_TLS_CERT").and_then(|v| v.as_str()),
            Some("/etc/openshell/tls/client/tls.crt"),
        );
        assert_eq!(
            env_map.get("OPENSHELL_TLS_KEY").and_then(|v| v.as_str()),
            Some("/etc/openshell/tls/client/tls.key"),
        );

        // Verify bind mounts exist for all three cert files.
        let mounts = spec["mounts"]
            .as_array()
            .expect("mounts should be an array");
        let bind_dests: Vec<&str> = mounts
            .iter()
            .filter(|m| m["type"].as_str() == Some("bind"))
            .filter_map(|m| m["destination"].as_str())
            .collect();
        assert!(
            bind_dests.contains(&"/etc/openshell/tls/client/ca.crt"),
            "should bind-mount CA cert"
        );
        assert!(
            bind_dests.contains(&"/etc/openshell/tls/client/tls.crt"),
            "should bind-mount client cert"
        );
        assert!(
            bind_dests.contains(&"/etc/openshell/tls/client/tls.key"),
            "should bind-mount client key"
        );

        // Verify SELinux relabel option is present iff SELinux is enabled.
        let tls_binds: Vec<&Value> = mounts
            .iter()
            .filter(|m| m["type"].as_str() == Some("bind"))
            .collect();
        let has_z = tls_binds.iter().all(|m| {
            m["options"]
                .as_array()
                .is_some_and(|opts| opts.iter().any(|o| o.as_str() == Some("z")))
        });
        assert_eq!(
            has_z,
            is_selinux_enabled(),
            "TLS bind mounts should include 'z' option iff SELinux is enabled"
        );
    }

    #[test]
    fn container_spec_omits_tls_without_config() {
        let sandbox = test_nested_sandbox("notls-id", "notls-name");
        let config = test_config();

        let spec = build_container_spec(&sandbox, &config);

        let env_map = spec["env"].as_object().expect("env should be an object");
        assert!(
            env_map.get("OPENSHELL_TLS_CA").is_none(),
            "TLS env vars should not be set without TLS config"
        );

        let mounts = spec["mounts"]
            .as_array()
            .expect("mounts should be an array");
        let bind_count = mounts
            .iter()
            .filter(|m| m["type"].as_str() == Some("bind"))
            .count();
        assert_eq!(bind_count, 0, "no bind mounts without TLS config");
    }

    #[test]
    fn nested_spec_includes_adc_bind_mount_when_configured() {
        let sandbox = test_nested_sandbox("adc-test-id", "adc-test");
        let config = PodmanComputeConfig {
            adc_host_path: Some(std::path::PathBuf::from("/host/adc.json")),
            ..test_config()
        };
        let spec = build_container_spec(&sandbox, &config);

        let mounts = spec["mounts"]
            .as_array()
            .expect("mounts should be an array");
        let adc_bind = mounts
            .iter()
            .find(|m| m["destination"].as_str() == Some("/run/gcloud/adc.json"));
        assert!(adc_bind.is_some(), "ADC bind mount should be present");

        let env_map = spec["env"].as_object().expect("env should be an object");
        assert_eq!(
            env_map
                .get("GOOGLE_APPLICATION_CREDENTIALS")
                .and_then(|v| v.as_str()),
            Some("/run/gcloud/adc.json"),
            "GOOGLE_APPLICATION_CREDENTIALS should be set for ADC"
        );
    }
}

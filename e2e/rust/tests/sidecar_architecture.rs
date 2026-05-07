// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "e2e")]

//! End-to-end tests for the Podman sidecar proxy architecture.
//!
//! Each sandbox is expected to consist of three Podman resources:
//!
//! 1. **Internal network** (`openshell-sbx-{id}`) — a `--internal` bridge with
//!    no default gateway, isolating the agent container from the internet.
//! 2. **Proxy sidecar container** (`openshell-proxy-{name}`) — attached to both
//!    the internal network and the default bridge, providing L7 proxy and TCP
//!    port forwarding as the sole egress path.
//! 3. **Agent container** (`openshell-sandbox-{name}`) — attached only to the
//!    internal network, with no direct internet route.
//!
//! These tests define the expected behavior of this architecture. They are
//! written before the backend implementation and are **expected to fail** until
//! the sidecar architecture is in place.

use std::process::Stdio;

use openshell_e2e::harness::binary::openshell_cmd;
use openshell_e2e::harness::output::strip_ansi;
use openshell_e2e::harness::sandbox::SandboxGuard;

/// Run a podman command and return its combined stdout+stderr.
async fn podman_cmd(args: &[&str]) -> String {
    let output = tokio::process::Command::new("podman")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("podman command should execute");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{stdout}{stderr}")
}

/// Helper to list sandbox names via the CLI.
async fn sandbox_list_names() -> Vec<String> {
    let mut cmd = openshell_cmd();
    cmd.args(["sandbox", "list", "--names"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await.expect("spawn openshell sandbox list");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = strip_ansi(&format!("{stdout}{stderr}"));

    combined
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Verify that sandbox creation produces all three sidecar resources
/// (internal network, proxy container, agent container) and that cleanup
/// removes all of them.
#[tokio::test]
async fn sidecar_creates_three_resources() {
    // Create a sandbox that runs a trivial command.
    let mut sb = SandboxGuard::create(&["--", "echo", "sidecar-ok"])
        .await
        .expect("sandbox create should succeed");

    let name = sb.name.clone();

    // 1. An internal network named "openshell-sbx-*" should exist.
    let networks = podman_cmd(&[
        "network",
        "ls",
        "--format",
        "{{.Name}}",
        "--filter",
        &format!("name=openshell-sbx-"),
    ])
    .await;
    assert!(
        !networks.trim().is_empty(),
        "expected an internal network matching 'openshell-sbx-*' after sandbox create, got:\n{networks}",
    );

    // 2. A proxy sidecar container named "openshell-proxy-{name}" should exist.
    let proxy_filter = format!("name=openshell-proxy-{name}");
    let proxy_containers = podman_cmd(&[
        "ps",
        "-a",
        "--format",
        "{{.Names}}",
        "--filter",
        &proxy_filter,
    ])
    .await;
    assert!(
        proxy_containers.contains(&format!("openshell-proxy-{name}")),
        "expected proxy container 'openshell-proxy-{name}' to exist, got:\n{proxy_containers}",
    );

    // 3. An agent container named "openshell-sandbox-{name}" should exist.
    let agent_filter = format!("name=openshell-sandbox-{name}");
    let agent_containers = podman_cmd(&[
        "ps",
        "-a",
        "--format",
        "{{.Names}}",
        "--filter",
        &agent_filter,
    ])
    .await;
    assert!(
        agent_containers.contains(&format!("openshell-sandbox-{name}")),
        "expected agent container 'openshell-sandbox-{name}' to exist, got:\n{agent_containers}",
    );

    // 4. Cleanup and verify all resources are gone.
    sb.cleanup().await;

    let networks_after = podman_cmd(&[
        "network",
        "ls",
        "--format",
        "{{.Name}}",
        "--filter",
        &format!("name=openshell-sbx-"),
    ])
    .await;
    // The specific network for this sandbox should be gone. There may be
    // networks from other concurrent tests, so check that none contain the
    // sandbox name.
    assert!(
        !networks_after.contains(&name),
        "internal network for sandbox '{name}' should be removed after cleanup, got:\n{networks_after}",
    );

    let proxy_after = podman_cmd(&[
        "ps",
        "-a",
        "--format",
        "{{.Names}}",
        "--filter",
        &proxy_filter,
    ])
    .await;
    assert!(
        !proxy_after.contains(&format!("openshell-proxy-{name}")),
        "proxy container should be removed after cleanup, got:\n{proxy_after}",
    );

    let agent_after = podman_cmd(&[
        "ps",
        "-a",
        "--format",
        "{{.Names}}",
        "--filter",
        &agent_filter,
    ])
    .await;
    assert!(
        !agent_after.contains(&format!("openshell-sandbox-{name}")),
        "agent container should be removed after cleanup, got:\n{agent_after}",
    );
}

/// Verify that the agent container cannot reach the internet without the proxy.
///
/// The agent runs on an `--internal` Podman network with no default gateway.
/// When we unset proxy env vars and attempt a direct connection, the request
/// must fail — either at the network level (no route) or because the proxy
/// denies the policy-unconfigured destination.
#[tokio::test]
async fn sidecar_proxy_bypass_fails() {
    let sb = SandboxGuard::create(&[
        "--",
        "bash",
        "-c",
        concat!(
            "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; ",
            "curl --noproxy '*' --connect-timeout 5 https://httpbin.org/get 2>&1; ",
            "echo EXIT_CODE=$?",
        ),
    ])
    .await
    .expect("sandbox create should succeed");

    let output = &sb.create_output;

    // The request must fail — exit code should be non-zero.
    assert!(
        output.contains("EXIT_CODE=") && !output.contains("EXIT_CODE=0"),
        "curl without proxy should fail with non-zero exit code, got:\n{output}",
    );

    // The failure should be network-level: unreachable, no route, DNS failure,
    // or connection timeout — proving the internal network blocks direct egress.
    let blocked = output.contains("Network unreachable")
        || output.contains("Network is unreachable")
        || output.contains("No route to host")
        || output.contains("Could not resolve host")
        || output.contains("Connection timed out")
        || output.contains("Couldn't connect to server")
        || output.contains("Failed to connect");

    assert!(
        blocked,
        "expected network-level failure (unreachable/no route/DNS/timeout), got:\n{output}",
    );
}

/// Verify that git-over-SSH fails inside the sandbox.
///
/// The agent container is on an `--internal` network with no route to external
/// SSH servers. Even without `GIT_SSH_COMMAND` blocking, git-over-SSH should
/// fail because the network topology prevents it.
#[tokio::test]
async fn sidecar_git_ssh_blocked() {
    // Try a git-over-SSH operation — it should fail, either because
    // GIT_SSH_COMMAND blocks it or because the network doesn't route.
    let sb = SandboxGuard::create(&[
        "--",
        "bash",
        "-c",
        "git ls-remote git@github.com:NVIDIA/OpenShell.git HEAD 2>&1; echo EXIT_CODE=$?",
    ])
    .await
    .expect("sandbox create should succeed");

    let output = &sb.create_output;

    // git-over-SSH must fail with a non-zero exit code.
    assert!(
        output.contains("EXIT_CODE=") && !output.contains("EXIT_CODE=0"),
        "git-over-SSH should fail, got:\n{output}",
    );
}

/// Verify that `openshell sandbox list` shows the sandbox exactly once —
/// the proxy sidecar container must NOT appear as a separate sandbox entry.
#[tokio::test]
async fn sidecar_sandbox_list_no_duplicates() {
    let mut sb = SandboxGuard::create(&["--", "echo", "list-test"])
        .await
        .expect("sandbox create should succeed");

    let name = sb.name.clone();

    let names = sandbox_list_names().await;

    // Count how many times the sandbox name appears in the list.
    let count = names.iter().filter(|n| *n == &name).count();
    assert_eq!(
        count, 1,
        "sandbox '{name}' should appear exactly once in list output, \
         but appeared {count} times. Full list: {names:?}",
    );

    // Also verify the proxy container name does NOT appear in the list.
    let proxy_name = format!("openshell-proxy-{name}");
    assert!(
        !names.contains(&proxy_name),
        "proxy container '{proxy_name}' should NOT appear in sandbox list. \
         Full list: {names:?}",
    );

    sb.cleanup().await;
}

/// Verify that deleting a sandbox leaves no orphaned Podman resources:
/// no networks, no containers (proxy or agent), and no TLS volumes
/// matching the sandbox name.
#[tokio::test]
async fn sidecar_cleanup_no_orphans() {
    // Create and immediately delete a sandbox.
    let mut sb = SandboxGuard::create(&["--", "echo", "orphan-test"])
        .await
        .expect("sandbox create should succeed");

    let name = sb.name.clone();
    sb.cleanup().await;

    // 1. No networks matching the sandbox name.
    let networks = podman_cmd(&[
        "network",
        "ls",
        "--format",
        "{{.Name}}",
    ])
    .await;
    let orphan_networks: Vec<&str> = networks
        .lines()
        .filter(|line| line.contains(&name))
        .collect();
    assert!(
        orphan_networks.is_empty(),
        "expected no orphaned networks for sandbox '{name}', found: {orphan_networks:?}",
    );

    // 2. No containers matching proxy or agent patterns.
    let containers = podman_cmd(&[
        "ps",
        "-a",
        "--format",
        "{{.Names}}",
    ])
    .await;

    let proxy_name = format!("openshell-proxy-{name}");
    let agent_name = format!("openshell-sandbox-{name}");

    assert!(
        !containers.contains(&proxy_name),
        "expected no orphaned proxy container '{proxy_name}', \
         but it still exists after cleanup",
    );
    assert!(
        !containers.contains(&agent_name),
        "expected no orphaned agent container '{agent_name}', \
         but it still exists after cleanup",
    );

    // 3. No volumes matching a TLS volume pattern for this sandbox.
    let volumes = podman_cmd(&["volume", "ls", "--format", "{{.Name}}"]).await;
    let orphan_volumes: Vec<&str> = volumes
        .lines()
        .filter(|line| line.contains(&name))
        .collect();
    assert!(
        orphan_volumes.is_empty(),
        "expected no orphaned volumes for sandbox '{name}', found: {orphan_volumes:?}",
    );
}

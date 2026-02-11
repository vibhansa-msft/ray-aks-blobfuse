import concurrent.futures
from functools import partial
import importlib.resources
import os
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import cast, ClassVar, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import click

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.workload.workload_sdk import WorkloadSDK
from anyscale.client.openapi_client.models import (
    CreateExperimentalWorkspace,
    ExperimentalWorkspace,
)
from anyscale.client.openapi_client.models.resource_tag_resource_type import (
    ResourceTagResourceType,
)
from anyscale.client.openapi_client.models.session_state import SessionState
from anyscale.utils.runtime_env import parse_requirements_file
from anyscale.workspace.models import (
    UpdateWorkspaceConfig,
    Workspace,
    WorkspaceConfig,
    WorkspaceSortField,
    WorkspaceSortOrder,
    WorkspaceState,
)


SSH_TEMPLATE = """
Host Head-Node
  HostName {head_node_ip}
  User ubuntu
  IdentityFile {key_path}
  StrictHostKeyChecking false
  UserKnownHostsFile /dev/null
  IdentitiesOnly yes
  LogLevel ERROR

Host {name}
  HostName 0.0.0.0
  ProxyJump Head-Node
  Port 5020
  User ray
  IdentityFile {key_path}
  StrictHostKeyChecking false
  UserKnownHostsFile /dev/null
  IdentitiesOnly yes
  LogLevel ERROR
"""

# TODO(bryce): These options are already stated in the SSH_TEMPLATE, but we
# still pass them in the ssh command. It should be safe to remove them.
ANYSCALE_WORKSPACES_SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "IdentitiesOnly=yes",
]

# HTTPS SSH constants for rsync
HTTPS_PORT = "443"
WSS_PATH = "/sshws"
PREFERRED_AUTH_METHOD = "PreferredAuthentications=publickey"


class PrivateWorkspaceSDK(WorkloadSDK):
    _POLLING_INTERVAL_SECONDS = 10.0
    _WAIT_TIMEOUT_SECONDS = 1800.0

    _BACKEND_SESSION_STATE_TO_WORKSPACE_STATE: ClassVar[
        Dict[SessionState, WorkspaceState]
    ] = {
        SessionState.STOPPED: WorkspaceState.TERMINATED,
        SessionState.TERMINATED: WorkspaceState.TERMINATED,
        SessionState.STARTINGUP: WorkspaceState.STARTING,
        SessionState.STARTUPERRORED: WorkspaceState.ERRORED,
        SessionState.RUNNING: WorkspaceState.RUNNING,
        SessionState.UPDATING: WorkspaceState.UPDATING,
        SessionState.UPDATINGERRORED: WorkspaceState.ERRORED,
        SessionState.STOPPING: WorkspaceState.TERMINATING,
        SessionState.TERMINATING: WorkspaceState.TERMINATING,
        SessionState.AWAITINGSTARTUP: WorkspaceState.STARTING,
        SessionState.AWAITINGFILEMOUNTS: WorkspaceState.STARTING,
        SessionState.TERMINATINGERRORED: WorkspaceState.ERRORED,
        SessionState.STOPPINGERRORED: WorkspaceState.ERRORED,
    }

    def _resolve_to_workspace_model(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> ExperimentalWorkspace:
        if name is None and id is None:
            raise ValueError("One of 'name' or 'id' must be provided.")

        if name is not None and id is not None:
            raise ValueError("Only one of 'name' or 'id' can be provided.")

        if id is not None and (cloud is not None or project is not None):
            raise ValueError("'cloud' and 'project' should only be used with 'name'.")

        model: Optional[ExperimentalWorkspace] = self.client.get_workspace(
            name=name, id=id, cloud=cloud, project=project
        )
        if model is None:
            if name is not None:
                raise ValueError(f"Workspace with name '{name}' was not found.")
            else:
                raise ValueError(f"Workspace with ID '{id}' was not found.")

        return model

    def create(self, config: WorkspaceConfig) -> str:
        if not config or config.name is None:
            raise ValueError("Workspace name must be configured")

        name = config.name

        compute_config_id, cloud_id = self.resolve_compute_config_and_cloud_id(
            compute_config=config.compute_config, cloud=config.cloud,  # type: ignore
        )

        project_id = self.client.get_project_id(
            parent_cloud_id=cloud_id, name=config.project
        )

        build_id = None

        if config.containerfile is not None:
            build_id = self._image_sdk.build_image_from_containerfile(
                name=f"image-for-workspace-{name}",
                containerfile=self.get_containerfile_contents(config.containerfile),
                ray_version=config.ray_version,
            )
        elif config.image_uri is not None:
            build_id = self._image_sdk.registery_image(
                image_uri=config.image_uri,
                registry_login_secret=config.registry_login_secret,
                ray_version=config.ray_version,
            )

        dynamic_requirements = None
        if (
            config.requirements
            and self._image_sdk.enable_image_build_for_tracked_requirements
        ):
            requirements = (
                parse_requirements_file(config.requirements)
                if isinstance(config.requirements, str)
                else config.requirements
            )
            if requirements:
                build_id = self._image_sdk.build_image_from_requirements(
                    name=f"image-for-workspace-{name}",
                    base_build_id=self.client.get_default_build_id(),
                    requirements=requirements,
                )
        elif config.requirements:
            dynamic_requirements = (
                parse_requirements_file(config.requirements)
                if isinstance(config.requirements, str)
                else config.requirements
            )

        if build_id is None:
            build_id = self.client.get_default_build_id()

        workspace_id = self.client.create_workspace(
            model=CreateExperimentalWorkspace(
                name=name,
                project_id=project_id,
                compute_config_id=compute_config_id,
                cluster_environment_build_id=build_id,
                idle_timeout_minutes=config.idle_termination_minutes,
                cloud_id=cloud_id,
                tags=config.tags,
                skip_start=True,
            )
        )

        self._logger.info(f"Workspace created successfully id: {workspace_id}")

        if dynamic_requirements:
            self.client.update_workspace_dependencies_offline_only(
                workspace_id=workspace_id, requirements=dynamic_requirements
            )
            self._logger.info(f"Applied dynamic requirements to workspace id: {name}")
        if config.env_vars:
            self.client.update_workspace_env_vars_offline_only(
                workspace_id=workspace_id, env_vars=config.env_vars
            )
            self._logger.info(f"Applied environment variables to workspace id: {name}")

        return workspace_id

    def start(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )
        self.client.start_workspace(workspace_model.id)
        self.logger.info(f"Starting workspace '{workspace_model.name}'")
        return workspace_model.id

    def terminate(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )
        self.client.terminate_workspace(workspace_model.id)
        self.logger.info(f"Terminating workspace '{workspace_model.name}'")
        return workspace_model.id

    def status(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )

        status = self._get_workspace_status(workspace_model.id)
        return status

    def wait(  # noqa: PLR0912
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        timeout_s: float = _WAIT_TIMEOUT_SECONDS,
        state: Union[str, WorkspaceState] = WorkspaceState.RUNNING,
        interval_s: float = _POLLING_INTERVAL_SECONDS,
    ):
        if not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a float")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be >= 0")

        if not isinstance(interval_s, (int, float)):
            raise TypeError("interval_s must be a float")
        if interval_s <= 0:
            raise ValueError("interval_s must be >= 0")

        if isinstance(state, str):
            try:
                state = WorkspaceState.validate(state)
            except KeyError:
                raise ValueError(f"Invalid state: {state}")

        if not isinstance(state, WorkspaceState):
            raise TypeError("'state' must be a WorkspaceState.")

        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )

        workspace_id_or_name = workspace_model.id or workspace_model.name
        curr_state = self._get_workspace_status(workspace_model.id)

        self.logger.info(
            f"Waiting for workspace '{workspace_id_or_name}' to reach target state {state}, currently in state: {curr_state}"
        )
        for _ in self.timer.poll(timeout_s=timeout_s, interval_s=interval_s):
            new_state = self._get_workspace_status(workspace_model.id)

            if new_state != curr_state:
                self.logger.info(
                    f"Workspace '{workspace_id_or_name}' transitioned from {curr_state} to {new_state}"
                )
                curr_state = new_state

            if curr_state == state:
                self.logger.info(
                    f"Workspace '{workspace_id_or_name}' reached target state, exiting"
                )
                break
        else:
            raise TimeoutError(
                f"Workspace '{workspace_id_or_name}' did not reach target state {state} within {timeout_s}s. Last seen state: {curr_state}."
            )

    def _get_workspace_status(self, workspace_id: Optional[str]) -> WorkspaceState:

        cluster = self.client.get_workspace_cluster(workspace_id)
        if not cluster:
            raise ValueError(f"Workspace {workspace_id} cluster not found.")
        return self._convert_cluster_state_to_workspace_state(cluster.state)

    def generate_ssh_config_file(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        ssh_config_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )

        assert (
            self.status(id=workspace_model.id) == WorkspaceState.RUNNING
        ), "Workspace must be running to generate SSH config file"

        head_node_ip = self.client.get_cluster_head_node_ip(workspace_model.cluster_id)
        ssh_key = self.client.get_cluster_ssh_key(workspace_model.cluster_id)

        def _store_ssh_key(name: str, ssh_key: str, key_dir: str) -> str:
            key_path = os.path.join(key_dir, f"{name}.pem")
            os.makedirs(os.path.dirname(key_path), exist_ok=True)

            with open(key_path, "w", opener=partial(os.open, mode=0o600)) as f:
                f.write(ssh_key)

            return key_path

        if ssh_config_path is None:
            ssh_config_path = tempfile.mkdtemp()

        key_path = _store_ssh_key(
            ssh_key.key_name, ssh_key.private_key, ssh_config_path
        )

        ssh_config = SSH_TEMPLATE.format(
            head_node_ip=head_node_ip, key_path=key_path, name=workspace_model.name
        )

        config_file_name = os.path.join(ssh_config_path, "config")
        with open(config_file_name, "w") as f:
            f.write(ssh_config)

        return workspace_model.name, config_file_name

    def run_command(
        self,
        command: str,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )
        host_name, config_file = self.generate_ssh_config_file(id=workspace_model.id)
        return subprocess.run(
            ["ssh"]
            + ANYSCALE_WORKSPACES_SSH_OPTIONS
            + ["-F", config_file, host_name, command],
            check=kwargs.pop("check", False),
            **kwargs,
        )

    def get_default_dir_name(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )
        return self.client.get_workspace_default_dir_name(workspace_model.id)

    def _parse_rsync_dry_run_output(
        self, dry_run_output: str
    ) -> Tuple[List[str], List[str]]:
        """Parse rsync dry-run output to detect file changes vs additions.
        Note that --itemize-changes is needed to detect file changes.
        Format is like "<fcsT...... test.py" where:
        - First char indicates operations (>, <, *)
        - Third char (c or .) indicates if checksum changed (only if --checksum is used)

        Returns:
            Tuple of (modifying_files, deleting_files)
        """
        modifying_files = []
        deleting_files = []

        ITEMISZED_PREFIX_LEN = 12

        changes = dry_run_output.strip().split("\n")
        for line in changes:
            if (
                not line or len(line) < ITEMISZED_PREFIX_LEN
            ):  # Need at least "<fcsT......" part
                continue
            if line.startswith("*deleting"):
                deleting_files.append(line[ITEMISZED_PREFIX_LEN:])
            elif line[2] == "c":  # Check the checksum position
                modifying_files.append(line[ITEMISZED_PREFIX_LEN:])

        return modifying_files, deleting_files

    def _dry_run_rsync(self, rsync_command: List[str], delete: bool):
        """Run rsync with --dry-run and warn if files are being deleted.
        """

        # --itemize-changes is needed to detect file changes
        dry_run_options = ["--dry-run", "--itemize-changes"]
        should_warn_delete = False

        if not delete:
            # --delete is needed to detect files that are being deleted in the destination
            should_warn_delete = True
            dry_run_options.append("--delete")

        try:
            result = subprocess.run(
                rsync_command + dry_run_options,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Error running rsync command: {e}")
            self._logger.error(f">>> stdout: {e.stdout}")
            self._logger.error(f">>> stderr: {e.stderr}")
            raise RuntimeError(f"Rsync failed with return code {e.returncode}")

        _, deleting_files = self._parse_rsync_dry_run_output(result.stdout)

        if should_warn_delete and len(deleting_files):
            click.echo(
                "Detected files that exist in the destination but not in the source. The files will not be deleted by default. You can add '--delete' option to delete the files:"
            )
            click.echo(
                "\n".join([click.style(file, fg="red") for file in deleting_files])
            )

    def _get_https_public_hostname(self, cluster) -> Optional[str]:
        """Extract public hostname from cluster for HTTPS connection."""
        # Attempt 1: Use cluster.hostname
        if hasattr(cluster, "hostname") and cluster.hostname:
            return cluster.hostname

        # Attempt 2: Fallback to parsing webterminal_auth_url
        if hasattr(cluster, "webterminal_auth_url") and cluster.webterminal_auth_url:
            parsed_url = urlparse(cluster.webterminal_auth_url)
            if parsed_url.netloc:
                return parsed_url.netloc

        return None

    def _get_https_cluster_access_token(self, cluster_id: str) -> Optional[str]:
        """Get cluster access token for HTTPS connection."""
        try:
            cluster_access_token = self.client._internal_api_client.get_cluster_access_token_api_v2_authentication_cluster_id_cluster_access_token_get(  # noqa: SLF001
                cluster_id=cluster_id
            )
            if not cluster_access_token:
                return None
            # Handle both str and bytes return types from API
            if isinstance(cluster_access_token, bytes):
                return cluster_access_token.decode("utf-8")
            return str(cluster_access_token)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
            self._logger.debug(
                f"Failed to get cluster access token: {type(e).__name__}"
            )
            return None

    def _create_https_proxy_command(
        self, public_hostname: str, cluster_access_token: str
    ) -> Optional[str]:
        """Create the proxy command for HTTPS connection."""
        wss_url = f"wss://{public_hostname}{WSS_PATH}"

        try:
            with importlib.resources.path(
                "anyscale.utils", "ssh_websocket_proxy.py"
            ) as proxy_path:
                proxy_script_path = str(proxy_path)
        except (ModuleNotFoundError, ImportError) as e:
            self._logger.debug(
                f"Failed to create HTTPS proxy command: {type(e).__name__}"
            )
            return None

        return " ".join(
            [
                shlex.quote(sys.executable),
                shlex.quote(proxy_script_path),
                shlex.quote(wss_url),
                shlex.quote(cluster_access_token),
            ]
        )

    def _build_https_ssh_command(
        self, workspace_id: str, config_file: str
    ) -> Optional[str]:
        """Build SSH command string with WebSocket ProxyCommand for rsync -e.

        Returns None if HTTPS connection cannot be set up.
        """
        try:
            cluster = self.client.get_workspace_cluster(workspace_id)
            if not cluster or not cluster.id:
                return None

            public_hostname = self._get_https_public_hostname(cluster)
            if not public_hostname:
                return None

            cluster_access_token = self._get_https_cluster_access_token(str(cluster.id))
            if not cluster_access_token:
                return None

            proxy_cmd = self._create_https_proxy_command(
                public_hostname, cluster_access_token
            )
            if not proxy_cmd:
                return None

            # Build SSH command with ProxyCommand for HTTPS tunnel
            ssh_options = subprocess.list2cmdline(ANYSCALE_WORKSPACES_SSH_OPTIONS)
            return (
                f"ssh -F {config_file} "
                f"-p {HTTPS_PORT} "
                f"-o ProxyCommand='{proxy_cmd}' "
                f"-o {PREFERRED_AUTH_METHOD} "
                f"{ssh_options}"
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
            self._logger.debug(f"Failed to build HTTPS SSH command: {type(e).__name__}")
            return None

    def _execute_rsync_with_ssh_cmd(
        self, ssh_cmd: str, base_rsync_args: List[str], delete: bool,
    ) -> None:
        """Execute rsync with the given SSH command."""
        # Use -c (--checksum) to avoid retransmitting files that haven't changed
        args = ["rsync", "-rzlc", "-e", ssh_cmd] + base_rsync_args

        # Run dry-run first to detect deletions
        self._dry_run_rsync(args, delete)

        # Add --progress for real-time feedback (shows filenames and per-file transfer progress)
        args_with_progress = args + ["--progress"]
        try:
            subprocess.run(
                args_with_progress, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f">>> Error running rsync command: {e}")
            self._logger.error(f">>> stdout: {e.stdout}")
            self._logger.error(f">>> stderr: {e.stderr}")
            raise RuntimeError(f"Rsync failed with return code {e.returncode}")

    def _run_rsync(  # noqa: PLR0913
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        local_dir: Optional[str] = None,
        rsync_args: Optional[List[str]] = None,
        is_pull: bool = False,
        include_git_state: bool = False,
        delete: bool = False,
        direct_ssh: bool = False,
    ):
        workspace_model = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )

        default_dir_name = self.get_default_dir_name(id=workspace_model.id)

        local_dir = local_dir or os.getcwd()

        with tempfile.TemporaryDirectory() as tmp_dir:
            host_name, config_file = self.generate_ssh_config_file(
                id=workspace_model.id, ssh_config_path=tmp_dir
            )

            if is_pull:
                source = f"ray@{host_name}:~/{default_dir_name}/"
                destination = local_dir
            else:
                # Ensure source directory has trailing slash for consistent rsync behavior
                # This ensures we sync the contents of the directory, not the directory itself
                source = os.path.join(local_dir, "")  # Ensures trailing slash
                destination = f"ray@{host_name}:~/{default_dir_name}/"

            rsync_args = rsync_args or []

            # exclude .git/objects/info/alternates since we will repack the git repo if needed
            # exclude .anyscale.yaml for legacy reasons
            rsync_args += [
                "--exclude",
                ".git/objects/info/alternates",
                "--exclude",
                ".anyscale.yaml",
            ]

            # repack git repos if needed
            if include_git_state and is_pull:
                self.run_command(
                    id=workspace_model.id,
                    command=f"cd ~/{default_dir_name} && python -m snapshot_util repack_git_repos",
                )
            elif not include_git_state:
                rsync_args += ["--exclude", ".git"]

            # Build base rsync args (without -e ssh command)
            base_rsync_args = [source, destination]
            if delete:
                # --delete removes files in the destination that are not in the source
                # Note: This preserves excluded files (like .git) and is compatible with all rsync implementations
                base_rsync_args.insert(0, "--delete")
            if rsync_args:
                base_rsync_args = rsync_args + base_rsync_args

            # Determine which SSH method to use (no fallback - use one or the other)
            if direct_ssh:
                # User explicitly requested direct SSH (port 22)
                ssh_cmd = (
                    f"ssh -F {config_file} "
                    f"{subprocess.list2cmdline(ANYSCALE_WORKSPACES_SSH_OPTIONS)}"
                )
                click.echo("Syncing via direct SSH connection (port 22)...")
            else:
                # Use HTTPS if it can be configured, otherwise use direct SSH
                # Note: No automatic fallback on transfer failure - if HTTPS fails mid-transfer,
                # the operation fails and user can retry with --direct-ssh flag
                https_ssh_cmd = self._build_https_ssh_command(
                    workspace_model.id, config_file
                )
                if https_ssh_cmd:
                    ssh_cmd = https_ssh_cmd
                else:
                    # HTTPS not configurable (e.g., missing cluster info), use direct SSH
                    ssh_cmd = (
                        f"ssh -F {config_file} "
                        f"{subprocess.list2cmdline(ANYSCALE_WORKSPACES_SSH_OPTIONS)}"
                    )

            # Execute rsync (no automatic fallback on failure)
            self._execute_rsync_with_ssh_cmd(ssh_cmd, base_rsync_args, delete)

    def pull(  # noqa: PLR0913
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        local_dir: Optional[str] = None,
        pull_git_state: bool = False,
        rsync_args: Optional[List[str]] = None,
        delete: bool = False,
        direct_ssh: bool = False,
    ):
        self._run_rsync(
            id=id,
            name=name,
            cloud=cloud,
            project=project,
            local_dir=local_dir,
            rsync_args=rsync_args,
            is_pull=True,
            include_git_state=pull_git_state,
            delete=delete,
            direct_ssh=direct_ssh,
        )

    def push(  # noqa: PLR0913
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        local_dir: Optional[str] = None,
        push_git_state: bool = False,
        rsync_args: Optional[List[str]] = None,
        delete: bool = False,
        direct_ssh: bool = False,
    ):
        self._run_rsync(
            id=id,
            name=name,
            cloud=cloud,
            project=project,
            local_dir=local_dir,
            is_pull=False,
            include_git_state=push_git_state,
            rsync_args=rsync_args,
            delete=delete,
            direct_ssh=direct_ssh,
        )

    def update(
        self, *, id: Optional[str], config: UpdateWorkspaceConfig  # noqa: A002
    ) -> str:
        workspace = self.client.get_workspace(id=id)  # type: ignore

        if not workspace:
            raise ValueError(f"Workspace with id '{id}' was not found.")

        current_status = self._get_workspace_status(id)
        if current_status != WorkspaceState.TERMINATED:
            raise ValueError(
                "Workspace must be in the TERMINATED state to be updated. Use `anyscale workspace terminate` to terminate the workspace, and `anyscale workspace wait` to wait for the workspace to terminate."
            )

        name = config.name or workspace.name

        compute_config_id = None
        if config.compute_config:
            compute_config_id, _ = self.resolve_compute_config_and_cloud_id(
                compute_config=config.compute_config, cloud=None,  # type: ignore
            )

        build_id = None
        if config.containerfile is not None:
            build_id = self._image_sdk.build_image_from_containerfile(
                name=f"image-for-workspace-{name}",
                containerfile=self.get_containerfile_contents(config.containerfile),
                ray_version=config.ray_version,
            )
        elif config.image_uri is not None:
            build_id = self._image_sdk.registery_image(
                image_uri=config.image_uri,
                registry_login_secret=config.registry_login_secret,
                ray_version=config.ray_version,
            )

        dynamic_requirements = None
        if (
            config.requirements
            and self._image_sdk.enable_image_build_for_tracked_requirements
        ):
            requirements = (
                parse_requirements_file(config.requirements)
                if isinstance(config.requirements, str)
                else config.requirements
            )
            if requirements:
                build_id = self._image_sdk.build_image_from_requirements(
                    name=f"image-for-workspace-{name}",
                    base_build_id=self.client.get_default_build_id(),
                    requirements=requirements,
                )
        elif config.requirements:
            dynamic_requirements = (
                parse_requirements_file(config.requirements)
                if isinstance(config.requirements, str)
                else config.requirements
            )

        self.client.update_workspace(
            workspace_id=id,
            name=config.name,
            compute_config_id=compute_config_id,
            cluster_environment_build_id=build_id,
            idle_timeout_minutes=config.idle_termination_minutes,
        )

        self._logger.info(f"Workspace updated successfully id: {id}")

        if dynamic_requirements:
            self.client.update_workspace_dependencies_offline_only(
                workspace_id=id, requirements=dynamic_requirements
            )
            self._logger.info(f"Applied dynamic requirements to workspace id: {id}")
        if config.env_vars:
            self.client.update_workspace_env_vars_offline_only(
                workspace_id=id, env_vars=config.env_vars
            )
            self._logger.info(f"Applied environment variables to workspace id: {id}")

        return id  # type: ignore

    def _convert_cluster_state_to_workspace_state(
        self, state: SessionState
    ) -> WorkspaceState:
        return cast(
            WorkspaceState,
            self._BACKEND_SESSION_STATE_TO_WORKSPACE_STATE.get(  # type: ignore
                state, WorkspaceState.UNKNOWN
            ),
        )

    def _convert_env_var_list_to_dict(
        self, env_vars: Optional[List[str]]
    ) -> Dict[str, str]:
        if not env_vars:
            return {}
        return dict([env_var.split("=", 1) for env_var in env_vars])

    def _convert_requirements_str_to_list(
        self, requirements: Optional[str]
    ) -> List[str]:
        if not requirements:
            return []
        return [req for req in requirements.split("\n") if req]

    def _to_workspace(
        self, workspace: ExperimentalWorkspace, *, include_config: bool = True
    ) -> Workspace:
        """Convert ExperimentalWorkspace to Workspace.

        Args:
            workspace: Internal workspace model from API
            include_config: If True, fetch full config (expensive). If False, only metadata.
        """
        state = self._BACKEND_SESSION_STATE_TO_WORKSPACE_STATE.get(
            workspace.state, WorkspaceState.UNKNOWN
        )

        # Optionally fetch expensive config data
        config = self._fetch_workspace_config(workspace) if include_config else None

        return Workspace(
            id=workspace.id or "",
            name=workspace.name or "",
            state=state,
            project_id=workspace.project_id,
            cloud_id=workspace.cloud_id,
            creator_id=workspace.creator_id,
            creator_email=workspace.creator_email,
            created_at=workspace.created_at,
            last_started_at=workspace.latest_started_at,
            cluster_id=workspace.cluster_id,
            config=config,
        )

    def _fetch_workspace_config(
        self, workspace: ExperimentalWorkspace
    ) -> WorkspaceConfig:
        """Fetch full workspace configuration (expensive, makes multiple API calls).

        Optimizes performance by parallelizing independent API calls using ThreadPoolExecutor.
        ThreadPoolExecutor is appropriate here because the OpenAPI clients use synchronous I/O
        (requests library), not async/await.
        """
        # First fetch cluster (needed for subsequent calls)
        cluster = self.client.get_workspace_cluster(workspace.id)
        if not cluster:
            raise ValueError(
                f"Workspace cluster with ID '{workspace.cluster_id}' was not found."
            )

        # Parallelize all independent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all fetches in parallel - more Pythonic dict pattern
            futures = {
                "image_uri": executor.submit(
                    self._image_sdk.get_image_uri_from_build_id, cluster.build_id
                ),
                "image_build": executor.submit(
                    self._image_sdk.get_image_build, cluster.build_id
                ),
                "compute": executor.submit(
                    self.get_user_facing_compute_config, workspace.compute_config_id
                ),
                "cloud": executor.submit(
                    self.client.get_cloud, cloud_id=workspace.cloud_id
                ),
                "project": executor.submit(
                    self.client.get_project, project_id=workspace.project_id
                ),
                "artifacts": executor.submit(
                    self.client.get_workspace_proxied_dataplane_artifacts,
                    workspace_id=workspace.id,
                ),
            }

            # Collect all results - will raise exception if any call failed
            results = {key: future.result() for key, future in futures.items()}

        # Validate required results
        if not results["image_uri"]:
            raise ValueError(f"Failed to get image URI for build {cluster.build_id}.")
        if not results["image_build"]:
            raise ValueError(f"Failed to get image build {cluster.build_id}.")

        env_vars = self._convert_env_var_list_to_dict(
            results["artifacts"].environment_variables
        )
        requirements = self._convert_requirements_str_to_list(
            results["artifacts"].requirements
        )

        return WorkspaceConfig(
            name=workspace.name,
            compute_config=results["compute"],
            registry_login_secret=results["image_build"].registry_login_secret,
            image_uri=str(results["image_uri"]),
            requirements=requirements if requirements else None,
            idle_termination_minutes=cluster.idle_timeout,
            env_vars=env_vars,
            project=results["project"].name if results["project"] else None,
            cloud=results["cloud"].name if results["cloud"] else None,
        )

    def get(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_config: bool = True,
    ) -> Workspace:
        """Get a workspace by name or ID.

        Args:
            include_config: If True, fetch full config (default). Set to False for efficiency.
        """
        workspace = self._resolve_to_workspace_model(
            id=id, name=name, cloud=cloud, project=project
        )
        return self._to_workspace(workspace, include_config=include_config)

    def list(  # noqa: PLR0913, PLR0917
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        state_filter: Optional[Union[List[WorkspaceState], List[str]]] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        include_config: bool = False,
        sort_field: Optional[Union[str, WorkspaceSortField]] = None,
        sort_order: Optional[Union[str, WorkspaceSortOrder]] = None,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> ResultIterator[Workspace]:
        """List workspaces with optional filters.

        Args:
            state_filter: List of states to include. May be WorkspaceState enums or case-insensitive strings.
                         If None, defaults to non-terminated states.
            tags_filter: Filter by tags. Dict mapping tag keys to lists of values.
                        Tags with the same key are ORed, different keys are ANDed.
            include_config: If True, fetch full config for each workspace (expensive, makes multiple API calls).
                           Defaults to False for efficiency. Set to True when full config is needed.
            sort_field: Field to sort by. Defaults to status (active first) then created_at (desc).
            sort_order: Sort order (ASC or DESC). Defaults to appropriate order for sort_field.
        """
        MAX_PAGE_SIZE = 100

        if page_size is not None and not (1 <= page_size <= MAX_PAGE_SIZE):
            raise ValueError(
                f"page_size must be between 1 and {MAX_PAGE_SIZE}, inclusive."
            )

        # Resolve cloud and project to IDs
        cloud_id = self.client.get_cloud_id(cloud_name=cloud) if cloud else None
        project_id = (
            self.client.get_project_id(parent_cloud_id=cloud_id, name=project)
            if project
            else None
        )

        # Normalize state filter
        # Backend expects SessionState enum values as strings
        normalized_states = _normalize_state_filter_to_backend(state_filter)

        # Convert tags_filter dict to backend format (list of "key:value" strings)
        tag_filter_list: Optional[List[str]] = None
        if tags_filter:
            tag_filter_list = []
            for key, values in tags_filter.items():
                for value in values:
                    tag_filter_list.append(f"{key}:{value}")

        # Handle single workspace lookup by ID
        if workspace_id is not None:
            workspace = self.client.get_workspace(id=workspace_id)

            if workspace is None:
                # Return empty iterator
                return ResultIterator(
                    page_token=None,
                    max_items=0,
                    fetch_page=lambda _: SimpleNamespace(
                        results=[], metadata=SimpleNamespace(next_paging_token=None),
                    ),
                    parse_fn=None,
                )

            # Return single-item iterator
            summary = self._to_workspace(workspace, include_config=include_config)
            return ResultIterator(
                page_token=None,
                max_items=1,
                fetch_page=lambda token: SimpleNamespace(
                    results=[summary] if token is None else [],
                    metadata=SimpleNamespace(next_paging_token=None),
                ),
                parse_fn=None,
            )

        # Handle paginated list
        def _fetch_page(token: Optional[str]) -> SimpleNamespace:
            # Use internal API client directly as there's no interface method for list_workspaces yet
            response = self.client._internal_api_client.list_workspaces_api_v2_experimental_workspaces_get(  # noqa: SLF001
                project_id=project_id,
                cloud_id=cloud_id,
                name=name,
                creator_id=creator_id,
                state_filter=normalized_states,
                tag_filter=tag_filter_list,
                sort_field=sort_field,
                sort_order=sort_order,
                count=page_size or 50,
                paging_token=token,
            )

            results = response.results if response.results else []
            next_token = (
                response.metadata.next_paging_token if response.metadata else None
            )

            return SimpleNamespace(
                results=results, metadata=SimpleNamespace(next_paging_token=next_token),
            )

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=lambda ws: self._to_workspace(ws, include_config=include_config),
        )

    def add_tags(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        tags: Dict[str, str],
    ) -> None:
        """Upsert (add/update) tag key/value pairs for a workspace."""
        if not tags:
            raise ValueError("At least one tag must be provided.")

        if id is not None:
            resource_id = id
        else:
            if name is None and not self.client.inside_workspace():
                raise ValueError(
                    "Either 'id' or 'name' must be provided when running outside of a workspace."
                )
            model = self._resolve_to_workspace_model(
                id=id, name=name, cloud=cloud, project=project
            )
            resource_id = model.id  # type: ignore

        self.client.upsert_resource_tags(
            ResourceTagResourceType.WORKSPACE, resource_id, tags
        )

    def remove_tags(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        keys: List[str],
    ) -> None:
        """Remove tag keys from a workspace."""
        if not keys:
            raise ValueError("At least one tag key must be provided.")

        if id is not None:
            resource_id = id
        else:
            if name is None and not self.client.inside_workspace():
                raise ValueError(
                    "Either 'id' or 'name' must be provided when running outside of a workspace."
                )
            model = self._resolve_to_workspace_model(
                id=id, name=name, cloud=cloud, project=project
            )
            resource_id = model.id  # type: ignore

        self.client.delete_resource_tags(
            ResourceTagResourceType.WORKSPACE, resource_id, keys
        )

    def list_tags(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, str]:
        """List tags for a workspace as a key/value mapping."""
        if id is not None:
            resource_id = id
        else:
            model = self._resolve_to_workspace_model(
                id=id, name=name, cloud=cloud, project=project
            )
            resource_id = model.id  # type: ignore

        records = self.client.list_resource_tags(
            ResourceTagResourceType.WORKSPACE, resource_id
        )
        return {r.key: r.value for r in records if r and r.key is not None}


def _normalize_state_filter_to_backend(
    states: Optional[Union[List[WorkspaceState], List[str]]],
) -> Optional[List[str]]:
    """Normalize frontend WorkspaceState filter to backend SessionState values.

    Maps frontend WorkspaceState enums to their corresponding backend SessionState values.
    None or empty list: no filtering (includes all states).
    """
    # Map WorkspaceState to backend SessionState values
    WORKSPACE_TO_SESSION_STATES: Dict[WorkspaceState, List[str]] = {
        WorkspaceState.STARTING: [
            "StartingUp",
            "AwaitingStartup",
            "AwaitingFileMounts",
        ],
        WorkspaceState.RUNNING: ["Running"],
        WorkspaceState.UPDATING: ["Updating"],
        WorkspaceState.TERMINATING: ["Stopping", "Terminating"],
        WorkspaceState.TERMINATED: ["Stopped", "Terminated"],
        WorkspaceState.ERRORED: [
            "UpdatingErrored",
            "StartupErrored",
            "TerminatingErrored",
            "StoppingErrored",
        ],
        WorkspaceState.UNKNOWN: [],
    }

    # None or empty list means no filtering
    if states is None or len(states) == 0:
        return None

    backend_states: List[str] = []
    for s in states:
        if isinstance(s, WorkspaceState):
            backend_states.extend(WORKSPACE_TO_SESSION_STATES.get(s, []))
        elif isinstance(s, str):
            backend_states.append(s)
        else:
            raise TypeError(
                "'state_filter' entries must be WorkspaceState or str, "
                f"got {type(s).__name__}"
            )
    return backend_states if backend_states else None

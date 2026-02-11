import copy
from dataclasses import replace
import os
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from anyscale._private.anyscale_client import (
    AnyscaleClientInterface,
    WORKSPACE_CLUSTER_NAME_PREFIX,
)
from anyscale._private.models.integrations import (
    CONNECTION_TYPE_TO_INTEGRATION_TYPE,
    ConnectionConfig,
    IntegrationType,
)
from anyscale._private.sdk.base_sdk import BaseSDK, Timer
from anyscale.cli_logger import BlockLogger
from anyscale.compute_config._private.compute_config_sdk import PrivateComputeConfigSDK
from anyscale.compute_config.models import (
    ComputeConfig,
    ComputeConfigType,
    MultiResourceComputeConfig,
)
from anyscale.image._private.image_sdk import PrivateImageSDK
from anyscale.utils.runtime_env import is_dir_remote_uri, parse_requirements_file


# Keep in sync with Ray's default excludes in python/ray/_private/ray_constants.py
_RAY_RUNTIME_ENV_DEFAULT_EXCLUDES = ".git,.venv,venv,__pycache__"
_RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES_ENV_VAR = (
    "RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES"
)
_WARNED_DEFAULT_EXCLUDES: Set[str] = set()


def _get_runtime_env_default_excludes() -> List[str]:
    """Mirror Ray's default excludes for packaging local runtime_env directories."""
    val = os.environ.get(
        _RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES_ENV_VAR,
        _RAY_RUNTIME_ENV_DEFAULT_EXCLUDES,
    )
    return [x.strip() for x in val.split(",") if x.strip()]


def _warn_if_default_excludes_present(
    *, target_dir: str, default_excludes: List[str], logger: BlockLogger
) -> None:
    # TODO(ricardo): 2026-01-07 Remove these warnings in a few releases. Added in
    # case users rely on these directories being uploaded with their working_dir
    # since this change would be difficult to debug.

    # Only warn for local directories.
    if is_dir_remote_uri(target_dir):
        return
    if not default_excludes:
        return

    base = pathlib.Path(target_dir)
    for d in default_excludes:
        # Skip if we've already warned about this exclude.
        if d in _WARNED_DEFAULT_EXCLUDES:
            continue
        if (base / d).exists():
            _WARNED_DEFAULT_EXCLUDES.add(d)
            logger.warning(
                f"Directory {d!r} is now ignored by default when packaging the working "
                "directory. To disable this behavior, set "
                "the `RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES=''` environment variable."
            )


class WorkloadSDK(BaseSDK):
    """Shared parent class for job, job queue, workspace, and service SDKs."""

    def __init__(
        self,
        *,
        logger: Optional[BlockLogger] = None,
        client: Optional[AnyscaleClientInterface] = None,
        timer: Optional[Timer] = None,
    ):
        super().__init__(logger=logger, client=client, timer=timer)
        self._compute_config_sdk = PrivateComputeConfigSDK(
            logger=self.logger, client=self.client,
        )
        self._image_sdk = PrivateImageSDK(logger=self.logger, client=self.client)

    @property
    def image_sdk(self) -> PrivateImageSDK:
        return self._image_sdk

    def update_env_vars(
        self,
        runtime_envs: List[Dict[str, Any]],
        *,
        env_vars_updates: Optional[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Replaces 'env_vars' fields in runtime_envs with the override."""
        new_runtime_envs = copy.deepcopy(runtime_envs)

        if env_vars_updates:
            for runtime_env in new_runtime_envs:
                if "env_vars" in runtime_env:
                    # the precedence should be config > runtime_env
                    runtime_env["env_vars"].update(env_vars_updates)
                else:
                    runtime_env["env_vars"] = env_vars_updates

        return new_runtime_envs

    def override_and_load_requirements_files(
        self,
        runtime_envs: List[Dict[str, Any]],
        *,
        requirements_override: Union[None, str, List[str]],
        workspace_requirements_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Replaces 'pip' fields in runtime_envs with their parsed file contents.

        The precedence for overrides is: explicit overrides passed in > fields in the existing
        runtime_envs > workspace defaults (if `autopopulate_in_workspace == True`).
        """
        new_runtime_envs = copy.deepcopy(runtime_envs)

        local_path_to_parsed_requirements: Dict[str, List[str]] = {}

        def _load_requirements_file_memoized(
            target: Union[str, List[str]]
        ) -> List[str]:
            if isinstance(target, list):
                return target
            elif target in local_path_to_parsed_requirements:
                return local_path_to_parsed_requirements[target]
            elif isinstance(target, str):
                parsed_requirements = parse_requirements_file(target)
                if parsed_requirements is None:
                    raise FileNotFoundError(
                        f"Requirements file {target} does not exist."
                    )
                local_path_to_parsed_requirements[target] = parsed_requirements
                return parsed_requirements
            else:
                raise TypeError("pip field in runtime_env must be a list or string.")

        for runtime_env in new_runtime_envs:
            if requirements_override is not None:
                # Explicitly-specified override from the user.
                runtime_env["pip"] = requirements_override
            elif (
                workspace_requirements_path is not None
                and "pip" not in runtime_env
                and "conda" not in runtime_env
            ):
                self.logger.info("Including workspace-managed pip dependencies.")
                runtime_env["pip"] = workspace_requirements_path

            if runtime_env.get("pip", None) is not None:
                # Load requirements from the file if necessary.
                runtime_env["pip"] = _load_requirements_file_memoized(
                    runtime_env["pip"]
                )

        return new_runtime_envs

    def override_and_upload_local_dirs_single_deployment(  # noqa: PLR0912
        self,
        runtime_envs: List[Dict[str, Any]],
        *,
        working_dir_override: Optional[str],
        excludes_override: Optional[List[str]],
        cloud_id: Optional[str] = None,
        autopopulate_in_workspace: bool = True,
        additional_py_modules: Optional[List[str]] = None,
        py_executable_override: Optional[str] = None,
        cloud_resource_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns modified runtime_envs with all local dirs converted to remote URIs.

        The precedence for overrides is: explicit overrides passed in > fields in the existing
        runtime_envs > workspace defaults (if `autopopulate_in_workspace == True`).

        Each unique local directory across these fields will be uploaded once to cloud storage,
        then all occurrences of it in the config will be replaced with the corresponding remote URI.
        """
        new_runtime_envs = copy.deepcopy(runtime_envs)

        local_path_to_uri: Dict[str, str] = {}
        default_excludes = _get_runtime_env_default_excludes()

        def _upload_dir_memoized(target: str, *, excludes: Optional[List[str]]) -> str:
            if is_dir_remote_uri(target):
                return target
            if target in local_path_to_uri:
                return local_path_to_uri[target]

            self.logger.info(f"Uploading local dir '{target}' to cloud storage.")
            assert cloud_id is not None
            _warn_if_default_excludes_present(
                target_dir=target, default_excludes=default_excludes, logger=self.logger
            )
            uri = self._client.upload_local_dir_to_cloud_storage(
                target,
                cloud_id=cloud_id,
                excludes=excludes,
                cloud_resource_name=cloud_resource_name,
            )
            local_path_to_uri[target] = uri
            return uri

        for runtime_env in new_runtime_envs:
            # Extend, don't overwrite, excludes if it's provided.
            if excludes_override is not None:
                existing_excludes = runtime_env.get("excludes", None) or []
                runtime_env["excludes"] = existing_excludes + excludes_override

            user_excludes = runtime_env.get("excludes") or []
            final_excludes = default_excludes + list(user_excludes)

            new_working_dir = None
            if working_dir_override is not None:
                new_working_dir = working_dir_override
            elif "working_dir" in runtime_env:
                new_working_dir = runtime_env["working_dir"]
            elif autopopulate_in_workspace and self._client.inside_workspace():
                new_working_dir = "."

            if new_working_dir is not None:
                runtime_env["working_dir"] = _upload_dir_memoized(
                    new_working_dir, excludes=final_excludes
                )

            if additional_py_modules:
                existing_py_modules = runtime_env.get("py_modules", [])
                runtime_env["py_modules"] = existing_py_modules + additional_py_modules

            final_py_modules = runtime_env.get("py_modules", None)
            if final_py_modules is not None:
                runtime_env["py_modules"] = [
                    _upload_dir_memoized(py_module, excludes=final_excludes)
                    for py_module in final_py_modules
                ]

            if py_executable_override:
                runtime_env["py_executable"] = py_executable_override

        return new_runtime_envs

    def override_and_upload_local_dirs_multi_cloud_resource(  # noqa: PLR0912
        self,
        runtime_envs: List[Dict[str, Any]],
        *,
        working_dir_override: Optional[str],
        excludes_override: Optional[List[str]],
        cloud_resource_names: List[Optional[str]],
        cloud_id: Optional[str] = None,
        autopopulate_in_workspace: bool = True,
        additional_py_modules: Optional[List[str]] = None,
        py_executable_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns modified runtime_envs with all local dirs converted to remote bucket paths,
        stored in the "relative_working_dir" and "relative_py_modules" fields.

        The precedence for overrides is: explicit overrides passed in > fields in the existing
        runtime_envs > workspace defaults (if `autopopulate_in_workspace == True`).

        Each unique local directory across these fields will be uploaded once to cloud storage,
        then all occurrences of it in the config will be replaced with the corresponding remote URI.
        """
        new_runtime_envs = copy.deepcopy(runtime_envs)

        local_path_to_bucket_path: Dict[str, str] = {}
        default_excludes = _get_runtime_env_default_excludes()

        def _upload_dir_memoized(target: str, *, excludes: Optional[List[str]]) -> str:
            if target in local_path_to_bucket_path:
                return local_path_to_bucket_path[target]

            self.logger.info(
                f"Uploading local dir '{target}' to object storage for all {len(cloud_resource_names)} cloud resources in the compute config."
            )
            assert cloud_id is not None
            _warn_if_default_excludes_present(
                target_dir=target, default_excludes=default_excludes, logger=self.logger
            )
            bucket_path = self._client.upload_local_dir_to_cloud_storage_multi_cloud_resource(
                target,
                cloud_id=cloud_id,
                excludes=excludes,
                cloud_resource_names=cloud_resource_names,
            )
            local_path_to_bucket_path[target] = bucket_path
            return bucket_path

        for runtime_env in new_runtime_envs:
            # Extend, don't overwrite, excludes if it's provided.
            if excludes_override is not None:
                existing_excludes = runtime_env.get("excludes", None) or []
                runtime_env["excludes"] = existing_excludes + excludes_override

            user_excludes = runtime_env.get("excludes") or []
            final_excludes = default_excludes + list(user_excludes)

            new_working_dir = None
            if working_dir_override is not None:
                new_working_dir = working_dir_override
            elif "working_dir" in runtime_env:
                new_working_dir = runtime_env["working_dir"]
            elif autopopulate_in_workspace and self._client.inside_workspace():
                new_working_dir = "."

            if new_working_dir is not None:
                if is_dir_remote_uri(new_working_dir):
                    runtime_env["working_dir"] = new_working_dir
                else:
                    runtime_env["relative_working_dir"] = _upload_dir_memoized(
                        new_working_dir, excludes=final_excludes
                    )
                    runtime_env.pop("working_dir", None)

            if additional_py_modules:
                existing_py_modules = runtime_env.get("py_modules", [])
                runtime_env["py_modules"] = existing_py_modules + additional_py_modules

            final_py_modules = runtime_env.get("py_modules", None)
            if final_py_modules is not None:
                py_modules = [
                    py_module
                    for py_module in final_py_modules
                    if is_dir_remote_uri(py_module)
                ]
                if len(py_modules) > 0:
                    runtime_env["py_modules"] = py_modules
                else:
                    # If there are no py_modules, remove the field.
                    runtime_env.pop("py_modules", None)

                relative_py_modules = [
                    _upload_dir_memoized(py_module, excludes=final_excludes)
                    for py_module in final_py_modules
                    if not is_dir_remote_uri(py_module)
                ]
                if len(relative_py_modules) > 0:
                    runtime_env["relative_py_modules"] = relative_py_modules

            if py_executable_override:
                runtime_env["py_executable"] = py_executable_override

        return new_runtime_envs

    def _resolve_compute_config_id(
        self,
        compute_config: Union[str, ComputeConfigType, None],
        cloud: Optional[str] = None,
    ) -> str:
        """Resolve the passed compute config to its ID.

        Accepts either:
            - A string of the form: '<name>[:<version>]'.
            - A dictionary from which an anonymous compute config will be built.

        Returns compute_config_id.
        """
        if isinstance(compute_config, str):
            compute_config_id = self._client.get_compute_config_id(
                compute_config_name=compute_config, cloud=cloud,
            )
            if compute_config_id is None:
                raise ValueError(
                    f"The compute config '{compute_config}' does not exist."
                )
        elif compute_config is None:
            cloud_id = self.client.get_cloud_id(cloud_name=cloud)  # type: ignore
            compute_config_id = self._client.get_default_compute_config(
                cloud_id=cloud_id
            ).id
            if compute_config_id is None:
                raise ValueError(
                    f"The default compute config for cloud '{cloud}' does not exist."
                )
        else:
            _, compute_config_id = self._compute_config_sdk.create_compute_config(
                compute_config
            )

        return compute_config_id

    def resolve_compute_config_and_cloud_id(
        self,
        *,
        compute_config: Union[ComputeConfigType, Dict, str, None],
        cloud: Union[None, str],
    ) -> Tuple[str, str]:
        """Resolve the passed compute config to its ID and corresponding cloud ID.

        Accepts either:
            - A string of the form: '<name>[:<version>]'.
            - A dictionary from which an anonymous compute config will be built.

        Returns (compute_config_id, cloud_id).
        """
        if compute_config is None and cloud is None:
            compute_config_id = self._client.get_compute_config_id()
            assert compute_config_id is not None
            return (
                compute_config_id,
                self.client.get_cloud_id(compute_config_id=compute_config_id),
            )
        elif compute_config is None:
            cloud_id = self.client.get_cloud_id(cloud_name=cloud)  # type: ignore
            compute_config_id = self._client.get_default_compute_config(
                cloud_id=cloud_id
            ).id
            return (compute_config_id, cloud_id)  # type: ignore
        elif cloud is None:
            compute_config_id = self._resolve_compute_config_id(
                compute_config=compute_config  # type: ignore
            )
            return (
                compute_config_id,
                self.client.get_cloud_id(compute_config_id=compute_config_id),
            )
        else:
            # If we are creating a new compute config, ensure cloud is set accordingly
            if (
                isinstance(compute_config, (ComputeConfig, MultiResourceComputeConfig))
                and compute_config.cloud is None
            ):
                compute_config = replace(compute_config, cloud=cloud)

            compute_config_id = self._resolve_compute_config_id(
                compute_config=compute_config,  # type: ignore
                cloud=cloud,
            )
            cloud_id_from_cc = self.client.get_cloud_id(
                compute_config_id=compute_config_id
            )
            cloud_id_from_cloud = self.client.get_cloud_id(cloud_name=cloud)  # type: ignore
            if cloud_id_from_cc != cloud_id_from_cloud:
                raise ValueError(
                    "Cloud does not match from provided `cloud` and `compute_config`. Either pass one of these or ensure they match."
                )
            else:
                return (compute_config_id, cloud_id_from_cc)

    def get_current_workspace_name(self) -> Optional[str]:
        """Get the name of the current workspace if running inside one.

        Prefer fetching the workspace model by ID to reflect recent renames.
        Falls back to deriving from the current workspace cluster name.
        """
        if not self._client.inside_workspace():
            return None

        # First try to fetch the workspace by ID for the latest name.
        workspace_id = self._client.get_current_workspace_id()
        if workspace_id:
            ws = self._client.get_workspace(id=workspace_id)
            if ws is not None and getattr(ws, "name", None):
                return ws.name

        # Fallback: derive from the current workspace cluster name.
        cluster = self._client.get_current_workspace_cluster()
        assert cluster is not None
        name = cluster.name
        if name.startswith(WORKSPACE_CLUSTER_NAME_PREFIX):
            name = name[len(WORKSPACE_CLUSTER_NAME_PREFIX) :]

        return name

    def get_containerfile_contents(self, path: str) -> str:
        """Get the full content of the containerfile as a string."""
        containerfile_path = pathlib.Path(path)
        if not containerfile_path.exists():
            raise FileNotFoundError(
                f"Containerfile '{containerfile_path}' does not exist."
            )
        if not containerfile_path.is_file():
            raise ValueError(f"Containerfile '{containerfile_path}' must be a file.")

        return containerfile_path.read_text()

    def _strip_none_values_from_dict(self, d: Dict) -> Dict:
        """Return a copy of the dictionary without any keys whose values are None.

        Recursively calls into any dictionary values.
        """
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._strip_none_values_from_dict(v)
            elif v is not None:
                result[k] = v

        return result

    def get_user_facing_compute_config(
        self, compute_config_id: str,
    ) -> Union[str, ComputeConfigType]:
        """Get the compute config in a format to be displayed in a user-facing status.

        If the compute config refers to an anonymous compute config, its config
        object will be returned. Else the name of the compute config will be
        returned in the form: '<name>:<version>'.
        """
        compute_config = self._client.get_compute_config(compute_config_id)
        if compute_config is None:
            raise RuntimeError(
                f"Failed to get compute config for ID {compute_config_id}."
            )

        compute_config_name = compute_config.name
        if compute_config.version is not None:
            compute_config_name += f":{compute_config.version}"

        if not compute_config.anonymous:
            return compute_config_name

        return self._compute_config_sdk._convert_api_model_to_compute_config_version(  # noqa: SLF001
            compute_config
        ).config

    def get_integration_type_from_connection_type(
        self, connection_type: str
    ) -> IntegrationType:
        """Get the integration type from the connection type."""
        integration_type = CONNECTION_TYPE_TO_INTEGRATION_TYPE.get(connection_type)
        if integration_type is None:
            raise ValueError(f"Unsupported connection type: {connection_type}")
        return integration_type

    def get_connection_config_from_connection(
        self, connection: Any
    ) -> ConnectionConfig:
        """Get the connection config from the connection."""
        integration_type = self.get_integration_type_from_connection_type(
            connection.connection_type
        )
        return ConnectionConfig(
            integration_type=integration_type, connection_name=connection.name,
        )

    def resolve_connection_ids(
        self, connections: Optional[List[ConnectionConfig]]
    ) -> Optional[List[str]]:
        """Resolve ConnectionConfig objects to connection IDs.

        Looks up connections by name and integration type and returns their IDs.
        Raises ValueError if a connection is not found, if an invalid integration
        type is specified, or if duplicate integration types are specified.
        """
        if not connections:
            return None

        # Check for duplicate integration types (only one connection per type allowed)
        seen_types: set = set()
        for conn_config in connections:
            # integration_type is guaranteed to be IntegrationType by ConnectionConfig validation
            integration_type = conn_config.integration_type
            if integration_type in seen_types:
                raise ValueError(
                    f"Duplicate integration type '{integration_type.value}' specified. "
                    f"Only one connection per integration type is allowed."
                )
            seen_types.add(integration_type)

        # Get connection IDs for each connection
        connection_ids = []
        for conn_config in connections:
            fetched_connections = self.client.list_databricks_connections(
                name=conn_config.connection_name
            )
            if len(fetched_connections) == 0:
                raise ValueError(
                    f"Connection '{conn_config.connection_name}' not found. Please check that the "
                    f"connection exists in your organization settings and has the correct name."
                )
            connection_ids.append(fetched_connections[0].id)

        return connection_ids

    def resolve_connection_ids_to_configs(
        self, connection_ids: Optional[List[str]]
    ) -> List[ConnectionConfig]:
        """Resolve connection IDs back to ConnectionConfig objects.

        Used when fetching job/workspace status to convert stored connection IDs
        back to user-facing ConnectionConfig objects. This makes an API call for
        each connection ID.

        Prefer using `resolve_decorated_connections_to_configs` when decorated connection
        data is already available (e.g., from DecoratedProductionJob.connections).
        """
        if not connection_ids:
            return []

        connections = []
        for connection_id in connection_ids:
            connection = self.client.get_databricks_connection(connection_id)
            connections.append(self.get_connection_config_from_connection(connection))

        return connections

    def resolve_decorated_connections_to_configs(
        self, decorated_connections: Optional[List]
    ) -> List[ConnectionConfig]:
        """Convert decorated connection objects to ConnectionConfig objects.

        Used when decorated connection data is already available (e.g., from
        DecoratedProductionJob.connections), avoiding additional API calls.
        """
        if not decorated_connections:
            return []

        connections = []
        for decorated_conn in decorated_connections:
            connection = decorated_conn.connection
            connections.append(self.get_connection_config_from_connection(connection))

        return connections

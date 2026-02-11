import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import click
import yaml

from anyscale.anyscale_pydantic import (
    BaseModel,
    Extra,
    Field,
    root_validator,
)
from anyscale.cli_logger import LogsLogger
from anyscale.cluster_compute import (
    get_cluster_compute_from_name,
    get_default_cluster_compute,
    register_compute_template,
)
from anyscale.cluster_env import (
    get_build_from_cluster_env_identifier,
    get_default_cluster_env_build,
    validate_successful_build,
)
from anyscale.project_utils import (
    get_parent_cloud_id_and_name_of_project,
    get_proj_id_from_name,
)
from anyscale.shared_anyscale_utils.utils.byod import BYODInfo
from anyscale.util import (
    extract_versions_from_image_name,
    PROJECT_NAME_ENV_VAR,
)


log = LogsLogger()


def _validate_conda_option(conda_option: Union[str, Dict]) -> Union[str, Dict]:
    """Parses and validates a user-provided 'conda' option.

    Can be one of three cases:
        1) A str that's the name of a pre-installed conda environment.
        2) A string pointing to a local conda environment YAML. In this case,
           the file contents will be read into a dict.
        3) A dict that defines a conda environment. This is passed through.
    """
    result = None
    if isinstance(conda_option, str):
        yaml_file = Path(conda_option)
        if yaml_file.suffix in (".yaml", ".yml"):
            if not yaml_file.is_file():
                raise click.ClickException(f"Can't find conda YAML file {yaml_file}.")
            try:
                result = yaml.safe_load(yaml_file.read_text())
            except Exception as e:  # noqa: BLE001
                raise click.ClickException(
                    f"Failed to read conda file {yaml_file}: {e}."
                )
        else:
            # Assume it's a pre-existing conda environment name.
            result = conda_option
    elif isinstance(conda_option, dict):
        result = conda_option

    return result


def _validate_pip_option(pip_option: Union[str, List[str]]) -> Optional[List[str]]:
    """Parses and validates a user-provided 'pip' option.

    Can be one of two cases:
        1) A List[str] describing the requirements. This is passed through.
        2) A string pointing to a local requirements file. In this case, the
           file contents will be read split into a list.
    """
    result = None
    if isinstance(pip_option, str):
        # We have been given a path to a requirements.txt file.
        pip_file = Path(pip_option)
        if not pip_file.is_file():
            raise click.ClickException(f"{pip_file} is not a valid file.")
        result = pip_file.read_text().strip().split("\n")
    elif isinstance(pip_option, list) and all(
        isinstance(dep, str) for dep in pip_option
    ):
        result = None if len(pip_option) == 0 else pip_option

    return result


def _validate_py_modules(py_modules_option: List[str]) -> List[str]:
    for entry in py_modules_option:
        if "://" not in entry:
            raise click.ClickException(
                "Only remote URIs are currently supported for py_modules in the job "
                "config (not local directories). Please see "
                "https://docs.ray.io/en/master/handling-dependencies.html#remote-uris for supported options."
            )

    return py_modules_option


def _working_dir_is_remote_uri(working_dir: str) -> bool:
    return "://" in working_dir


def _validate_working_dir(working_dir_option: str) -> str:
    """If working_dir is a local directory, check that it exists."""
    # We have been given a path to a local directory.
    if (
        not _working_dir_is_remote_uri(working_dir_option)
        and not Path(working_dir_option).is_dir()
    ):
        raise click.ClickException(
            f"working_dir {working_dir_option} is not a valid local directory or remote URI."
        )

    return working_dir_option


def _validate_working_dir_and_upload_path(
    working_dir: Optional[str], upload_path: Optional[str]
):
    """Check that the combination of working_dir and upload_path is valid.

    Exception should be thrown if both working dir is a remote uri and upload path is defined
    Otherwise, all other permutations are valid
    """
    if (
        upload_path is not None
        and working_dir
        and _working_dir_is_remote_uri(working_dir)
    ):
        raise click.ClickException(
            f"`upload_path` was specified, but `working_dir` is not a local directory.  Recieved `upload_path`: {upload_path} and `working_dir`: {working_dir}."
        )
    elif upload_path is not None and not working_dir:
        raise click.ClickException(
            f"`upload_path` was specified, but no `working_dir` is defined. Recieved `upload_path`: {upload_path} and `working_dir`: None."
        )


def _validate_upload_path(upload_path: str) -> str:
    """Check that the upload path is a valid S3 or GS remote URI."""
    try:
        parsed_upload_path = urlparse(upload_path)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Failed to parse `upload_path` {upload_path} as a URI (e.g. 's3://my_bucket/my_dir'): {e}"
        )
    if parsed_upload_path.scheme not in ["s3", "gs"]:
        raise click.ClickException(
            f"Only Amazon S3 (e.g. 's3://bucket', 's3://bucket/path') and Google Storage URIs (e.g. 'gs://bucket', 'gs://bucket/path') are supported. Received {upload_path}."
        )
    return upload_path


def _validate_env_vars(env_vars: Dict[str, str]) -> Dict[str, str]:
    for key, value in env_vars.items():
        if not isinstance(key, str):
            raise click.ClickException(
                f"env_vars key {key} is not a string. Please check the formatting."
            )
        if not isinstance(value, str):
            raise click.ClickException(
                f"env_vars value {value} is not a string. Please check the formatting."
            )

    return env_vars


def _validate_and_modify_runtime_env(runtime_env: Dict):
    """Validates, then modifies runtime_env in place.

    Raises: click.ClickException if the runtime env doesn't pass validation.
    """

    if "conda" in runtime_env:
        conda_option = runtime_env["conda"]
        if not isinstance(conda_option, (str, dict)):
            raise click.ClickException(
                f"runtime_env['conda'] must be str or dict, got type({conda_option})."
            )
        runtime_env["conda"] = _validate_conda_option(conda_option)
    if "pip" in runtime_env:
        pip_option = runtime_env["pip"]
        if not isinstance(pip_option, (str, list)):
            raise click.ClickException(
                f"runtime_env['pip'] must be str or list, got type({pip_option})."
            )
        runtime_env["pip"] = _validate_pip_option(runtime_env["pip"])
    if "py_modules" in runtime_env:
        py_modules_option = runtime_env["py_modules"]
        if not isinstance(py_modules_option, list):
            raise click.ClickException(
                f"runtime_env['py_modules'] must be list, got type({py_modules_option})."
            )
        runtime_env["py_modules"] = _validate_py_modules(py_modules_option)
    if "upload_path" in runtime_env:
        upload_path_option = runtime_env["upload_path"]
        if not isinstance(upload_path_option, str):
            raise click.ClickException(
                f"runtime_env['upload_path'] must be str, got type({upload_path_option})."
            )
        runtime_env["upload_path"] = _validate_upload_path(upload_path_option)
    if "working_dir" in runtime_env:
        working_dir_option = runtime_env["working_dir"]
        if not isinstance(working_dir_option, str):
            raise click.ClickException(
                f"runtime_env['working_dir'] must be str, got type({working_dir_option})."
            )
        runtime_env["working_dir"] = _validate_working_dir(working_dir_option)
    _validate_working_dir_and_upload_path(
        runtime_env.get("working_dir"), runtime_env.get("upload_path")
    )
    if "env_vars" in runtime_env:
        env_vars_option = runtime_env["env_vars"]
        if not isinstance(env_vars_option, dict):
            raise click.ClickException(
                f"runtime_env['env_vars'] must be dict, got type({env_vars_option})."
            )
        runtime_env["env_vars"] = _validate_env_vars(env_vars_option)


def _get_project_id_from_id_or_name(
    project_id: Optional[str] = None, project_name: Optional[str] = None
) -> Optional[str]:
    """
    Get project id from PROJECT_NAME_ENV_VAR or `project_id` or `project_name`
    fields in the job config. Return None if no project was specified by the user,
    and the default project will selected later in the job submit code.
    """
    if project_id and project_name:
        raise click.ClickException(
            "Only one of `project_id` or `project` can be provided in the config file. "
        )
    project_name_env_var = os.environ.get(PROJECT_NAME_ENV_VAR)
    if project_name_env_var:
        # Get project from environment variable regardless of if is provided in config
        project_id = get_proj_id_from_name(project_name_env_var)
    elif project_name:
        project_id = get_proj_id_from_name(project_name)
    return project_id


class BaseHAJobConfig(BaseModel):
    """
    Base job configuration for both Production Jobs and Services.
    """

    name: Optional[str] = Field(
        None,
        description="Name of job to be submitted. Default will be used if not provided.",
    )
    description: Optional[str] = Field(
        None,
        description="Description of job to be submitted. Default will be used if not provided.",
    )
    runtime_env: Optional[Dict[str, Any]] = Field(
        None,
        description="A ray runtime env json. Your entrypoint will be run in the environment specified by this runtime env.",
    )
    build_id: Optional[str] = Field(
        None,
        description="The id of the cluster env build. This id will determine the docker image your job is run on.",
    )
    cluster_env: Optional[str] = Field(
        None,
        description="The name of the cluster environment and build revision in format `my_cluster_env:1`.",
    )
    docker: Optional[str] = Field(None, description="Custom docker image name.")
    python_version: Optional[str] = Field(
        None, description="Python version for the custom docker image."
    )
    ray_version: Optional[str] = Field(
        None, description="Ray version for the custom docker image."
    )
    compute_config_id: Optional[str] = Field(
        None,
        description="The id of the compute configuration that you want to use. This id will specify the resources required for your job",
    )
    project_id: Optional[str] = Field(
        None,
        description="The id of the project you want to use. If not specified, and no project is inferred from the directory, no project will be used.",
    )
    workspace_id: Optional[str] = Field(
        None, description="The id of the workspace that this job is submitted from.",
    )
    project: Optional[str] = Field(
        None,
        description="The name of the project you want to use. If not specified, and no project is inferred from the directory, no project will be used.",
    )
    compute_config: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="The name of the compute configuration and version that you want to use in format `my_compute_config:1`. This will specify the resources required for your job."
        "This field also accepts a one-off config as a dictionary.",
    )
    cloud: Optional[str] = Field(
        None,
        description="The cloud name to specify a default compute configuration with. This will specify the resources required for your job",
    )
    max_retries: Optional[int] = Field(
        5,
        description="The number of retries this job will attempt on failure. Set to None to set infinite retries",
    )

    job_queue_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration specifying semantic of the execution using job queues",
    )

    tags: Optional[Dict[str, str]] = Field(
        None, description="Key/value tags to associate with the job."
    )

    class Config:
        extra = Extra.forbid

    @root_validator
    def fill_project_id(cls: Any, values: Any) -> Any:
        project_id, project_name = (
            values.get("project_id"),
            values.get("project"),
        )
        values["project_id"] = _get_project_id_from_id_or_name(project_id, project_name)
        return values

    @root_validator
    def fill_cluster_env_from_custom_docker(cls: Any, values: Any) -> Any:
        docker, python_version, ray_version, cluster_env, build_id = (
            values.get("docker"),
            values.get("python_version"),
            values.get("ray_version"),
            values.get("cluster_env"),
            values.get("build_id"),
        )
        if docker is not None:
            if cluster_env is not None:
                raise click.ClickException(
                    "`cluster_env` and `docker` cannot both be specified. Please only provide one"
                    "of these in the job config file."
                )
            if build_id is not None:
                raise click.ClickException(
                    "`build_id` and `docker` cannot both be specified. Please only provide one"
                    "of these in the job config file."
                )
            if python_version is None and ray_version is None:
                python_version, ray_version = extract_versions_from_image_name(docker)

            if python_version is None:
                raise click.ClickException(
                    "`python_version` should be specified when `docker` is used."
                )
            if ray_version is None:
                raise click.ClickException(
                    "`ray_version` should be specified when `docker` is used."
                )
            values["cluster_env"] = BYODInfo(
                docker_image_name=docker,
                python_version=python_version,
                ray_version=ray_version,
            ).encode()
        return values

    @root_validator
    def fill_build_id(cls: Any, values: Any) -> Any:
        build_id, cluster_env = (
            values.get("build_id"),
            values.get("cluster_env"),
        )
        if cluster_env and build_id:
            raise click.ClickException(
                "Only one of `cluster_env` or `build_id` can be provided in the config file. "
            )
        if cluster_env:
            build_id = get_build_from_cluster_env_identifier(cluster_env).id
            values["build_id"] = build_id
        elif not build_id:
            log.info(
                "No cluster environment provided, setting default based on local Python and Ray versions."
            )
            build_id = get_default_cluster_env_build().id
            values["build_id"] = build_id
        validate_successful_build(values["build_id"])
        return values

    @root_validator
    def fill_compute_config_id(cls: Any, values: Any) -> Any:
        compute_config_id, compute_config, cloud = (
            values.get("compute_config_id"),
            values.get("compute_config"),
            values.get("cloud"),
        )
        project_id = values.get("project_id")
        if not project_id:
            project_id = _get_project_id_from_id_or_name(
                values.get("project_id"), values.get("project")
            )
        # Validation: allow (`compute_config` + optional `cloud`) OR `compute_config_id` OR `cloud` alone.
        # Disallow providing `compute_config_id` together with either `compute_config` or `cloud`.
        if compute_config_id and (compute_config or cloud):
            raise click.ClickException(
                "Provide either `compute_config_id` or (`compute_config` with optional `cloud`), not both."
            )
        if compute_config and isinstance(compute_config, str):
            # If a cloud name is provided alongside the compute config name, use it to disambiguate.
            if cloud:
                compute_config_id = get_cluster_compute_from_name(
                    compute_config, cloud_name=cloud
                ).id
            else:
                compute_config_id = get_cluster_compute_from_name(compute_config).id
        elif compute_config and isinstance(compute_config, dict):
            compute_config_id = register_compute_template(compute_config).id
        elif cloud:
            # Get default cluster compute for the specified cloud.
            compute_config_id = get_default_cluster_compute(
                cloud_name=cloud, project_id=None
            ).id
            log.info(
                f"Using default compute config for specified cloud {cloud}: {compute_config_id}."
            )
        elif not compute_config_id:
            parent_cloud_name = None
            if project_id:
                parent_cloud_id_and_name = get_parent_cloud_id_and_name_of_project(
                    project_id
                )
                if parent_cloud_id_and_name:
                    _, parent_cloud_name = parent_cloud_id_and_name
            # Get default cluster compute for the parent cloud if it exists or the default cloud default cloud.
            compute_config_id = get_default_cluster_compute(
                cloud_name=parent_cloud_name, project_id=None
            ).id
            msg_about_cloud = (
                f" for cloud {parent_cloud_name}" if parent_cloud_name else ""
            )
            log.info(
                f"No cloud or compute config specified, using the default{msg_about_cloud}: {compute_config_id}."
            )
        values["compute_config_id"] = compute_config_id

        return values

    @root_validator
    def validate_runtime_env(cls: Any, values: Any) -> Any:  # noqa: PLR0912
        runtime_env = values.get("runtime_env")
        if runtime_env is not None:
            _validate_and_modify_runtime_env(runtime_env)
        return values


class JobConfig(BaseHAJobConfig):
    """
    Job Config model for CLI. Validate and preprocess so `entrypoint`, `runtime_env_config`,
    `build_id`, `compute_config_id`, `max_retries` have the correct values to call
    `/api/v2/decorated_ha_jobs/create`.
    """

    entrypoint: str = Field(
        ...,
        description="A script that will be run to start your job. This command will be run in the root directory of the specified runtime env. Eg. 'python script.py'",
    )

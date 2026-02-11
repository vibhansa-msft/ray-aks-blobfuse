import os
from typing import Any, Dict, Optional, Tuple, Union

import click
from click import ClickException
import yaml

from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client import Project
from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.cloud_utils import get_cloud_id_and_name
from anyscale.cluster_compute import (
    get_cluster_compute_from_name,
    get_selected_cloud_id_or_default,
)
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as SDKDefaultApi
from anyscale.sdk.anyscale_client.models.cluster_compute_config import (
    ClusterComputeConfig,
)
from anyscale.shared_anyscale_utils.util import slugify
from anyscale.util import get_endpoint, PROJECT_NAME_ENV_VAR, send_json_request


ANYSCALE_PROJECT_FILE = ".anyscale.yaml"
log = BlockLogger()


def find_project_root(directory: str) -> Optional[str]:
    """Find root directory of the project.
    Args:
        directory (str): Directory to start the search in.
    Returns:
        Path of the parent directory containing the project
        or None if no such project is found.
    """
    prev, directory = None, os.path.abspath(directory)
    while prev != directory:
        if os.path.exists(os.path.join(directory, ANYSCALE_PROJECT_FILE)):
            return directory
        prev, directory = directory, os.path.abspath(os.path.join(directory, os.pardir))
    return None


class ProjectDefinition:
    def __init__(self, root_dir: str):
        self.root = os.path.join(root_dir, "")
        anyscale_yaml = os.path.join(root_dir, ANYSCALE_PROJECT_FILE)
        if os.path.exists(anyscale_yaml):
            with open(anyscale_yaml) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}


def load_project_or_throw() -> ProjectDefinition:
    # First check if there is a .anyscale.yaml.
    root_dir = find_project_root(os.getcwd())
    if not root_dir:
        raise ClickException("No project directory found")
    return ProjectDefinition(root_dir)


def get_project_id(project_dir: str) -> str:
    """
    Args:
        project_dir: Project root directory.
    Returns:
        The ID of the associated Project in the database.
    Raises:
        ValueError: If the current project directory does
            not contain a project ID.
    """
    project_filename = os.path.join(project_dir, ANYSCALE_PROJECT_FILE)
    if os.path.isfile(project_filename):
        with open(project_filename) as f:
            config = yaml.safe_load(f)
            project_id = config.get("project_id")
    else:
        # TODO(pcm): Consider doing this for the user and retrying the command
        # they were trying to run.
        raise ClickException(
            "Ray project in {} not registered yet. "
            "Did you run 'anyscale project init'?".format(project_dir)
        )
    try:
        result = str(project_id)
    except ValueError:
        # TODO(pcm): Tell the user what to do here.
        raise ClickException(f"{project_filename} does not contain a valid project ID")
    return result


def validate_project_name(project_name: str) -> bool:
    return " " not in project_name.strip()


def get_project_sessions(
    project_id: str,
    session_name: Optional[str],
    api_client: DefaultApi = None,
    all_active_states: bool = False,
) -> Any:
    """
    Returns active project clusters. If `all_active_states` is set, returns clusters with
        the following states: StartingUp, _StartingUp, StartupErrored, Running, Updating,
        UpdatingErrored, AwaitingFileMounts. Otherwise, only return clusters in the
        Running and AwaitingFileMounts state.
    """
    if api_client is None:
        return _get_project_sessions(project_id, session_name)

    if all_active_states:
        response = api_client.list_sessions_api_v2_sessions_get(
            project_id=project_id,
            name=session_name,
            active_only=True,
            _request_timeout=30,
        )
    else:
        response = api_client.list_sessions_api_v2_sessions_get(
            project_id=project_id,
            name=session_name,
            state_filter=["AwaitingFileMounts", "Running"],
            _request_timeout=30,
        )
    sessions = response.results
    if len(sessions) == 0:
        raise ClickException(f"No active cluster matching pattern {session_name} found")
    return sessions


# TODO (jbai): DEPRECATED - will be removed when OpenApi migration is completed
def _get_project_sessions(project_id: str, session_name: Optional[str]) -> Any:
    response = send_json_request(
        "/api/v2/sessions/",
        {"project_id": project_id, "name_match": session_name, "active_only": True},
    )
    sessions = response["results"]
    if len(sessions) == 0:
        raise ClickException(f"No active cluster matching pattern {session_name} found")
    return sessions


def get_project_session(
    project_id: str,
    session_name: Optional[str],
    api_client: DefaultApi = None,
    is_workspace=False,
) -> Any:
    if api_client is None:
        return _get_project_session(project_id, session_name)

    sessions = get_project_sessions(project_id, session_name, api_client)
    if is_workspace:
        # TODO(https://github.com/anyscale/product/issues/11585): We need a more robust way to find workspaces.
        sessions = [
            session for session in sessions if session.name.startswith("workspace-")
        ]

    if len(sessions) > 1:
        raise ClickException(
            "Multiple active clusters: {}\n"
            "Please specify the one you want to refer to.".format(
                [session.name for session in sessions]
            )
        )
    return sessions[0]


# TODO (jbai): DEPRECATED - will be removed when OpenApi migration is completed
def _get_project_session(project_id: str, session_name: Optional[str]) -> Any:
    sessions = get_project_sessions(project_id, session_name)
    if len(sessions) > 1:
        raise ClickException(
            "Multiple active clusters: {}\n"
            "Please specify the one you want to refer to.".format(
                [session["name"] for session in sessions]
            )
        )
    return sessions[0]


def get_proj_name_from_id(project_id: str, api_client: DefaultApi) -> str:
    resp = api_client.get_project_api_v2_projects_project_id_get(
        project_id=project_id, _request_timeout=30
    )

    if resp is None:
        raise ClickException(
            "This local project is not registered with anyscale. Please re-run `anyscale project init`."
        )
    else:
        return str(resp.result.name)


def get_proj_id_from_name(
    project_name: str,
    api_client: Optional[DefaultApi] = None,
    owner: Optional[str] = None,  # this can be the email or the username of the owner
) -> str:
    if api_client is None:
        api_client = get_auth_api_client().api_client

    resp = api_client.find_project_by_project_name_api_v2_projects_find_by_name_get(
        name=project_name, _request_timeout=30, owner=owner
    )

    if not resp.results:
        raise ClickException(
            f"There is no project '{project_name}' that is registered with Anyscale. "
            "View the registered projects with `anyscale project list`."
        )

    projects = resp.results
    my_projects = [x for x in projects if x.is_owner]

    selected_project = None

    # If there is more than one result, choose the one that you own
    # If there is one project, select it
    if len(projects) == 1:
        selected_project = projects[0]

    # Only one of the projects is mine. Let's select this one
    elif len(my_projects) == 1:
        selected_project = my_projects[0]

    # We know there is at least one element. If none of the projects are mine
    # then we don't know which one to select
    else:
        raise ClickException(
            f"There are multiple projects '{project_name}' registered with Anyscale. "
            "View the registered projects with `anyscale project list`."
            "Please specify the --owner flag to specify an alternate owner"
        )

    # Return the id of this project
    return str(selected_project.id)


def get_project_id_for_cloud_from_name(
    project_name: str,
    parent_cloud_id: str,
    api_client: Optional[DefaultApi] = None,
    anyscale_api_client: Optional[DefaultApi] = None,
) -> str:
    if api_client is None:
        api_client = get_auth_api_client().api_client

    if project_name == "default":  # get default project
        return get_default_project(api_client, anyscale_api_client, parent_cloud_id).id

    existing_projects = api_client.find_project_by_project_name_api_v2_projects_find_by_name_get(
        project_name
    ).results
    if len(existing_projects) == 0:
        raise ClickException(f"Project '{project_name}' was not found.")
    else:
        for project in existing_projects:
            if project.parent_cloud_id == parent_cloud_id:
                return project.id
        raise ClickException(
            f"Project '{project_name}' under cloud '{parent_cloud_id}' was not found."
        )


def get_parent_cloud_id_and_name_of_project(
    project_id: str, api_client: Optional[DefaultApi] = None,
) -> Optional[Tuple[str, str]]:
    """
    Returns tuple (parent_cloud_id, parent_cloud_name) of project if cloud isolation is enabled.
    Return (None, None) if cloud isolation isn't enabled or a project with this id doesn't exist.
    """
    if api_client is None:
        api_client = get_auth_api_client().api_client

    project = api_client.get_project_api_v2_projects_project_id_get(
        project_id=project_id
    ).result
    if project:
        parent_cloud_id, parent_cloud_name = get_cloud_id_and_name(
            api_client, cloud_id=project.parent_cloud_id
        )
        return (parent_cloud_id, parent_cloud_name)
    return None


def write_project_file_to_disk(project_id: str, directory: str) -> None:
    with open(os.path.join(directory, ANYSCALE_PROJECT_FILE), "w") as f:
        f.write("{}".format(f"project_id: {project_id}"))


def create_new_proj_def(
    name: str, api_client: DefaultApi = None,  # noqa: ARG001
) -> Tuple[str, ProjectDefinition]:
    if slugify(name) != name:
        name = slugify(name)
        log.info(f"Normalized project name to {name}")

    project_definition = ProjectDefinition(os.getcwd())
    project_definition.config["name"] = name
    return name, project_definition


def _do_attach(project_id: str, is_create_project: bool) -> None:
    with open(ANYSCALE_PROJECT_FILE, "w") as f:
        yaml.dump(
            {"project_id": project_id}, f,
        )

    # Print success message
    url = get_endpoint(f"/projects/{project_id}")
    if is_create_project:
        log.info(f"Project {project_id} created. View at {url}")
    else:
        log.info(f"Attached to project {project_id}. View at {url}")


def register_or_attach_to_project(
    project_definition: ProjectDefinition, api_client: DefaultApi
) -> None:

    project_name = project_definition.config["name"]
    description = project_definition.config.get("description", "")

    # Find if project with name already exists
    existing_projects = api_client.find_project_by_project_name_api_v2_projects_find_by_name_get(
        project_name
    )
    existing_project = None
    if existing_projects:
        if len(existing_projects.results) == 1:
            existing_project = existing_projects.results[0]
        elif len(existing_projects.results) > 1:
            try:
                user_info = api_client.get_user_info_api_v2_userinfo_get()
            except Exception:  # noqa: BLE001
                raise ClickException(
                    f"More than one project with name {project_name} was found. Please connect to this project using --project-id parameter"
                )

            for project in existing_projects.results:
                if project.creator_id == user_info.result.id:
                    existing_project = project

            if not existing_project:
                raise ClickException(
                    f"More than one project with name {project_name} was found. Please connect to this project using --project-id parameter"
                )
            else:
                log.info(
                    f"Multiple projects with name {project_name} was found. Connecting to the one created by you."
                )

    if not existing_project:
        # Add a database entry for the new Project.
        resp = api_client.create_project_api_v2_projects_post(
            write_project={
                "name": project_name,
                "description": description,
                "initial_cluster_config": None,
            }
        )
        result = resp.result
        project_id = result.id
    else:
        project_id = existing_project.id

    _do_attach(project_id, not existing_project)


def attach_to_project_with_id(project_id: str, api_client: DefaultApi) -> None:
    validate_project_id(project_id, api_client)

    _do_attach(project_id, False)


def validate_project_id(project_id, api_client: DefaultApi) -> Project:
    # Find if project with name already exists
    try:
        resp = api_client.get_project_api_v2_projects_project_id_get(project_id)
    except Exception:  # noqa: BLE001
        resp = None

    if not resp:
        raise click.ClickException(
            f"Project with id {project_id} does not exist. Instead, you can provide a name to anyscale project init to create a new project."
        )

    return resp.result


def get_and_validate_project_id(
    project_id: Optional[str],
    project_name: Optional[str],
    parent_cloud_id: Optional[str],
    api_client: DefaultApi,
    anyscale_api_client: Union[DefaultApi, SDKDefaultApi],
) -> str:
    project_name = project_name or os.environ.get(PROJECT_NAME_ENV_VAR)
    if project_id:
        validate_project_id(project_id, api_client)
    elif project_name:
        project_id = get_proj_id_from_name(project_name, api_client)
    else:
        try:
            # .anyscale.yaml detected in root directory and has correct format
            project_definition = load_project_or_throw()
            project_id = get_project_id(project_definition.root)
        except click.ClickException:
            default_project = get_default_project(
                api_client, anyscale_api_client, parent_cloud_id=parent_cloud_id
            )
            project_id = default_project.id
            log.info(
                "No project context detected or `--project-id` provided. Continuing without a project."
            )
    assert project_id  # for mypy
    return project_id


def get_default_project(
    api_client: Optional[DefaultApi] = None,
    anyscale_api_client: Optional[Union[DefaultApi, SDKDefaultApi]] = None,
    parent_cloud_id: Optional[str] = None,
):
    """
    Gets the default project to use if no project has been selected.

    parent_cloud_id: cloud_id to get a default project for. This field is
        required if cloud isolation is enabled, because there are multiple
        default projects in an organization, each associated with a
        different cloud.
    """
    anyscale_api_client = (
        anyscale_api_client or get_auth_api_client(log_output=False).anyscale_api_client
    )
    api_client = api_client or get_auth_api_client(log_output=False).api_client
    if not parent_cloud_id:
        raise click.ClickException(
            "Please specify a cloud for this command either through the cloud or "
            "compute config arguments."
        )
    default_project = anyscale_api_client.get_default_project(  # type: ignore
        parent_cloud_id=parent_cloud_id
    ).result
    return default_project


def infer_project_id(  # noqa PLR0913
    anyscale_api_client: Union[DefaultApi, SDKDefaultApi],
    api_client: DefaultApi,
    log: BlockLogger,
    project_id: Optional[str],
    cluster_compute_id: Optional[str] = None,
    cluster_compute: Optional[Union[str, Dict[str, Any]]] = None,
    cloud: Optional[str] = None,
) -> str:
    if not project_id:
        # Check directory of .anyscale.yaml to decide whether to use default project.
        root_dir = find_project_root(os.getcwd())
        if root_dir is not None:
            project_definition = ProjectDefinition(root_dir)
            project_id = get_project_id(project_definition.root)
        else:
            # TODO: add cloud compute config args here
            if not cluster_compute_id and isinstance(cluster_compute, str):
                cluster_compute_id = get_cluster_compute_from_name(
                    cluster_compute, api_client
                ).id
            cluster_compute_config = (
                ClusterComputeConfig(**cluster_compute)
                if isinstance(cluster_compute, dict)
                else None
            )
            parent_cloud_id = get_selected_cloud_id_or_default(
                api_client,
                anyscale_api_client,
                cluster_compute_id=cluster_compute_id,
                cluster_compute_config=cluster_compute_config,
                cloud_id=None,
                cloud_name=cloud,
            )
            default_project = get_default_project(
                api_client, anyscale_api_client, parent_cloud_id=parent_cloud_id
            )
            project_id = default_project.id
            log.info("No project specified. Continuing without a project.")
    return str(project_id)

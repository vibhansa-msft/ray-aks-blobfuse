import os
from typing import Any, Dict, Optional, Tuple

import click
import yaml

import anyscale
from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.cloud_utils import (
    get_cloud_id_and_name,
    get_last_used_cloud,
    get_organization_default_cloud,
)
from anyscale.cluster_compute import get_cluster_compute_from_name
import anyscale.conf
import anyscale.project_utils
from anyscale.sdk.anyscale_client import ComputeTemplateConfig
from anyscale.utils.connect_helpers import find_project_id


# The cluster compute type. It can either be a string, eg my_template or a dict,
# eg, {"cloud_id": "id-123" ...}
CLUSTER_COMPUTE_DICT_TYPE = Dict[str, Any]


class ProjectBlock:
    """
    Class to determine which project ray.client uses. This class should never be
    instantiated directly. Instead call `get_project_block` to ensure a new
    ProjectBlock object is correctly created.
    """

    def __init__(  # noqa PLR0913
        self,
        project_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        cloud_name: Optional[str] = None,
        cluster_compute_name: Optional[str] = None,
        cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE] = None,
        log_output: bool = True,
    ):
        """
        If project_dir is passed or exists in the directory structure: ensure the project_dir
            is correctly set up. Otherwise use the default project.
        Create an ProjectBlock object with the following properties that are useful
        externally: project_id, project_name, project_dir.

        Arguments:
            project_dir (Optional[str]): Project directory location. This doesn't need to contain
                a .anyscale.yaml file.
            project_name (Optional[str]): Project name if it is necessary to register the project
                with Anyscale. This will not be used if a registered project already exists
                in the project_dir.
            cloud_name (Optional[str]): This argument is only relevant if cloud isolation is enabled.
                If cloud_name is specified and no project arguements are specified, take the default
                project of the cloud.
        """
        self.log = BlockLogger(log_output=log_output)
        auth_api_client = get_auth_api_client(log_output=log_output)
        self.api_client = auth_api_client.api_client
        self.anyscale_api_client = auth_api_client.anyscale_api_client
        self.project_dir = (
            os.path.abspath(os.path.expanduser(project_dir)) if project_dir else None
        )
        self.cloud_name = cloud_name
        self.cluster_compute_name = cluster_compute_name
        self.cluster_compute_dict = cluster_compute_dict

        self.block_label = "Project"
        self.log.open_block(self.block_label, block_title="Choosing a project")

        if self.project_dir is None and project_name is None:
            self.project_dir = anyscale.project_utils.find_project_root(os.getcwd())

        parent_cloud_id = self._get_parent_cloud_id(
            self.cloud_name, self.cluster_compute_name, self.cluster_compute_dict,
        )

        left_pad = " " * 2
        if self.project_dir:
            # TODO(nikita): Remove this logic when removing .anyscale.yaml in Q3 2022
            self.project_id, self.project_name = self._ensure_project_setup_at_dir(
                self.project_dir, project_name, parent_cloud_id
            )
            self.log.info(
                f"{left_pad}{'name:': <20}{self.project_name}",
                block_label=self.block_label,
            )
            self.log.info(
                f"{left_pad}{'project id:': <20}{self.project_id}",
                block_label=self.block_label,
            )
            self.log.info(
                f"{left_pad}{'project directory:': <20}{self.project_dir}",
                block_label=self.block_label,
            )
        elif project_name:
            self.project_id, self.project_name = self._create_or_get_project_from_name(
                project_name, parent_cloud_id
            )
            self.log.info(
                f"{left_pad}{'name:': <20}{self.project_name}",
                block_label=self.block_label,
            )
            self.log.info(
                f"{left_pad}{'project id:': <20}{self.project_id}",
                block_label=self.block_label,
            )
        else:
            self.project_id, self.project_name = self._get_default_project(
                parent_cloud_id
            )
        self.log.close_block(self.block_label)

    def _get_default_project(self, parent_cloud_id: str) -> Tuple[str, str]:
        """
        Get default project id and name.

        Returns:
        The project id and project name of the project being used.
        """
        default_project = self.anyscale_api_client.get_default_project(
            parent_cloud_id=parent_cloud_id
        ).result
        project_id = default_project.id
        project_name = default_project.name
        self.log.info(
            "No project defined. Continuing without a project.",
            block_label=self.block_label,
        )
        return project_id, project_name

    def _get_parent_cloud_id(
        self,
        cloud_name: Optional[str],
        cluster_compute_name: Optional[str],
        cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE],
    ) -> str:
        parent_cloud_id: Optional[str] = None
        if cloud_name:
            parent_cloud_id, _ = get_cloud_id_and_name(
                api_client=self.api_client, cloud_id=None, cloud_name=cloud_name,
            )
        elif cluster_compute_name:
            cluster_compute = get_cluster_compute_from_name(
                cluster_compute_name, self.api_client
            )
            parent_cloud_id = cluster_compute.config.cloud_id
        elif cluster_compute_dict:
            cluster_compute_config_obj = ComputeTemplateConfig(**cluster_compute_dict)
            parent_cloud_id = cluster_compute_config_obj.cloud_id
        else:
            # Get organization default cloud or last used cloud
            default_cloud_name = get_organization_default_cloud(self.api_client)
            cloud_name = default_cloud_name or get_last_used_cloud(
                None, self.anyscale_api_client
            )
            parent_cloud_id, _ = get_cloud_id_and_name(
                self.api_client, cloud_id=None, cloud_name=cloud_name
            )

        if parent_cloud_id is None:
            raise click.ClickException(
                "Unable to determine the cloud for this operation. "
                "Please specify a cloud explicitly or provide a cluster_compute config."
            )
        return parent_cloud_id

    def _ensure_project_setup_at_dir(
        self, project_dir: str, project_name: Optional[str], parent_cloud_id: str,
    ) -> Tuple[str, str]:
        """
        Get or create an Anyscale project rooted at the given dir. If .anyscale.yaml
        exists in the given project dir and that project is registered with Anyscale,
        return information about that project. Otherwise, create a project with this
        project_dir and project_name (use default if not provided).

        Returns:
        The project id and project name of the project being used.
        """
        os.makedirs(project_dir, exist_ok=True)
        if project_name is None:
            project_name = os.path.basename(project_dir)

        # If the project yaml exists, assume we're already setup.
        project_yaml = os.path.join(project_dir, ".anyscale.yaml")
        if os.path.exists(project_yaml):
            # Validate format of project yaml and get project id
            proj_def = anyscale.project_utils.ProjectDefinition(project_dir)
            project_id: Optional[str] = anyscale.project_utils.get_project_id(
                proj_def.root
            )
            if not project_id:
                raise click.ClickException(
                    f"{project_yaml} is not correctly formatted. Please attach to a different "
                    "project using `anyscale project init`."
                )
            try:
                project_response = self.anyscale_api_client.get_project(project_id)
            except click.ClickException:
                raise click.ClickException(
                    f"Unable to get project with id {project_id} from Anyscale. Please attach to "
                    "a different project using `anyscale project init`."
                )
            self.log.info(
                f"Using the project defined in {project_yaml}:",
                block_label=self.block_label,
            )
            return project_id, project_response.result.name

        project_id = find_project_id(
            self.anyscale_api_client, project_name, parent_cloud_id
        )
        if project_id is None:
            # Create a new project in the local directory with given name, because
            # project with this name doesn't exist yet.
            self.log.info(
                f"Creating new project named {BlockLogger.highlight(project_name)} for local dir {project_dir}.",
                block_label=self.block_label,
            )
            self.log.info(
                f"Using the project {BlockLogger.highlight(project_name)}:",
                block_label=self.block_label,
            )
            project_response = self.anyscale_api_client.create_project(
                {
                    "name": project_name,
                    "description": "Automatically created by Anyscale Connect",
                    "parent_cloud_id": parent_cloud_id,
                }
            )
            project_id = project_response.result.id
        else:
            # Project already exists with this name, yet directory doesn't contain
            # project yaml.
            self.log.info(
                f"Connecting local directory {project_dir} to project {BlockLogger.highlight(project_name)}.",
                block_label=self.block_label,
            )
            self.log.info(
                f"Using the project {BlockLogger.highlight(project_name)}:",
                block_label=self.block_label,
            )

        if not os.path.exists(project_yaml):
            with open(project_yaml, "w+") as f:
                f.write(yaml.dump({"project_id": project_id}))

        return project_id, project_name

    def _create_or_get_project_from_name(
        self, project_name: str, parent_cloud_id: str
    ) -> Tuple[str, str]:
        """
        Get or create an Anyscale project not rooted in any directory
        (without .anyscale.yaml).

        Returns:
        The project id and project name of the project being used.
        """
        project_id = find_project_id(
            self.anyscale_api_client, project_name, parent_cloud_id
        )
        if project_id is None:
            self.log.info(
                f"Creating new project named {BlockLogger.highlight(project_name)}.",
                block_label=self.block_label,
            )
            project_response = self.anyscale_api_client.create_project(
                {
                    "name": project_name,
                    "description": "Automatically created by Anyscale Connect",
                    "parent_cloud_id": parent_cloud_id,
                }
            )
            project_id = project_response.result.id
        self.log.info(
            f"Using the project {BlockLogger.highlight(project_name)}:",
            block_label=self.block_label,
        )
        return project_id, project_name


def create_project_block(
    project_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    cloud_name: Optional[str] = None,
    cluster_compute_name: Optional[str] = None,
    cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE] = None,
    log_output: bool = True,
):
    """
    Function to create new ProjectBlock object. The ProjectBlock object
    is not a global variable an will be reinstantiated on each call to
    get_project_block.
    """
    return ProjectBlock(
        project_dir=project_dir,
        project_name=project_name,
        cloud_name=cloud_name,
        cluster_compute_name=cluster_compute_name,
        cluster_compute_dict=cluster_compute_dict,
        log_output=log_output,
    )

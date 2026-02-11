from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from anyscale._private.models import ModelBase


@dataclass(frozen=True)
class CreateResourceQuota(ModelBase):
    """Resource quota creation model.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.resource_quota.models import CreateResourceQuota

create_resource_quota = CreateResourceQuota(
    # Name of the resource quota to create
    name="resource_quota_name",
    # Name of the cloud that this resource quota applies to
    cloud="cloud_name",
    # Name of the project that this resource quota applies to (optional)
    project="project_name",
    # Email of the user that this resource quota applies to (optional)
    user_email="test@anyscale.com",
    # The quota limit for the number of CPUs (optional)
    num_cpus=50,
    # The quota limit for the number of instances (optional)
    num_instances=100,
    # The quota limit for the total number of GPUs (optional)
    num_gpus=30,
    # The quota limit for the number of accelerators (optional)
    num_accelerators={"A100-80G": 10, "T4": 20},
)
"""

    name: str = field(metadata={"docstring": "Name of the resource quota to create."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    cloud: str = field(
        metadata={"docstring": "Name of the cloud that this resource quota applies to."}
    )

    def _validate_cloud(self, cloud: str):
        if not isinstance(cloud, str):
            raise TypeError("cloud must be a string.")

    project: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Name of the project that this resource quota applies to (optional)."
        },
    )

    def _validate_project(self, project: Optional[str]):
        if project is not None and not isinstance(project, str):
            raise TypeError("project must be a string.")

    user_email: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Email of the user that this resource quota applies to (optional)."
        },
    )

    def _validate_user_email(self, user_email: Optional[str]):
        if user_email is not None and not isinstance(user_email, str):
            raise TypeError("user_email must be a string.")

    num_cpus: Optional[int] = field(
        default=None,
        metadata={"docstring": "The quota limit for the number of CPUs (optional)."},
    )

    def _validate_num_cpus(self, num_cpus: Optional[int]):
        if num_cpus is not None and not isinstance(num_cpus, int):
            raise TypeError("num_cpus must be an integer.")

    num_instances: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the number of instances. (optional)."
        },
    )

    def _validate_num_instances(self, num_instances: Optional[int]):
        if num_instances is not None and not isinstance(num_instances, int):
            raise TypeError("num_instances must be an integer.")

    num_gpus: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the total number of GPUs (optional)."
        },
    )

    def _validate_num_gpus(self, num_gpus: Optional[int]):
        if num_gpus is not None and not isinstance(num_gpus, int):
            raise TypeError("num_gpus must be an integer.")

    num_accelerators: Optional[Dict[str, int]] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the number of accelerators (optional)."
        },
    )

    def _validate_num_accelerators(self, num_accelerators: Optional[Dict[str, int]]):
        if num_accelerators is not None and not isinstance(num_accelerators, dict):
            raise TypeError("num_accelerators must be a dictionary.")
        if num_accelerators is not None:
            for key, value in num_accelerators.items():
                if not isinstance(key, str):
                    raise TypeError("num_accelerators keys must be strings.")
                if not isinstance(value, int):
                    raise TypeError("num_accelerators values must be integers.")


@dataclass(frozen=True)
class Quota(ModelBase):
    """Resource quota limit
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.resource_quota.models import CreateResourceQuota, ResourceQuota, Quota

create_resource_quota = CreateResourceQuota(
    # Name of the resource quota to create
    name="resource_quota_name",
    # Name of the cloud that this resource quota applies to
    cloud="cloud_name",
    # Name of the project that this resource quota applies to (optional)
    project="project_name",
    # Email of the user that this resource quota applies to (optional)
    user_email="test@anyscale.com",
    # The quota limit for the number of CPUs (optional)
    num_cpus=50,
    # The quota limit for the number of instances (optional)
    num_instances=100,
    # The quota limit for the total number of GPUs (optional)
    num_gpus=30,
    # The quota limit for the number of accelerators (optional)
    num_accelerators={"A100-80G": 10, "T4": 20},
)

resource_quota: ResourceQuota = anyscale.resource_quota.create(create_resource_quota)

quota: Quota = resource_quota.quota
"""
    num_cpus: Optional[int] = field(
        default=None,
        metadata={"docstring": "The quota limit for the number of CPUs (optional)."},
    )

    def _validate_num_cpus(self, num_cpus: Optional[int]):
        if num_cpus is not None and not isinstance(num_cpus, int):
            raise TypeError("num_cpus must be an integer.")

    num_instances: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the number of instances. (optional)."
        },
    )

    def _validate_num_instances(self, num_instances: Optional[int]):
        if num_instances is not None and not isinstance(num_instances, int):
            raise TypeError("num_instances must be an integer.")

    num_gpus: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the total number of GPUs (optional)."
        },
    )

    def _validate_num_gpus(self, num_gpus: Optional[int]):
        if num_gpus is not None and not isinstance(num_gpus, int):
            raise TypeError("num_gpus must be an integer.")

    num_accelerators: Optional[Dict[str, int]] = field(
        default=None,
        metadata={
            "docstring": "The quota limit for the number of accelerators (optional)."
        },
    )

    def _validate_num_accelerators(self, num_accelerators: Optional[Dict[str, int]]):
        if num_accelerators is not None and not isinstance(num_accelerators, dict):
            raise TypeError("num_accelerators must be a dictionary.")
        if num_accelerators is not None:
            for key, value in num_accelerators.items():
                if not isinstance(key, str):
                    raise TypeError("num_accelerators keys must be strings.")
                if not isinstance(value, int):
                    raise TypeError("num_accelerators values must be integers.")


@dataclass(frozen=True)
class ResourceQuota(ModelBase):
    """Resource quota
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.resource_quota.models import CreateResourceQuota, ResourceQuota

create_resource_quota = CreateResourceQuota(
    # Name of the resource quota to create
    name="resource_quota_name",
    # Name of the cloud that this resource quota applies to
    cloud="cloud_name",
    # Name of the project that this resource quota applies to (optional)
    project="project_name",
    # Email of the user that this resource quota applies to (optional)
    user_email="test@anyscale.com",
    # The quota limit for the number of CPUs (optional)
    num_cpus=50,
    # The quota limit for the number of instances (optional)
    num_instances=100,
    # The quota limit for the total number of GPUs (optional)
    num_gpus=30,
    # The quota limit for the number of accelerators (optional)
    num_accelerators={"A100-80G": 10, "T4": 20},
)
resource_quota: ResourceQuota = anyscale.resource_quota.create(create_resource_quota)
"""
    id: str = field(metadata={"docstring": "The ID of the resource quota."})

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("id must be a string.")

    name: str = field(metadata={"docstring": "Name of the resource quota."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    quota: Quota = field(metadata={"docstring": "The quota limit."})

    def _validate_quota(self, quota: Quota):
        if not isinstance(quota, Quota):
            raise TypeError("quota must be a Quota.")

    created_at: datetime = field(
        metadata={"docstring": "The timestamp when this resource quota was created."}
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    cloud_id: str = field(
        metadata={"docstring": "ID of the cloud that this resource quota applies to."}
    )

    def _validate_cloud_id(self, cloud_id: str):
        if not isinstance(cloud_id, str):
            raise TypeError("cloud_id must be a string.")

    project_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "ID of the project that this resource quota applies to (optional)."
        },
    )

    def _validate_project_id(self, project_id: Optional[str]):
        if project_id is not None and not isinstance(project_id, str):
            raise TypeError("project_id must be a string.")

    user_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "ID of the user that this resource quota applies to (optional)."
        },
    )

    def _validate_user_id(self, user_id: Optional[str]):
        if user_id is not None and not isinstance(user_id, str):
            raise TypeError("user_id must be a string.")

    is_enabled: bool = field(
        default=True, metadata={"docstring": "Whether the resource quota is enabled."}
    )

    def _validate_is_enabled(self, is_enabled: bool):
        if not isinstance(is_enabled, bool):
            raise TypeError("is_enabled must be a boolean.")

    deleted_at: Optional[datetime] = field(
        default=None,
        metadata={"docstring": "The timestamp when this resource quota was deleted."},
    )

    def _validate_deleted_at(self, deleted_at: Optional[datetime]):
        if deleted_at is not None and not isinstance(deleted_at, datetime):
            raise TypeError("deleted_at must be a datetime.")

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Union

from anyscale._private.models import ModelBase, ModelEnum


class CloudPermissionLevel(ModelEnum):
    """Permission levels for cloud collaborators."""

    WRITE = "WRITE"
    READONLY = "READONLY"

    __docstrings__: ClassVar[Dict[str, str]] = {
        WRITE: "Write permission level for the cloud",
        READONLY: "Readonly permission level for the cloud",
    }


@dataclass(frozen=True)
class CreateCloudCollaborator(ModelBase):
    """User to be added as a collaborator to a cloud.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.cloud.models import CloudPermissionLevel, CreateCloudCollaborator

create_cloud_collaborator = CreateCloudCollaborator(
   # Email of the user to be added as a collaborator
    email="test@anyscale.com",
    # Permission level for the user to the cloud (CloudPermissionLevel.WRITE, CloudPermissionLevel.READONLY)
    permission_level=CloudPermissionLevel.READONLY,
)
"""

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("Email must be a string.")

    email: str = field(
        metadata={"docstring": "Email of the user to be added as a collaborator."},
    )

    def _validate_permission_level(
        self, permission_level: CloudPermissionLevel
    ) -> CloudPermissionLevel:
        if isinstance(permission_level, str):
            return CloudPermissionLevel.validate(permission_level)  # type: ignore
        elif isinstance(permission_level, CloudPermissionLevel):
            return permission_level
        else:
            raise TypeError(
                f"'permission_level' must be a 'CloudPermissionLevel' (it is {type(permission_level)})."
            )

    permission_level: CloudPermissionLevel = field(  # type: ignore
        default=CloudPermissionLevel.READONLY,  # type: ignore
        metadata={
            "docstring": "Permission level the added user should have for the cloud"  # type: ignore
            f"(one of: {','.join([str(m.value) for m in CloudPermissionLevel])}",  # type: ignore
        },
    )


@dataclass(frozen=True)
class CreateCloudCollaborators(ModelBase):
    """List of users to be added as collaborators to a cloud.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.cloud.models import CloudPermissionLevel, CreateCloudCollaborator, CreateCloudCollaborators

create_cloud_collaborator = CreateCloudCollaborator(
   # Email of the user to be added as a collaborator
    email="test@anyscale.com",
    # Permission level for the user to the cloud (CloudPermissionLevel.WRITE, CloudPermissionLevel.READONLY)
    permission_level=CloudPermissionLevel.READONLY,
)
create_cloud_collaborators = CreateCloudCollaborators(
    collaborators=[create_cloud_collaborator]
)
"""

    collaborators: List[Dict[str, Any]] = field(
        metadata={
            "docstring": "List of users to be added as collaborators to a cloud."
        },
    )

    def _validate_collaborators(self, collaborators: List[Dict[str, Any]]):
        if not isinstance(collaborators, list):
            raise TypeError("Collaborators must be a list.")


class ComputeStack(ModelEnum):
    """Type of compute stack for the cloud."""

    UNKNOWN = "UNKNOWN"
    VM = "VM"
    K8S = "K8S"

    __docstrings__: ClassVar[Dict[str, str]] = {
        UNKNOWN: "Unknown compute stack.",
        VM: "Virtual machine-based compute stack.",
        K8S: "Kubernetes-based compute stack.",
    }


class CloudProvider(ModelEnum):
    """Cloud infrastructure provider."""

    UNKNOWN = "UNKNOWN"
    AWS = "AWS"
    GCP = "GCP"
    AZURE = "AZURE"
    GENERIC = "GENERIC"

    __docstrings__: ClassVar[Dict[str, str]] = {
        UNKNOWN: "Unknown cloud provider.",
        AWS: "Amazon Web Services.",
        GCP: "Google Cloud Platform.",
        AZURE: "Microsoft Azure.",
        GENERIC: "Generic cloud provider.",
    }


@dataclass(frozen=True)
class Cloud(ModelBase):
    """Minimal Cloud resource model."""

    __doc_py_example__ = """\
from datetime import datetime
from anyscale.cloud.models import Cloud, CloudProvider, ComputeStack

cloud = Cloud(
    name="my-cloud",
    id="cloud-123",
    provider="AWS",  # This will be validated as CloudProvider.AWS
    region="us-west-2",
    created_at=datetime.now(),
    is_default=True,
    compute_stack="VM"  # This will be validated as ComputeStack.VM
)
"""

    name: str = field(metadata={"docstring": "Name of this Cloud."})
    id: str = field(metadata={"docstring": "Unique identifier for this Cloud."})
    provider: Union[CloudProvider, str] = field(
        metadata={
            "docstring": "Cloud provider (AWS, GCP, AZURE, GENERIC) or UNKNOWN if not recognized."
        },
    )
    compute_stack: Union[ComputeStack, str] = field(
        metadata={
            "docstring": "The compute stack associated with this cloud's primary cloud resource, or UNKNOWN if not recognized."
        },
    )
    region: Optional[str] = field(
        default=None, metadata={"docstring": "Region for this Cloud."}
    )
    created_at: Optional[datetime] = field(
        default=None, metadata={"docstring": "When the Cloud was created."}
    )
    is_default: Optional[bool] = field(
        default=None, metadata={"docstring": "Whether this is the default cloud."}
    )
    is_aggregated_logs_enabled: Optional[bool] = field(
        default=None,
        metadata={"docstring": "Whether aggregated logs are enabled for this cloud."},
    )

    def _validate_name(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        return name

    def _validate_id(self, id: str) -> str:  # noqa: A002
        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        return id

    def _validate_provider(self, provider: Union[CloudProvider, str]) -> CloudProvider:
        if isinstance(provider, str):
            # This will raise a ValueError if the provider is unrecognized.
            provider = CloudProvider(provider)
        elif not isinstance(provider, CloudProvider):
            raise TypeError("'provider' must be a CloudProvider.")

        return provider

    def _validate_region(self, region: Optional[str]) -> Optional[str]:
        if region is not None and not isinstance(region, str):
            raise TypeError("region must be a string")
        return region

    def _validate_created_at(
        self, created_at: Optional[datetime]
    ) -> Optional[datetime]:
        if created_at is None:
            return None
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime object")
        return created_at

    def _validate_is_default(self, is_default: Optional[bool]) -> Optional[bool]:
        if is_default is not None and not isinstance(is_default, bool):
            raise TypeError("is_default must be a bool")
        return is_default

    def _validate_compute_stack(
        self, compute_stack: Union[ComputeStack, str]
    ) -> ComputeStack:
        if isinstance(compute_stack, str):
            # This will raise a ValueError if the compute_stack is unrecognized.
            compute_stack = ComputeStack(compute_stack)
        elif not isinstance(compute_stack, ComputeStack):
            raise TypeError("'compute_stack' must be a ComputeStack.")

        return compute_stack

    def _validate_is_aggregated_logs_enabled(
        self, is_aggregated_logs_enabled: Optional[bool]
    ) -> Optional[bool]:
        if is_aggregated_logs_enabled is not None and not isinstance(
            is_aggregated_logs_enabled, bool
        ):
            raise TypeError("is_aggregated_logs_enabled must be a bool")
        return is_aggregated_logs_enabled


class NetworkingMode(ModelEnum):
    """Networking mode for cloud resources."""

    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"

    __docstrings__: ClassVar[Dict[str, str]] = {
        PUBLIC: "Direct networking.",
        PRIVATE: "Customer-defined networking.",
    }


@dataclass(frozen=True)
class NFSMountTarget(ModelBase):
    """NFS mount target configuration."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
nfs_mount_target:
  address: 123.456.789.012
"""

    address: str = field(metadata={"docstring": "The address of the NFS mount target."})
    zone: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The zone of the NFS mount target. If not set, this mount target may be used in any zone."
        },
    )


@dataclass(frozen=True)
class ObjectStorage(ModelBase):
    """Object storage configuration."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
object_storage:
  bucket_name: s3://my-bucket
"""

    bucket_name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The cloud storage bucket name, prefixed with the storage scheme (s3://bucket-name, gs://bucket-name, or abfss://bucket-name@account.dfs.core.windows.net)."
        },
    )
    region: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The region for the cloud storage bucket. Defaults to the region of the cloud resource."
        },
    )
    endpoint: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The cloud storage endpoint, used to override the default cloud storage scheme's endpoint. For example, for S3, this will be passed to the AWS_ENDPOINT_URL environment variable."
        },
    )


@dataclass(frozen=True)
class FileStorage(ModelBase):
    """File storage configuration."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
file_storage:
  file_storage_id: fs-12345678901234567
"""

    file_storage_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "For AWS, the EFS ID. For GCP, the Filestore instance name."
        },
    )
    mount_targets: Optional[List[NFSMountTarget]] = field(
        default=None, metadata={"docstring": "The mount target(s) to use."}
    )
    mount_path: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "For GCP, the Filestore root directory. For NFS, the path of the server to mount from (e.g., <mount-target-address>/<mount-path> will be mounted)."
        },
    )
    persistent_volume_claim: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "For Kubernetes resources, the name of the persistent volume claim used to mount shared storage into pods."
        },
    )
    csi_ephemeral_volume_driver: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "For Kubernetes resources, the CSI ephemeral volume driver used to mount shared storage into pods."
        },
    )


@dataclass(frozen=True)
class AWSConfig(ModelBase):
    """AWS provider-specific configurations."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
aws_config:
  vpc_id: vpc-12345678901234567
  subnet_ids:
    - subnet-11111111111111111
    - subnet-22222222222222222
  security_group_ids:
    - sg-12345678901234567
  anyscale_iam_role_id: arn:aws:iam::123456789012:role/anyscale-iam-role
  cluster_iam_role_id: arn:aws:iam::123456789012:role/cluster-node-role
  memorydb_cluster_name: my-memorydb-cluster
"""

    vpc_id: Optional[str] = field(
        default=None, metadata={"docstring": "The VPC ID."},
    )
    subnet_ids: Optional[List[str]] = field(
        default=None, metadata={"docstring": "List of subnet IDs."},
    )
    zones: Optional[List[str]] = field(
        default=None,
        metadata={
            "docstring": "The availability zone corresponding to each subnet ID."
        },
    )
    security_group_ids: Optional[List[str]] = field(
        default=None, metadata={"docstring": "List of security group IDs."},
    )
    anyscale_iam_role_id: Optional[str] = field(
        default=None, metadata={"docstring": "The Anyscale IAM role ARN."},
    )
    external_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The trust policy external ID for the cross-account IAM role"
        },
    )
    cluster_iam_role_id: Optional[str] = field(
        default=None, metadata={"docstring": "The IAM role ARN used by Ray clusters."},
    )
    memorydb_cluster_name: Optional[str] = field(
        default=None, metadata={"docstring": "The MemoryDB cluster name."},
    )
    memorydb_cluster_arn: Optional[str] = field(
        default=None, metadata={"docstring": "The MemoryDB cluster ARN."},
    )
    memorydb_cluster_endpoint: Optional[str] = field(
        default=None, metadata={"docstring": "The MemoryDB cluster endpoint."},
    )
    cloudformation_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The CloudFormation stack ID, for Anyscale-managed resources."
        },
    )


@dataclass(frozen=True)
class GCPConfig(ModelBase):
    """GCP provider-specific configurations."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
gcp_config:
  project_id: my-project
  provider_name: projects/123456789012/locations/global/workloadIdentityPools/my-cloud/providers/my-provider
  vpc_name: my-vpc
  subnet_names:
    - my-subnet
  firewall_policy_names:
    - my-firewall-policy
  anyscale_service_account_email: my-anyscale-service-account@my-project.iam.gserviceaccount.com
  cluster_service_account_email: my-cluster-service-account@my-project.iam.gserviceaccount.com
  memorystore_instance_name: my-memorystore-instance
"""

    project_id: Optional[str] = field(
        default=None, metadata={"docstring": "The GCP project ID."},
    )
    host_project_id: Optional[str] = field(
        default=None, metadata={"docstring": "The host project ID for shared VPCs."},
    )
    provider_name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Workload Identity Federation provider name for Anyscale access."
        },
    )
    vpc_name: Optional[str] = field(
        default=None, metadata={"docstring": "VPC name."},
    )
    subnet_names: Optional[List[str]] = field(
        default=None, metadata={"docstring": "List of GCP subnet names."},
    )
    firewall_policy_names: Optional[List[str]] = field(
        default=None, metadata={"docstring": "List of GCP firewall policy names."},
    )
    anyscale_service_account_email: Optional[str] = field(
        default=None, metadata={"docstring": "The Anyscale service account email."},
    )
    cluster_service_account_email: Optional[str] = field(
        default=None,
        metadata={"docstring": "The service account email attached to Ray clusters."},
    )
    memorystore_instance_name: Optional[str] = field(
        default=None, metadata={"docstring": "The Memorystore instance name."},
    )
    memorystore_endpoint: Optional[str] = field(
        default=None, metadata={"docstring": "The Memorystore instance endpoint."},
    )
    deployment_manager_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The deployment manager deployment ID, for Anyscale-managed resources."
        },
    )


@dataclass(frozen=True)
class KubernetesConfig(ModelBase):
    """Kubernetes stack configurations."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
kubernetes_config:
  anyscale_operator_iam_id: arn:aws:iam::123456789012:role/anyscale-operator-role
  zones:
    - us-west-2a
    - us-west-2b
    - us-west-2c
"""

    anyscale_operator_iam_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The cloud provider IAM identity federated with the Anyscale Operator's Kubernetes service account, which will be used by Anyscale control plane for validation during Anyscale Operator bootstrap in the dataplane. IN AWS EKS, this is the ARN of the IAM role. For GCP GKE, this is the service account email."
        },
    )
    zones: Optional[List[str]] = field(
        default=None, metadata={"docstring": "List of zones to launch pods in."},
    )


################################################################################
# NOTE: The CloudResource model below is copied from the OpenAPI CloudDeployment
# model, which is what is actually used in the CLI. It is only defined here so
# that it appears in the generated docs, to provide users with examples of the
# expected YAML format. There is no CloudResource SDK support, so the name of
# this model should not actually matter. (There is also no Cloud SDK support.)
################################################################################
@dataclass(frozen=True)
class CloudResource(ModelBase):
    """Cloud resource configuration."""

    __skip_py_example__ = True

    __doc_yaml_example__ = """\
cloud_resource_id: cldrsrc_12345678901234567890123456
name: my-cloud-resource
provider: AWS
compute_stack: VM
region: us-west-2
networking_mode: PUBLIC
object_storage:
  bucket_name: s3://my-bucket
file_storage:
  file_storage_id: fs-12345678901234567
aws_config:
  vpc_id: vpc-12345678901234567
  subnet_ids:
  - subnet-11111111111111111
  - subnet-22222222222222222
  security_group_ids:
  - sg-12345678901234567
  anyscale_iam_role_id: arn:aws:iam::123456789012:role/anyscale-iam-role
  cluster_iam_role_id: arn:aws:iam::123456789012:role/cluster-node-role
  memorydb_cluster_name: my-memorydb-cluster
"""

    cloud_resource_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Unique identifier for this cloud resource."},
    )
    name: Optional[str] = field(
        default=None, metadata={"docstring": "The name of this cloud resource."},
    )
    provider: Union[CloudProvider, str] = field(
        default=CloudProvider.UNKNOWN,
        metadata={
            "docstring": "The cloud provider type (e.g., AWS, GCP, AZURE, or GENERIC)."
        },
    )
    compute_stack: Union[ComputeStack, str] = field(
        default=ComputeStack.VM,
        metadata={"docstring": "The compute stack (VM or K8S)."},
    )
    region: Optional[str] = field(
        default=None, metadata={"docstring": "The region (e.g., us-west-2)."},
    )
    networking_mode: Optional[NetworkingMode] = field(
        default=None,
        metadata={"docstring": "Whether to use public or private networking."},
    )
    object_storage: Optional[ObjectStorage] = field(
        default=None, metadata={"docstring": "Object storage configuration."},
    )
    file_storage: Optional[FileStorage] = field(
        default=None, metadata={"docstring": "File storage configuration."},
    )
    aws_config: Optional[AWSConfig] = field(
        default=None, metadata={"docstring": "AWS provider-specific configurations."},
    )
    gcp_config: Optional[GCPConfig] = field(
        default=None, metadata={"docstring": "GCP provider-specific configurations."},
    )
    kubernetes_config: Optional[KubernetesConfig] = field(
        default=None, metadata={"docstring": "Kubernetes stack configurations."},
    )

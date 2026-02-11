"""Entrypoint to generate reference docs for the Anyscale CLI/SDK.

Usage: python -m anyscale._private.docgen --help
"""

import os

import click

import anyscale
from anyscale import scripts
from anyscale._private.docgen.generator import MarkdownGenerator, Module
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters
from anyscale.cloud.models import (
    AWSConfig,
    Cloud,
    CloudPermissionLevel,
    CloudProvider,
    CloudResource,
    ComputeStack,
    CreateCloudCollaborator,
    FileStorage,
    GCPConfig,
    KubernetesConfig,
    NetworkingMode,
    NFSMountTarget,
    ObjectStorage,
)
from anyscale.commands import (
    aggregated_instance_usage_commands,
    cloud_commands,
    compute_config_commands,
    image_commands,
    job_commands,
    job_queue_commands,
    login_commands,
    logs_commands,
    machine_commands,
    machine_pool_commands,
    organization_invitation_commands,
    policy_commands,
    project_commands,
    resource_quota_commands,
    schedule_commands,
    scim_commands,
    service_account_commands,
    service_commands,
    user_commands,
    user_group_commands,
    workspace_commands_v2,
)
from anyscale.compute_config.models import (
    CloudDeployment as CloudDeploymentSelector,
    ComputeConfig,
    ComputeConfigListResult,
    ComputeConfigVersion,
    HeadNodeConfig,
    MarketType,
    MultiResourceComputeConfig,
    PhysicalResources,
    WorkerNodeGroupConfig,
)
from anyscale.image.models import ImageBuild, ImageBuildStatus
from anyscale.job.models import (
    JobConfig,
    JobLogMode,
    JobQueueConfig,
    JobQueueExecutionMode,
    JobQueueSpec,
    JobRunState,
    JobRunStatus,
    JobState,
    JobStatus,
)
from anyscale.job_queue.models import (
    ClusterState,
    ExecutionMode,
    JobQueueSortDirective,
    JobQueueSortField,
    JobQueueState,
    JobQueueStatus,
    SessionState,
    SortOrder,
)
from anyscale.organization_invitation.models import OrganizationInvitation
from anyscale.policy.models import (
    Policy,
    PolicyBinding,
    PolicyConfig,
    PolicySyncStatus,
    ResourcePolicy,
)
from anyscale.project.models import (
    CreateProjectCollaborator,
    Project,
    ProjectPermissionLevel,
    ProjectSortField,
    ProjectSortOrder,
)
from anyscale.resource_quota.models import CreateResourceQuota, Quota, ResourceQuota
from anyscale.schedule.models import ScheduleConfig, ScheduleState, ScheduleStatus
from anyscale.service.models import (
    RayGCSExternalStorageConfig,
    ServiceConfig,
    ServiceSortField,
    ServiceSortOrder,
    ServiceState,
    ServiceStatus,
    ServiceVersionState,
    ServiceVersionStatus,
    TracingConfig,
)
from anyscale.service_account.models import OrganizationPermissionLevel, ServiceAccount
from anyscale.user.models import (
    AdminCreatedUser,
    AdminCreateUser,
    AdminCreateUsers,
    User,
)
from anyscale.user_group.models import UserGroup
from anyscale.workspace.models import (
    Workspace,
    WorkspaceConfig,
    WorkspaceSortField,
    WorkspaceSortOrder,
    WorkspaceState,
)


# Defines all modules to be documented.
ALL_MODULES = [
    Module(
        title="Aggregated Instance Usage",
        filename="aggregated-instance-usage.md",
        cli_prefix="anyscale aggregated-instance-usage",
        cli_commands=[aggregated_instance_usage_commands.download_csv],
        sdk_prefix="anyscale.aggregated_instance_usage",
        sdk_commands=[anyscale.aggregated_instance_usage.download_csv,],
        models=[DownloadCSVFilters],
    ),
    Module(
        title="User",
        filename="user.md",
        cli_prefix="anyscale user",
        cli_commands=[
            user_commands.admin_batch_create,
            user_commands.list_users,
            user_commands.get_user,
            user_commands.list_permissions,
        ],
        sdk_prefix="anyscale.user",
        sdk_commands=[
            anyscale.user.admin_batch_create,
            anyscale.user.list,
            anyscale.user.get,
        ],
        models=[AdminCreateUser, AdminCreateUsers, AdminCreatedUser, User],
    ),
    Module(
        title="Project",
        filename="project-api.md",
        cli_prefix="anyscale project",
        cli_commands=[
            project_commands.get,
            project_commands.list,
            project_commands.create,
            project_commands.delete,
            project_commands.get_default,
            project_commands.add_collaborators,
        ],
        sdk_prefix="anyscale.project",
        sdk_commands=[
            anyscale.project.get,
            anyscale.project.list,
            anyscale.project.create,
            anyscale.project.delete,
            anyscale.project.get_default,
            anyscale.project.add_collaborators,
        ],
        models=[
            Project,
            ProjectSortField,
            ProjectSortOrder,
            CreateProjectCollaborator,
            ProjectPermissionLevel,
        ],
        legacy_cli_commands=[],
        legacy_sdk_commands={},
        legacy_sdk_models=[],
    ),
    Module(
        title="Job",
        filename="job-api.md",
        cli_prefix="anyscale job",
        cli_commands=[
            job_commands.submit,
            job_commands.status,
            job_commands.terminate,
            job_commands.archive,
            job_commands.delete,
            job_commands.logs,
            job_commands.wait,
            job_commands.list,
            job_commands.add_tags,
            job_commands.remove_tags,
            job_commands.list_tags,
        ],
        sdk_prefix="anyscale.job",
        sdk_commands=[
            anyscale.job.submit,
            anyscale.job.status,
            anyscale.job.terminate,
            anyscale.job.archive,
            anyscale.job.delete,
            anyscale.job.get_logs,
            anyscale.job.wait,
            anyscale.job.list,
            anyscale.job.add_tags,
            anyscale.job.remove_tags,
            anyscale.job.list_tags,
        ],
        models=[
            JobQueueSpec,
            JobQueueConfig,
            JobQueueExecutionMode,
            JobConfig,
            JobState,
            JobStatus,
            JobRunStatus,
            JobRunState,
            JobLogMode,
        ],
        cli_command_group_prefix={
            job_commands.add_tags: "tags",
            job_commands.remove_tags: "tags",
            job_commands.list_tags: "tags",
        },
        # The following commands are legacy
        legacy_sdk_commands={
            "create_job": anyscale.job.submit,
            "get_production_job": anyscale.job.status,
            "terminate_job": anyscale.job.terminate,
            "fetch_job_logs": anyscale.job.get_logs,
            "fetch_production_job_logs": anyscale.job.get_logs,
            "get_job_logs_download": anyscale.job.get_logs,
            "get_job_logs_stream": anyscale.job.get_logs,
            # limited support, no replacement yet
            "search_jobs": None,
            "list_production_jobs": None,
        },
        legacy_sdk_models=[
            "BaseJobStatus",
            "CreateClusterComputeConfig",
            "CreateJobQueueConfig",
            "CreateProductionJob",
            "CreateProductionJobConfig",
            "HaJobGoalStates",
            "HaJobStates",
            "Job",
            "JobListResponse",
            "JobQueueConfig",
            "JobQueueExecutionMode",
            "JobQueueSpec",
            "JobRunType",
            "JobStatus",
            "JobsSortField",
            "ProductionJob",
            "ProductionJobConfig",
            "ProductionJobStateTransition",
            "ProductionjobListResponse",
            "ProductionjobResponse",
            "RayRuntimeEnvConfig",
            "SortByClauseJobsSortField",
        ],
    ),
    Module(
        title="Schedule",
        filename="schedule-api.md",
        cli_prefix="anyscale schedule",
        cli_commands=[
            schedule_commands.apply,
            schedule_commands.list,
            schedule_commands.pause,
            schedule_commands.resume,
            schedule_commands.status,
            schedule_commands.trigger,
            schedule_commands.url,
            schedule_commands.delete,
        ],
        sdk_prefix="anyscale.schedule",
        sdk_commands=[
            anyscale.schedule.apply,
            anyscale.schedule.set_state,
            anyscale.schedule.status,
            anyscale.schedule.trigger,
            anyscale.schedule.list,
            anyscale.schedule.url,
            anyscale.schedule.delete,
        ],
        models=[ScheduleConfig, ScheduleState, ScheduleStatus],
        # The following commands are legacy
        legacy_sdk_commands={
            "create_or_update_schedule": anyscale.schedule.apply,
            "get_schedule": anyscale.schedule.status,
            "pause_schedule": anyscale.schedule.set_state,
            "run_schedule": anyscale.schedule.trigger,
            # limited support, no replacement yet
            "list_schedules": None,
        },
    ),
    Module(
        title="Service",
        filename="service-api.md",
        cli_prefix="anyscale service",
        cli_commands=[
            service_commands.list,
            service_commands.deploy,
            service_commands.status,
            service_commands.wait,
            service_commands.rollback,
            service_commands.terminate,
            service_commands.archive,
            service_commands.delete,
            service_commands.add_tags,
            service_commands.remove_tags,
            service_commands.list_tags,
        ],
        sdk_prefix="anyscale.service",
        sdk_commands=[
            anyscale.service.list,
            anyscale.service.deploy,
            anyscale.service.status,
            anyscale.service.wait,
            anyscale.service.rollback,
            anyscale.service.terminate,
            anyscale.service.archive,
            anyscale.service.delete,
            anyscale.service.add_tags,
            anyscale.service.remove_tags,
            anyscale.service.list_tags,
        ],
        models=[
            ServiceConfig,
            RayGCSExternalStorageConfig,
            TracingConfig,
            ServiceStatus,
            ServiceState,
            ServiceVersionStatus,
            ServiceVersionState,
            ServiceSortField,
            ServiceSortOrder,
        ],
        cli_command_group_prefix={
            service_commands.add_tags: "tags",
            service_commands.remove_tags: "tags",
            service_commands.list_tags: "tags",
        },
        # The following commands are legacy
        legacy_sdk_commands={
            "get_service": anyscale.service.status,
            "rollback_service": anyscale.service.rollback,
            "rollout_service": anyscale.service.deploy,
            "terminate_service": anyscale.service.terminate,
            # limited support, no replacement yet
            "list_services": None,
        },
        legacy_sdk_models=[
            "ApplyProductionServiceV2Model",
            "ApplyServiceModel",
            "ListServiceModel",
            "ListservicemodelListResponse",
            "ProductionServiceV2Model",
            "ProductionServiceV2VersionModel",
            "Productionservicev2ModelResponse",
            "RollbackServiceModel",
            "RolloutStrategy",
            "ServiceConfig",
            "ServiceEventCurrentState",
            "ServiceGoalStates",
            "ServiceModel",
            "ServiceObservabilityUrls",
            "ServiceSortField",
            "ServiceType",
            "ServiceVersionState",
            "ServicemodelListResponse",
            "ServicemodelResponse",
        ],
    ),
    Module(
        title="Compute Config",
        filename="compute-config-api.md",
        cli_prefix="anyscale compute-config",
        cli_commands=[
            compute_config_commands.create_compute_config,
            compute_config_commands.get_compute_config,
            compute_config_commands.archive_compute_config,
            compute_config_commands.list_compute_configs,
        ],
        sdk_prefix="anyscale.compute_config",
        sdk_commands=[
            anyscale.compute_config.create,
            anyscale.compute_config.get,
            anyscale.compute_config.get_default,
            anyscale.compute_config.archive,
            anyscale.compute_config.list,
        ],
        models=[
            ComputeConfig,
            ComputeConfigListResult,
            MultiResourceComputeConfig,
            HeadNodeConfig,
            WorkerNodeGroupConfig,
            MarketType,
            CloudDeploymentSelector,
            ComputeConfigVersion,
            PhysicalResources,
        ],
        legacy_sdk_commands={},
        legacy_sdk_models=[],
    ),
    Module(
        title="Service Account",
        filename="service-account.md",
        cli_prefix="anyscale service-account",
        cli_commands=[
            service_account_commands.create,
            service_account_commands.create_api_key,
            service_account_commands.list_service_accounts,
            service_account_commands.delete,
        ],
        sdk_prefix="anyscale.service_account",
        sdk_commands=[
            anyscale.service_account.create,
            anyscale.service_account.create_api_key,
            anyscale.service_account.list,
            anyscale.service_account.delete,
        ],
        models=[ServiceAccount, OrganizationPermissionLevel],
    ),
    Module(
        title="Image",
        filename="image.md",
        cli_prefix="anyscale image",
        cli_commands=[
            image_commands.build,
            image_commands.get,
            image_commands.list,
            image_commands.register,
            image_commands.archive,
        ],
        sdk_prefix="anyscale.image",
        sdk_commands=[
            anyscale.image.build,
            anyscale.image.get,
            anyscale.image.list,
            anyscale.image.register,
            anyscale.image.archive,
        ],
        models=[ImageBuildStatus, ImageBuild],
    ),
    Module(
        title="Cloud",
        filename="cloud.md",
        cli_prefix="anyscale cloud",
        cli_commands=[
            cloud_commands.setup_cloud,
            cloud_commands.register_cloud,
            cloud_commands.cloud_update,
            cloud_commands.cloud_delete,
            cloud_commands.cloud_verify,
            cloud_commands.list_cloud,
            cloud_commands.cloud_resource_create,
            cloud_commands.cloud_resource_delete,
            cloud_commands.cloud_config_get,
            cloud_commands.cloud_config_update,
            cloud_commands.cloud_set_default,
            cloud_commands.add_collaborators,
            cloud_commands.get_cloud,
            cloud_commands.get_default_cloud,
            cloud_commands.terminate_system_cluster,
        ],
        sdk_prefix="anyscale.cloud",
        sdk_commands=[
            anyscale.cloud.add_collaborators,
            anyscale.cloud.list,
            anyscale.cloud.get,
            anyscale.cloud.get_default,
            anyscale.cloud.terminate_system_cluster,
        ],
        models=[
            Cloud,
            CloudPermissionLevel,
            CreateCloudCollaborator,
            CloudResource,
            ComputeStack,
            CloudProvider,
            NetworkingMode,
            ObjectStorage,
            FileStorage,
            NFSMountTarget,
            AWSConfig,
            GCPConfig,
            KubernetesConfig,
        ],
        cli_command_group_prefix={
            cloud_commands.cloud_resource_create: "resource",
            cloud_commands.cloud_resource_delete: "resource",
            cloud_commands.cloud_config_get: "config",
            cloud_commands.cloud_config_update: "config",
        },
        legacy_sdk_commands={
            # limited support, no replacement yet
            "get_cloud": None,
            "get_default_cloud": None,
            "search_clouds": None,
        },
        legacy_sdk_models=[
            "Cloud",
            "CloudConfig",
            "CloudListResponse",
            "CloudProviders",
            "CloudResponse",
            "CloudState",
            "CloudStatus",
            "CloudTypes",
            "CloudVersion",
            "CloudsQuery",
        ],
    ),
    Module(
        title="Logs",
        filename="logs.md",
        cli_prefix="anyscale logs",
        cli_commands=[
            logs_commands.anyscale_logs_workspace,
            logs_commands.anyscale_logs_service,
            logs_commands.anyscale_logs_job,
            logs_commands.anyscale_logs_cluster,
        ],
        sdk_prefix="anyscale.logs",
        sdk_commands=[],
        models=[],
    ),
    Module(
        title="Workspaces",
        filename="workspaces.md",
        cli_prefix="anyscale workspace_v2",
        cli_commands=[
            workspace_commands_v2.create,
            workspace_commands_v2.start,
            workspace_commands_v2.terminate,
            workspace_commands_v2.status,
            workspace_commands_v2.wait,
            workspace_commands_v2.ssh,
            workspace_commands_v2.run_command,
            workspace_commands_v2.pull,
            workspace_commands_v2.push,
            workspace_commands_v2.update,
            workspace_commands_v2.get,
            workspace_commands_v2.list,
            workspace_commands_v2.add_tags,
            workspace_commands_v2.remove_tags,
            workspace_commands_v2.list_tags,
        ],
        sdk_prefix="anyscale.workspace",
        sdk_commands=[
            anyscale.workspace.create,
            anyscale.workspace.start,
            anyscale.workspace.terminate,
            anyscale.workspace.status,
            anyscale.workspace.wait,
            anyscale.workspace.generate_ssh_config_file,
            anyscale.workspace.run_command,
            anyscale.workspace.list,
            anyscale.workspace.add_tags,
            anyscale.workspace.remove_tags,
            anyscale.workspace.list_tags,
        ],
        models=[
            WorkspaceConfig,
            WorkspaceState,
            Workspace,
            WorkspaceSortField,
            WorkspaceSortOrder,
        ],
        cli_command_group_prefix={
            workspace_commands_v2.add_tags: "tags",
            workspace_commands_v2.remove_tags: "tags",
            workspace_commands_v2.list_tags: "tags",
        },
    ),
    Module(
        title="Job Queue",
        filename="job-queue.md",
        cli_prefix="anyscale job-queue",
        cli_commands=[
            job_queue_commands.list_job_queues,
            job_queue_commands.update_job_queue,
            job_queue_commands.status,
            job_queue_commands.archive_job_queue,
            job_queue_commands.terminate_job_queue,
            job_queue_commands.delete_job_queue,
            job_queue_commands.add_tags,
            job_queue_commands.remove_tags,
            job_queue_commands.list_tags,
        ],
        sdk_prefix="anyscale.job_queue",
        sdk_commands=[
            anyscale.job_queue.list,
            anyscale.job_queue.status,
            anyscale.job_queue.update,
            anyscale.job_queue.archive,
            anyscale.job_queue.terminate,
            anyscale.job_queue.delete,
            anyscale.job_queue.add_tags,
            anyscale.job_queue.remove_tags,
            anyscale.job_queue.list_tags,
        ],
        models=[
            JobQueueStatus,
            JobQueueState,
            ExecutionMode,
            ClusterState,
            JobQueueSortField,
            JobQueueSortDirective,
            SortOrder,
            SessionState,
        ],
        cli_command_group_prefix={
            job_queue_commands.add_tags: "tags",
            job_queue_commands.remove_tags: "tags",
            job_queue_commands.list_tags: "tags",
        },
    ),
    Module(
        title="Machine Pool",
        filename="machine-pool.md",
        cli_prefix="anyscale machine-pool",
        cli_commands=[
            machine_pool_commands.create_machine_pool,
            machine_pool_commands.update_machine_pool,
            machine_pool_commands.delete_machine_pool,
            machine_pool_commands.attach_machine_pool_to_cloud,
            machine_pool_commands.detach_machine_pool_from_cloud,
            machine_pool_commands.list_machine_pools,
            machine_pool_commands.describe,
        ],
        sdk_prefix="anyscale.machine_pool",
        sdk_commands=[],
        models=[],
    ),
    Module(
        title="Cluster",
        filename="cluster.md",
        cli_prefix="anyscale cluster",
        cli_commands=[],
        sdk_prefix="anyscale.cluster",  # fake name used as id, cluster has no SDK commands
        sdk_commands=[],
        models=[],
        legacy_sdk_commands={
            "launch_cluster": None,
            "launch_cluster_with_new_cluster_environment": None,
            "create_cluster": None,
            "delete_cluster": None,
            "get_cluster": None,
            "search_clusters": None,
            "start_cluster": None,
            "terminate_cluster": None,
            "update_cluster": None,
        },
        legacy_sdk_models=[
            "ClusterHeadNodeInfo",
            "ClusterListResponse",
            "ClusterManagementStackVersions",
            "ClusterOperation",
            "ClusterResponse",
            "ClusterServicesUrls",
            "ClusterState",
            "ClusterStatus",
            "ClusterStatusDetails",
            "ClusteroperationResponse",
            "ClustersQuery",
            "CreateCluster",
            "StartClusterOptions",
            "TerminateClusterOptions",
            "UpdateCluster",
        ],
    ),
    Module(
        title="Resource quotas",
        filename="resource-quotas.md",
        cli_prefix="anyscale resource-quota",
        cli_commands=[
            resource_quota_commands.create,
            resource_quota_commands.list_resource_quotas,
            resource_quota_commands.delete,
            resource_quota_commands.enable,
            resource_quota_commands.disable,
        ],
        sdk_prefix="anyscale.resource_quota",
        sdk_commands=[
            anyscale.resource_quota.create,
            anyscale.resource_quota.list,
            anyscale.resource_quota.delete,
            anyscale.resource_quota.enable,
            anyscale.resource_quota.disable,
        ],
        models=[CreateResourceQuota, Quota, ResourceQuota],
    ),
    Module(
        title="Organization Invitation",
        filename="organization-invitation.md",
        cli_prefix="anyscale organization-invitation",
        cli_commands=[
            organization_invitation_commands.create,
            organization_invitation_commands.list,
            organization_invitation_commands.delete,
        ],
        sdk_prefix="anyscale.organization-invitation",
        sdk_commands=[
            anyscale.organization_invitation.create,
            anyscale.organization_invitation.list,
            anyscale.organization_invitation.delete,
        ],
        models=[OrganizationInvitation],
    ),
    Module(
        title="User Group",
        filename="user-group.md",
        cli_prefix="anyscale user-group",
        cli_commands=[
            user_group_commands.list_user_groups,
            user_group_commands.get_user_group,
            user_group_commands.list_memberships,
        ],
        sdk_prefix="anyscale.user_group",
        sdk_commands=[anyscale.user_group.list, anyscale.user_group.get,],
        models=[UserGroup],
    ),
    Module(
        title="Policy",
        filename="policy.md",
        cli_prefix="anyscale policy",
        cli_commands=[
            policy_commands.set_policy,
            policy_commands.get_policy,
            policy_commands.list_policies,
        ],
        sdk_prefix="anyscale.policy",
        sdk_commands=[anyscale.policy.set, anyscale.policy.get, anyscale.policy.list,],
        models=[Policy, PolicyBinding, PolicyConfig, PolicySyncStatus, ResourcePolicy],
    ),
    Module(
        title="SCIM",
        filename="scim.md",
        cli_prefix="anyscale scim",
        cli_commands=[scim_commands.enforce_group_permissions,],
        sdk_prefix="anyscale.scim",
        sdk_commands=[],
        models=[],
    ),
    Module(
        title="Other",
        filename="other.md",
        cli_prefix="anyscale",
        cli_commands=[
            login_commands.anyscale_login,
            login_commands.anyscale_logout,
            scripts.version_cli,
        ],
        sdk_prefix="",
        sdk_commands=[],
        models=[],
        # The following commands are legacy
        legacy_cli_prefix="anyscale",
        legacy_cli_commands=[
            machine_commands.delete_machine,
            machine_commands.list_machines,
        ],
        legacy_sdk_commands={
            "partial_update_organization": None,
            "upsert_sso_config": None,
            "upsert_test_sso_config": None,
        },
        legacy_sdk_models=[
            "AccessConfig",
            "ArchiveStatus",
            "BASEIMAGESENUM",
            "BaseJobStatus",
            "Build",
            "BuildResponse",
            "BuildStatus",
            "CreateSSOConfig",
            "GrpcProtocolConfig",
            "HTTPValidationError",
            "HttpProtocolConfig",
            "IdleTerminationStatus",
            "ListResponseMetadata",
            "NodeType",
            "OperationError",
            "OperationProgress",
            "OperationResult",
            "Organization",
            "OrganizationResponse",
            "PageQuery",
            "Protocols",
            "PythonModules",
            "PythonVersion",
            "RayGCSExternalStorageConfig",
            "Resources",
            "SSOConfig",
            "SSOMode",
            "SUPPORTEDBASEIMAGESENUM",
            "SortOrder",
            "SsoconfigResponse",
            "StaticSSOConfig",
            "TextQuery",
            "TracingConfig",
            "UpdateOrganization",
            "UserServiceAccessTypes",
            "WorkerNodeType",
        ],
    ),
]


@click.command(help="Generate markdown docs for the Anyscale CLI & SDK.")
@click.argument("output_dir")
@click.option(
    "-r",
    "--remove-existing",
    is_flag=True,
    default=False,
    help="If set, all files in the 'output_dir' that were not generated will be removed.",
)
def generate(
    output_dir: str, *, remove_existing: bool = False,
):
    if not os.path.isdir(output_dir):
        raise RuntimeError(f"output_dir '{output_dir}' does not exist.")

    gen = MarkdownGenerator(ALL_MODULES)

    generated_files = set()
    os.makedirs(output_dir, exist_ok=True)

    # Create legacy subdirectory
    legacy_dir = os.path.join(output_dir, "legacy")
    os.makedirs(legacy_dir, exist_ok=True)

    for filename, content in gen.generate().items():
        generated_files.add(filename)
        full_path = os.path.join(output_dir, filename)

        # Create directory if it doesn't exist (for legacy/ files)
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        print(f"Writing output file {full_path}")
        with open(full_path, "w") as f:
            f.write(content)

    if remove_existing:
        # Get all existing files (including in subdirectories)
        existing_files = set()
        for root, _dirs, files in os.walk(output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                existing_files.add(rel_path)

        # Remove files that weren't generated
        to_remove = existing_files - generated_files
        for path in to_remove:
            full_path = os.path.join(output_dir, path)
            if os.path.exists(full_path):
                print(f"Removing existing file {full_path}")
                os.unlink(full_path)


if __name__ == "__main__":
    generate()

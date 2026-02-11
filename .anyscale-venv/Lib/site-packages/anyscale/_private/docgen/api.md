---
title: "API"
description: >-
  A comprehensive list of all available functions and models available in our
  Python SDK.
sidebar_position: 2
---

# Python SDK Reference

:::note
Please note that only `AnyscaleSDK` and imports under `anyscale.sdk` are considered public.
:::

## AnyscaleSDK

The AnyscaleSDK class must be constructed in order to make calls to the SDK. This class allows you to create an authenticated client in which to use the SDK.

| Param | Type | Description |
| :--- | :--- | :--- |
| `auth_token` | Optional String | Authentication token used to verify you have permissions to access Anyscale. If not provided, we will default to the credentials set for your current user. Credentials can be set by following the instructions on this page: https://console.anyscale.com/credentials |

**Example**
```python
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK()
```

## Utility Functions

### build_cluster_environment

Creates a new Cluster Environment and waits for build to complete.

If a Cluster Environment with the same name already exists, this will create an updated build of that environment.

Returns the newly created ClusterEnvironmentBuild object.

Raises an Exception if building Cluster Environment fails or times out.

| Param | Type | Description |
| :--- | :--- | :--- |
| `create_cluster_environment` | CreateClusterEnvironment | CreateClusterEnvironment object |
| `poll_rate_seconds` | Optional Integer | seconds to wait when polling build operation status; defaults to 15 |
| `timeout_seconds` | Optional Integer | maximum number of seconds to wait for build operation to complete before timing out; defaults to no timeout |

**Example**

```python
from anyscale.sdk.anyscale_client.models.create_cluster_environment import (
    CreateClusterEnvironment,
)
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

create_cluster_environment = CreateClusterEnvironment(
    name="my-cluster-environment",
    config_json={"base_image": "anyscale/ray:1.4.1-py37"}
)

cluster_environment_build = sdk.build_cluster_environment(
    create_cluster_environment=create_cluster_environment
)

print(f"Cluster Environment built successfully: {cluster_environment_build}")
```

### fetch_actor_logs

Retrieves logs for an Actor.

This function may take several minutes if the Cluster this Actor ran on has been terminated.

Returns the log output as a string.

Raises an Exception if fetching logs fails.

| Param | Type | Description |
| :--- | :--- | :--- |
| `actor_id` | String | ID of the Actor |

**Example**

```python
from anyscale.sdk.anyscale_client.models.create_cluster_environment import (
    CreateClusterEnvironment,
)
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

actor_logs = sdk.fetch_actor_logs(actor_id="actor_id")

print(actor_logs)
```

### fetch_job_logs

Retrieves logs for a Job.

This function may take several minutes if the Cluster this Job ran on has been terminated.

Returns the log output as a string.

Raises an Exception if fetching logs fails.

| Param | Type | Description |
| :--- | :--- | :--- |
| `job_id` | String | ID of the Job |

**Example**

```python
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

job_logs = sdk.fetch_job_logs(job_id="job_id")

print(job_logs)
```

### fetch_production_job_logs

Retrieves logs for a Production Job.

This function may take several minutes if the Cluster this Production Job ran on has been terminated.

Returns the log output as a string.

Raises an Exception if fetching logs fails.

| Param | Type | Description |
| :--- | :--- | :--- |
| `job_id` | String | ID of the Job |

**Example**

```python
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

job_logs = sdk.fetch_production_job_logs(job_id="production_job_id")

print(job_logs)
```

### launch_cluster

Starts a Cluster in the specified Project.

If a Cluster with the specified name already exists, we will update that Cluster.
Otherwise, a new Cluster will be created.

Returns the started Cluster object.

Raises an Exception if starting Cluster fails or times out.

| Param | Type | Description |
| :--- | :--- | :--- |
| `project_id` | String | ID of the Project the Cluster belongs to |
| `cluster_name` | String | Name of the Cluster |
| `cluster_environment_build_id` | String | ID of the Cluster Environment Build to start this Cluster with |
| `cluster_compute_id` | Optional String | ID of the Cluster Compute to start this Cluster with |
| `cluster_compute_config` | Optional Dict | One-off Cluster Compute that this Cluster will use, with same fields as ClusterComputeConfig |
| `poll_rate_seconds` | Optional Integer | seconds to wait when polling cluster operation status; defaults to 15 |
| `idle_timeout_minutes` | Optional Integer | Idle timeout (in minutes), after which the Cluster is terminated; Defaults to 120 minutes. |
| `timeout_seconds` | Optional Integer | maximum number of seconds to wait for cluster operation to complete before timing out; defaults to no timeout |

**Example**

```python
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

cluster = sdk.launch_cluster(
    project_id="project_id",
    cluster_name="my-cluster",
    cluster_environment_build_id="cluster_environment_build_id",
    cluster_compute_id="cluster_compute_id")

print(f"Cluster started successfully: {cluster}")
```

### launch_cluster_with_new_cluster_environment

Builds a new Cluster Environment, then starts a Cluster in the specified Project with the new build.

If a Cluster with the specified name already exists, we will update that Cluster. Otherwise, a new Cluster will be created.

Returns the started Cluster object.

Raises an Exception if building Cluster Environment fails or starting the Cluster fails.

| Param | Type | Description |
| :--- | :--- | :--- |
| `project_id` | String | ID of the Project the Cluster belongs to |
| `cluster_name` | String | Name of the Cluster |
| `create_cluster_environment` | CreateClusterEnvironment | CreateClusterEnvironment object |
| `cluster_compute_id` | Optional String | Cluster Compute to start this Cluster with |
| `cluster_compute_config` | Optional Dict | One-off Cluster Compute that this Cluster will use, with same fields as ClusterComputeConfig |
| `poll_rate_seconds` | Optional Integer | seconds to wait when polling operation status; defaults to 15 |
| `timeout_seconds` | Optional Integer | maximum number of seconds to wait for operations to complete before timing out; defaults to no timeout |

**Example**

```python
from anyscale import AnyscaleSDK
from anyscale.sdk.anyscale_client.models.create_cluster_environment import (
    CreateClusterEnvironment,
)

sdk = AnyscaleSDK(auth_token="sss_YourAuthToken")

create_cluster_environment = CreateClusterEnvironment(
    name="my-cluster-environment",
    config_json={"base_image": "anyscale/ray:1.4.1-py37"}
)

cluster = sdk.launch_cluster_with_new_cluster_environment(
    project_id="project_id",
    cluster_name="my-cluster",
    create_cluster_environment=create_cluster_environment,
    cluster_compute_id="cluster_compute_id",
)

print(f"Cluster started successfully: {cluster}")
```

:::info
The OpenAPI schemas for types below can be found at the [Anyscale OpenAPI Documentation](https://api.anyscale.com/v0/docs#)

All of the following functions are synchronous by default. To make an asynchronous HTTP request, please pass async\_req=True. The return value for the asynchronous calls is a thread.

```text
thread = api.create_cloud(create_cloud, async_req=True)
result = thread.get()
```
:::

## Clouds

### get_cloud

Retrieves a Cloud.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cloud_id` | str| ID of the Cloud to retrieve. | Defaults to null

Returns [CloudResponse](./models.md#cloudresponse)

### get_default_cloud

Retrieves the default cloud for the logged in user. First prefers the default cloud set by the user's org, then the last used cloud.

Parameters
This function does not have any parameters.

Returns [CloudResponse](./models.md#cloudresponse)

### search_clouds

Searches for all accessible Clouds that satisfies the query.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `clouds_query` | [CloudsQuery](./models.md#cloudsquery)|  | 

Returns [CloudListResponse](./models.md#cloudlistresponse)

## Cluster Computes

### create_cluster_compute

Creates a Cluster Compute. If the specified compute config is anonymous, returns an existing compute config if an anonymous one exists with the same config.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_cluster_compute` | [CreateClusterCompute](./models.md#createclustercompute)|  | 

Returns [ClustercomputeResponse](./models.md#clustercomputeresponse)

### delete_cluster_compute

Deletes a Cluster Compute.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_compute_id` | str| ID of the Cluster Compute to delete. | Defaults to null

Returns void (empty response body)

### get_cluster_compute

Retrieves a Cluster Compute.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_compute_id` | str| ID of the Cluster Compute to retrieve. | Defaults to null

Returns [ClustercomputeResponse](./models.md#clustercomputeresponse)

### get_default_cluster_compute

Returns a default cluster compute that can be used with a given cloud.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cloud_id` | optional str| The cloud id whose default cluster compute you want to fetch. If None, will use the default cloud. | Defaults to null

Returns [ClustercomputeResponse](./models.md#clustercomputeresponse)

### search_cluster_computes

Lists all Cluster Computes the user has access to, matching the input query.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_computes_query` | [ClusterComputesQuery](./models.md#clustercomputesquery)|  | 

Returns [ClustercomputeListResponse](./models.md#clustercomputelistresponse)

## Cluster Environment Builds

### create_byod_cluster_environment_build

Creates and starts a BYOD Cluster Environment Build.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_byod_cluster_environment_build` | [CreateBYODClusterEnvironmentBuild](./models.md#createbyodclusterenvironmentbuild)|  | 

Returns [ClusterenvironmentbuildoperationResponse](./models.md#clusterenvironmentbuildoperationresponse)

### create_cluster_environment_build

Creates and starts a Cluster Environment Build. This is a long running operation.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_cluster_environment_build` | [CreateClusterEnvironmentBuild](./models.md#createclusterenvironmentbuild)|  | 

Returns [ClusterenvironmentbuildoperationResponse](./models.md#clusterenvironmentbuildoperationresponse)

### find_cluster_environment_build_by_identifier

Looks for a cluster environment build given a cluster environment identifier. Identifiers are in the format my-cluster-env:1

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `identifier` | str| Identifier of the cluster env to look for. Identifiers are in the format my-cluster-env:1 | Defaults to null

Returns [ClusterenvironmentbuildResponse](./models.md#clusterenvironmentbuildresponse)

### get_cluster_environment_build

Retrieves a Cluster Environment Build.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_environment_build_id` | str| ID of the Cluster Environment Build to retrieve. | Defaults to null

Returns [ClusterenvironmentbuildResponse](./models.md#clusterenvironmentbuildresponse)

### get_default_cluster_environment_build

Retrieves a default cluster environment with the preferred attributes.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `python_version` | [PythonVersion](./models.md#pythonversion)| Python version for the cluster environment | Defaults to null
 `ray_version` | str| Ray version to use for this cluster environment. Should match a version string found in the ray version history on pypi. See here for full list: https://pypi.org/project/ray/#history | Defaults to null

Returns [ClusterenvironmentbuildResponse](./models.md#clusterenvironmentbuildresponse)

### list_cluster_environment_builds

Lists all Cluster Environment Builds belonging to an Cluster Environment.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_environment_id` | str| ID of the Cluster Environment to list builds for. | Defaults to null
 `desc` | optional bool| Orders the list of builds from latest to oldest. | Defaults to false
 `paging_token` | optional str|  | Defaults to null
 `count` | optional int|  | Defaults to 10

Returns [ClusterenvironmentbuildListResponse](./models.md#clusterenvironmentbuildlistresponse)

## Cluster Environments

### create_byod_cluster_environment

Creates a BYOD Cluster Environment.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_byod_cluster_environment` | [CreateBYODClusterEnvironment](./models.md#createbyodclusterenvironment)|  | 

Returns [ClusterenvironmentResponse](./models.md#clusterenvironmentresponse)

### create_cluster_environment

Creates a Cluster Environment.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_cluster_environment` | [CreateClusterEnvironment](./models.md#createclusterenvironment)|  | 

Returns [ClusterenvironmentResponse](./models.md#clusterenvironmentresponse)

### get_cluster_environment

Retrieves a Cluster Environment.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_environment_id` | str| ID of the Cluster Environment to retrieve. | Defaults to null

Returns [ClusterenvironmentResponse](./models.md#clusterenvironmentresponse)

### search_cluster_environments

Lists all Cluster Environments that the logged in user has permissions to access.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_environments_query` | [ClusterEnvironmentsQuery](./models.md#clusterenvironmentsquery)|  | 

Returns [ClusterenvironmentListResponse](./models.md#clusterenvironmentlistresponse)

## Cluster Operations

### get_cluster_operation

Retrieves a Cluster Operation.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_operation_id` | str| ID of the Cluster Operation to retrieve. | Defaults to null

Returns [ClusteroperationResponse](./models.md#clusteroperationresponse)

## Clusters

### create_cluster

Creates a Cluster.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_cluster` | [CreateCluster](./models.md#createcluster)|  | 

Returns [ClusterResponse](./models.md#clusterresponse)

### delete_cluster

Deletes a Cluster.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_id` | str| ID of the Cluster to delete. | Defaults to null

Returns void (empty response body)

### get_cluster

Retrieves a Cluster.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_id` | str| ID of the Cluster to retreive. | Defaults to null

Returns [ClusterResponse](./models.md#clusterresponse)

### search_clusters

Searches for all Clusters the user has access to that satisfies the query.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `clusters_query` | [ClustersQuery](./models.md#clustersquery)|  | 

Returns [ClusterListResponse](./models.md#clusterlistresponse)

### start_cluster

Initializes workflow to transition the Cluster into the Running state. This is a long running operation. Clients will need to poll the operation's status to determine completion. The options parameter is DEPRECATED.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_id` | str| ID of the Cluster to start. | Defaults to null
 `start_cluster_options` | [StartClusterOptions](./models.md#startclusteroptions)|  | 

Returns [ClusteroperationResponse](./models.md#clusteroperationresponse)

### terminate_cluster

Initializes workflow to transition the Cluster into the Terminated state. This is a long running operation. Clients will need to poll the operation's status to determine completion.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_id` | str| ID of the Cluster to terminate. | Defaults to null
 `terminate_cluster_options` | [TerminateClusterOptions](./models.md#terminateclusteroptions)|  | 

Returns [ClusteroperationResponse](./models.md#clusteroperationresponse)

### update_cluster

Updates a Cluster.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `cluster_id` | str| ID of the Cluster to update. | Defaults to null
 `update_cluster` | [UpdateCluster](./models.md#updatecluster)|  | 

Returns [ClusterResponse](./models.md#clusterresponse)

## Jobs

### search_jobs

DEPRECATED: This API is now deprecated. Use list_production_jobs instead.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `jobs_query` | [JobsQuery](./models.md#jobsquery)|  | 

Returns [JobListResponse](./models.md#joblistresponse)

## Logs

### get_job_logs_download



Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `job_id` | str|  | Defaults to null
 `all_logs` | optional bool| Whether to grab all logs. | Defaults to true

Returns [LogdownloadresultResponse](./models.md#logdownloadresultresponse)

### get_job_logs_stream



Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `job_id` | str|  | Defaults to null

Returns [LogstreamResponse](./models.md#logstreamresponse)

## Organizations

### partial_update_organization

Update an organization's requirement for Single Sign On (SSO). If SSO is required for an organization, SSO will be the only way to log in to it.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `organization_id` | str| ID of the Organization to update. | Defaults to null
 `update_organization` | [UpdateOrganization](./models.md#updateorganization)|  | 

Returns [OrganizationResponse](./models.md#organizationresponse)

## Production Jobs

### create_job

Create an Production Job

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_production_job` | [CreateProductionJob](./models.md#createproductionjob)|  | 

Returns [ProductionjobResponse](./models.md#productionjobresponse)

### get_production_job

Get an Production Job

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `production_job_id` | str|  | Defaults to null

Returns [ProductionjobResponse](./models.md#productionjobresponse)

### list_production_jobs



Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `project_id` | optional str| project_id to filter by | Defaults to null
 `name` | optional str| name to filter by | Defaults to null
 `state_filter` | [List[HaJobStates]](./models.md#hajobstates)| A list of session states to filter by | Defaults to []
 `creator_id` | optional str| filter by creator id | Defaults to null
 `paging_token` | optional str|  | Defaults to null
 `count` | optional int|  | Defaults to null

Returns [ProductionjobListResponse](./models.md#productionjoblistresponse)

### terminate_job

Terminate an Production Job

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `production_job_id` | str|  | Defaults to null

Returns [ProductionjobResponse](./models.md#productionjobresponse)

## Projects

### create_project

Creates a Project.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_project` | [CreateProject](./models.md#createproject)|  | 

Returns [ProjectResponse](./models.md#projectresponse)

### delete_project

Deletes a Project.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `project_id` | str| ID of the Project to delete. | Defaults to null

Returns void (empty response body)

### get_default_project

Retrieves the default project.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `parent_cloud_id` | optional str| Cloud to fetch this default project for. This is only required if cloud isolation is enabled. | Defaults to null

Returns [ProjectResponse](./models.md#projectresponse)

### get_project

Retrieves a Project.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `project_id` | str| ID of the Project to retrieve. | Defaults to null

Returns [ProjectResponse](./models.md#projectresponse)

### search_projects

Searches for all Projects the user has access to that satisfies the query.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `projects_query` | [ProjectsQuery](./models.md#projectsquery)|  | 

Returns [ProjectListResponse](./models.md#projectlistresponse)

## Schedules

### create_or_update_schedule

Create or update a Schedule

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_schedule` | [CreateSchedule](./models.md#createschedule)|  | 

Returns [ScheduleapimodelResponse](./models.md#scheduleapimodelresponse)

### get_schedule

Get Schedules

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `schedule_id` | str|  | Defaults to null

Returns [ScheduleapimodelResponse](./models.md#scheduleapimodelresponse)

### list_schedules

List Schedules

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `project_id` | optional str| project_id to filter by | Defaults to null
 `name` | optional str| name to filter by | Defaults to null
 `creator_id` | optional str| filter by creator id | Defaults to null

Returns [ScheduleapimodelListResponse](./models.md#scheduleapimodellistresponse)

### pause_schedule

Pause a Schedule

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `schedule_id` | str|  | Defaults to null
 `pause_schedule` | [PauseSchedule](./models.md#pauseschedule)|  | 

Returns [ScheduleapimodelResponse](./models.md#scheduleapimodelresponse)

### run_schedule

Run a Schedule manually

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `schedule_id` | str|  | Defaults to null

Returns [ProductionjobResponse](./models.md#productionjobresponse)

## Services

### archive_service

Archives a Service. It is a no-op if already archived.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `service_id` | str|  | Defaults to null

Returns void (empty response body)

### get_service

Get a Service

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `service_id` | str|  | Defaults to null

Returns [ServicemodelResponse](./models.md#servicemodelresponse)

### list_services



Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `project_id` | optional str| project_id to filter by | Defaults to null
 `name` | optional str| name to filter by | Defaults to null
 `state_filter` | [List[ServiceEventCurrentState]](./models.md#serviceeventcurrentstate)| A list of Service states to filter by | Defaults to []
 `archive_status` | [ArchiveStatus](./models.md#archivestatus)| The archive status to filter by. Defaults to unarchived. | Defaults to null
 `creator_id` | optional str| creator_id to filter by | Defaults to null
 `cloud_id` | optional str| cloud_id to filter by | Defaults to null
 `sort_field` | [ServiceSortField](./models.md#servicesortfield)| If absent, the default sorting order is 1. status (active first).2. Last updated at (desc). 3. Name (asc). | Defaults to null
 `sort_order` | [SortOrder](./models.md#sortorder)| If sort_field is absent, this field is ignored.If absent, this field defaults to ascending. | Defaults to null
 `paging_token` | optional str|  | Defaults to null
 `count` | optional int|  | Defaults to null

Returns [ServicemodelListResponse](./models.md#servicemodellistresponse)

### rollback_service

Rollback a Service

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `service_id` | str|  | Defaults to null
 `rollback_service_model` | [RollbackServiceModel](./models.md#rollbackservicemodel)|  | 

Returns [ServicemodelResponse](./models.md#servicemodelresponse)

### rollout_service

Rollout a service

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `apply_service_model` | [ApplyServiceModel](./models.md#applyservicemodel)|  | 

Returns [ServicemodelResponse](./models.md#servicemodelresponse)

### terminate_service

Terminate a Service

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `service_id` | str|  | Defaults to null

Returns [ServicemodelResponse](./models.md#servicemodelresponse)

## Sso Configs

### upsert_sso_config

Create or update the single sign on (SSO) configuration for your organization.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_sso_config` | [CreateSSOConfig](./models.md#createssoconfig)|  | 

Returns [SsoconfigResponse](./models.md#ssoconfigresponse)

### upsert_test_sso_config

Create or update the test single sign on (SSO) configuration for your organization.

Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 `create_sso_config` | [CreateSSOConfig](./models.md#createssoconfig)|  | 

Returns [SsoconfigResponse](./models.md#ssoconfigresponse)


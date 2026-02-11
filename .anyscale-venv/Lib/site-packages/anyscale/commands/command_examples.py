JOB_SUBMIT_EXAMPLE = """\
# Submit a job without a yaml file, this will inherit the environment from the workspace.
$ anyscale job submit --name my-job --wait -- python main.py
Output
(anyscale +1.0s) Submitting job with config JobConfig(name='my-job', image_uri=None, compute_config=None, env_vars=None, py_modules=None, cloud=None, project=None, ray_version=None, job_queue_config=None).
(anyscale +1.7s) Uploading local dir '.' to cloud storage.
(anyscale +2.6s) Including workspace-managed pip dependencies.
(anyscale +3.2s) Job 'my-job' submitted, ID: 'prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe'.
(anyscale +3.2s) View the job in the UI: https://console.anyscale.com/jobs/prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe
(anyscale +3.2s) Waiting for the job to run. Interrupting this command will not cancel the job.
(anyscale +3.5s) Waiting for job 'prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe' to reach target state SUCCEEDED, currently in state: STARTING
(anyscale +1m19.7s) Job 'prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe' transitioned from STARTING to SUCCEEDED
(anyscale +1m19.7s) Job 'prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe' reached target state, exiting

# Submit a job specified by a yaml file, this will run a job with the specified settings.
# See https://docs.anyscale.com/reference/job-api/#jobconfig for a list of all available fields.
$ cat job.yaml
name: my-job
entrypoint: python main.py
image_uri: anyscale/ray:2.44.1-slim-py312-cu128 # (Optional) Exclusive with `containerfile`.
# compute_config: # (Optional) An inline dictionary can also be provided.
#   head_node:
#     instance_type: m5.8xlarge
working_dir: . # (Optional) Use current working directory "." as the working_dir. Can be any local path or remote .zip file in cloud storage.
requirements: # (Optional) List of requirements files to install. Can also be a path to a requirements.txt.
  - emoji
max_retries: 0 # (Optional) Maximum number of times the job will be retried before being marked failed. Defaults to `1`.
$ anyscale job submit -f job.yaml
Output
(anyscale +1.0s) Submitting job with config JobConfig(name='my-job', image_uri='anyscale/ray:2.44.1-slim-py312-cu128', compute_config=None, env_vars=None, py_modules=None, py_executable=None, cloud=None, project=None, ray_version=None, job_queue_config=None).
(anyscale +4.2s) Uploading local dir '.' to cloud storage.
(anyscale +5.4s) Job 'my-job' submitted, ID: 'prodjob_ugk21r7pt6sbsxfuxn1an82x3e'.
(anyscale +5.4s) View the job in the UI: https://console.anyscale.com/jobs/prodjob_ugk21r7pt6sbsxfuxn1an82x3e
(anyscale +5.4s) Use `--wait` to wait for the job to run and stream logs.
"""

JOB_STATUS_EXAMPLE = """\
$ anyscale job status -n my-job
id: prodjob_6ntzknwk1i9b1uw1zk1gp9dbhe
name: my-job
state: STARTING
runs:
- name: raysubmit_ynxBVGT1SmzndiXL
  state: SUCCEEDED
"""

JOB_TERMINATE_EXAMPLE = """\
$ anyscale job terminate -n my-job
(anyscale +5.4s) Marked job 'my-job' for termination
(anyscale +5.4s) Query the status of the job with `anyscale job status --name my-job`.
"""

JOB_ARCHIVE_EXAMPLE = """\
$ anyscale job archive -n my-job
(anyscale +8.5s) Job prodjob_vzq2pvkzyz3c1jw55kl76h4dk1 is successfully archived.
"""

JOB_DELETE_EXAMPLE = """\
$ anyscale job delete -n my-job
(anyscale +3.2s) Job 'my-job' (ID: prodjob_abc123) has been deleted.
"""

JOB_LOGS_EXAMPLE = """\
$ anyscale job logs -n my-job
2024-08-23 20:31:10,913 INFO job_manager.py:531 -- Runtime env is setting up.
hello world
"""

JOB_WAIT_EXAMPLE = """\
$ anyscale job wait -n my-job
(anyscale +5.7s) Waiting for job 'my-job' to reach target state SUCCEEDED, currently in state: STARTING
(anyscale +1m34.2s) Job 'my-job' transitioned from STARTING to SUCCEEDED
(anyscale +1m34.2s) Job 'my-job' reached target state, exiting
"""

JOB_LIST_EXAMPLE = """\
$ anyscale job list -n my-job
Output
View your Jobs in the UI at https://console.anyscale.com/jobs
JOBS:
NAME    ID                                    COST  PROJECT NAME    CLUSTER NAME                                    CURRENT STATE           CREATOR           ENTRYPOINT
my-job  prodjob_s9x4uzc5jnkt5z53g4tujb3y2e       0  default         cluster_for_prodjob_s9x4uzc5jnkt5z53g4tujb3y2e  SUCCESS                 doc@anyscale.com  python main.py
"""

JOB_TAGS_ADD_EXAMPLE = """\
$ anyscale job tags add --name my-job --tag owner=alice --tag env=staging
"""

JOB_TAGS_REMOVE_EXAMPLE = """\
$ anyscale job tags remove --name my-job --key owner --key env
"""

JOB_TAGS_LIST_EXAMPLE = """\
$ anyscale job tags list --name my-job
"""
JOB_QUEUE_LIST = """\
$ anyscale job-queue list
Output
JOB QUEUES:
             ID                 NAME              CLUSTER ID                      CREATOR ID             MAX CONCURRENCY    IDLE TIMEOUT SEC    CURRENT CLUSTER STATE
jq_h8fcze2qkr8wttuuvapi1hvyuc  queue_3  ses_cjr7uaf1yh2ue5uzvd11p24p4u  usr_we8x7d7u8hq8mj2488ed9x47n6          3                 5000               Terminated
jq_v5bx9z1sd4pbxasxhdms37j4gi  queue_2  ses_k86raeu6k1t6z1bvyejn3vblad  usr_we8x7d7u8hq8mj2488ed9x47n6         10                 5000               Terminated
jq_ni6hk66nt3194msr7hzzj9daun  queue_1  ses_uhb8a9gamtarz68kcurpjh86sa  usr_we8x7d7u8hq8mj2488ed9x47n6         10                 5000               Terminated
"""

JOB_QUEUE_STATUS = """\
$ anyscale job-queue status --id jq_h8fcze2qkr8wttuuvapi1hvyuc
Output
ID                            : jq_h8fcze2qkr8wttuuvapi1hvyuc
USER PROVIDED ID              : queue_3
NAME                          : queue_3
CURRENT JOB QUEUE STATE       : ACTIVE
EXECUTION MODE                : PRIORITY
MAX CONCURRENCY               : 3
IDLE TIMEOUT SEC              : 5000
CREATED AT                    : 2025-04-15 20:40:44
CREATOR ID                    : usr_we8x7d7u8hq8mj2488ed9x47n6
CREATOR EMAIL                 : test@anyscale.com
COMPUTE CONFIG ID             : cpt_8hzsv1t4jvb6kwjhfqbfjw5i6b
CURRENT CLUSTER STATE         : Terminated
CLUSTER ID                    : ses_cjr7uaf1yh2ue5uzvd11p24p4u
PROJECT ID                    : prj_7FWKGPGPaD3Q5mvk9zK2viBD
CLOUD ID                      : cld_kvedZWag2qA8i5BjxUevf5i7
TOTAL JOBS                    : 6
SUCCESSFUL JOBS               : 6
FAILED JOBS                   : 0
ACTIVE JOBS                   : 0
"""

JOB_QUEUE_UPDATE = """\
$ anyscale job-queue update --id jq_h8fcze2qkr8wttuuvapi1hvyuc --max-concurrency 5
Output
ID                            : jq_h8fcze2qkr8wttuuvapi1hvyuc
USER PROVIDED ID              : queue_3
NAME                          : queue_3
CURRENT JOB QUEUE STATE       : ACTIVE
EXECUTION MODE                : PRIORITY
MAX CONCURRENCY               : 5
IDLE TIMEOUT SEC              : 5000
CREATED AT                    : 2025-04-15 20:40:44
CREATOR ID                    : usr_we8x7d7u8hq8mj2488ed9x47n6
CREATOR EMAIL                 : test@anyscale.com
COMPUTE CONFIG ID             : cpt_8hzsv1t4jvb6kwjhfqbfjw5i6b
CURRENT CLUSTER STATE         : Terminated
CLUSTER ID                    : ses_cjr7uaf1yh2ue5uzvd11p24p4u
PROJECT ID                    : prj_7FWKGPGPaD3Q5mvk9zK2viBD
CLOUD ID                      : cld_kvedZWag2qA8i5BjxUevf5i7
TOTAL JOBS                    : 6
SUCCESSFUL JOBS               : 6
FAILED JOBS                   : 0
ACTIVE JOBS                   : 0
"""

JOB_QUEUE_TAGS_ADD_EXAMPLE = """\
$ anyscale job-queue tags add --name my-queue --tag team=data --tag priority=high
"""

JOB_QUEUE_TAGS_REMOVE_EXAMPLE = """\
$ anyscale job-queue tags remove --name my-queue --key team --key priority
"""

JOB_QUEUE_TAGS_LIST_EXAMPLE = """\
$ anyscale job-queue tags list --name my-queue --json
"""

JOB_QUEUE_ARCHIVE_EXAMPLE = """\
$ anyscale job-queue archive --id jq_abc123
Archiving job queue 'jq_abc123'...
Job queue 'jq_abc123' has been archived.
Query the status with `anyscale job-queue status --id jq_abc123`.

$ anyscale job-queue archive --name my-queue --project my-project --cloud my-cloud
Archiving job queue 'my-queue'...
Job queue 'my-queue' has been archived.
Query the status with `anyscale job-queue status --id jq_abc123`.
"""

JOB_QUEUE_TERMINATE_EXAMPLE = """\
$ anyscale job-queue terminate --id jq_abc123
Terminating job queue 'jq_abc123'...
Job queue 'jq_abc123' has been marked for termination.
Query the status with `anyscale job-queue status --id jq_abc123`.

$ anyscale job-queue terminate --name my-queue --project my-project --cloud my-cloud
Terminating job queue 'my-queue'...
Job queue 'my-queue' has been marked for termination.
Query the status with `anyscale job-queue status --id jq_abc123`.
"""

JOB_QUEUE_DELETE_EXAMPLE = """\
$ anyscale job-queue delete --id jq_abc123
Deleting job queue 'jq_abc123'...
Job queue 'jq_abc123' has been deleted.

$ anyscale job-queue delete --name my-queue --project my-project --cloud my-cloud
Deleting job queue 'my-queue'...
Job queue 'my-queue' has been deleted.
"""

SCHEDULE_APPLY_EXAMPLE = """\
$ anyscale schedule apply -n my-schedule -f my-schedule.yaml
(anyscale +0.5s) Applying schedule with config ScheduleConfig(job_config=JobConfig(name='my-schedule', image_uri=None, compute_config=None, env_vars=None, py_modules=None, cloud=None, project=None, ray_version=None, job_queue_config=None), cron_expression='0 0 * * * *', timezone='UTC').
(anyscale +2.3s) Uploading local dir '.' to cloud storage.
(anyscale +3.7s) Including workspace-managed pip dependencies.
(anyscale +4.9s) Schedule 'my-schedule' submitted, ID: 'cronjob_vrjrbwcnfjjid7fsld3sfkn8jz'.

$ cat my-schedule.yaml
timezone: local
cron_expression: 0 0 * * * *
job_config:
    name: my-job
    entrypoint: python main.py
    max_retries: 5
"""

SCHEDULE_LIST_EXAMPLE = """\
$ anyscale schedule list -n my-schedule
Output
+------------------------------------+-------------+---------------+-----------+-------------+------------------+------------+------------------+-----------------------+
| ID                                 | NAME        | DESCRIPTION   | PROJECT   | CRON        | NEXT TRIGGER     | TIMEZONE   | CREATOR          | LATEST EXECUTION ID   |
|------------------------------------+-------------+---------------+-----------+-------------+------------------+------------+------------------+-----------------------|
| cronjob_vrjrbwcnfjjid7fsld3sfkn8jz | my-schedule |               | default   | 0 0 * * * * | 2 hours from now | UTC        | doc@anyscale.com |                       |
+------------------------------------+-------------+---------------+-----------+-------------+------------------+------------+------------------+-----------------------+
"""

SCHEDULE_PAUSE_EXAMPLE = """\
$ anyscale schedule pause -n my-schedule
(anyscale +3.6s) Set schedule 'my-schedule' to state DISABLED
"""

SCHEDULE_RESUME_EXAMPLE = """\
$ anyscale schedule resume -n my-schedule
(anyscale +4.1s) Set schedule 'my-schedule' to state ENABLED
"""

SCHEDULE_STATUS_EXAMPLE = """\
$ anyscale schedule status -n my-schedule
id: cronjob_vrjrbwcnfjjid7fsld3sfkn8jz
name: my-schedule
state: ENABLED
"""

SCHEDULE_RUN_EXAMPLE = """\
$ anyscale schedule run -n my-schedule
(anyscale +2.5s) Triggered job for schedule 'my-schedule'.
"""

SCHEDULE_URL_EXAMPLE = """\
$ anyscale schedule url -n my-schedule
Output
(anyscale +2.3s) View your schedule at https://console.anyscale.com/scheduled-jobs/cronjob_7zj
"""

SCHEDULE_DELETE_EXAMPLE = """\
$ anyscale schedule delete --id cronjob_vrjrbwcnfjjid7fsld3sfkn8jz
(anyscale +2.3s) Schedule 'my-schedule' deleted.

$ anyscale schedule delete --name my-schedule --cloud my-cloud --project my-project
(anyscale +2.5s) Schedule 'my-schedule' deleted.
"""

WORKSPACE_CREATE_EXAMPLE = """\
$ anyscale workspace_v2 create -f config.yml
(anyscale +2.7s) Workspace created successfully id: expwrk_jstjkv15a1vmle2j1t59s4bm35
(anyscale +3.9s) Applied dynamic requirements to workspace id: my-workspace
(anyscale +4.8s) Applied environment variables to workspace id: my-workspace

$ cat config.yml
name: my-workspace
idle_termination_minutes: 10
env_vars:
    HUMPDAY: WEDS
requirements: requirements.txt
"""

WORKSPACE_START_EXAMPLE = """\
$ anyscale workspace_v2 start --name my-workspace
(anyscale +5.8s) Starting workspace 'my-workspace'
"""

WORKSPACE_TERMINATE_EXAMPLE = """\
$ anyscale workspace_v2 terminate --name my-workspace
(anyscale +2.5s) Terminating workspace 'my-workspace'
"""

WORKSPACE_STATUS_EXAMPLE = """\
$ anyscale workspace_v2 status --name my-workspace
(anyscale +2.3s) STARTING
"""

WORKSPACE_WAIT_EXAMPLE = """\
$ anyscale workspace_v2 wait --name my-workspace --state RUNNING
(anyscale +2.5s) Waiting for workspace 'expwrk_jstjkv15a1vmle2j1t59s4bm35' to reach target state RUNNING, currently in state: RUNNING
(anyscale +2.8s) Workspace 'expwrk_jstjkv15a1vmle2j1t59s4bm35' reached target state, exiting
"""

WORKSPACE_SSH_EXAMPLE = """\
$ anyscale workspace_v2 ssh --name my-workspace
Authorized uses only. All activity may be monitored and reported.
Warning: Permanently added '[0.0.0.0]:5020' (ED25519) to the list of known hosts.
(base) ray@ip-10-0-24-60:~/default$
"""

WORKSPACE_RUN_COMMAND_EXAMPLE = """\
$ anyscale workspace_v2 run_command --name my-workspace "echo hello world"
Authorized uses only. All activity may be monitored and reported.
Warning: Permanently added '[0.0.0.0]:5020' (ED25519) to the list of known hosts.
hello world
"""

WORKSPACE_PULL_EXAMPLE = """\
$ anyscale workspace_v2 pull --name my-workspace --local-dir my-local
Warning: Permanently added '54.212.209.253' (ED25519) to the list of known hosts.
Authorized uses only. All activity may be monitored and reported.
Warning: Permanently added '[0.0.0.0]:5020' (ED25519) to the list of known hosts.
receiving incremental file list
created directory my-local
./
my-local/
my-local/main.py

sent 118 bytes  received 141 bytes  172.67 bytes/sec
total size is 0  speedup is 0.00
"""

WORKSPACE_PUSH_EXAMPLE = """\
$ anyscale workspace_v2 push --name my-workspace --local-dir my-local
Warning: Permanently added '52.10.22.124' (ED25519) to the list of known hosts.
Authorized uses only. All activity may be monitored and reported.
Warning: Permanently added '[0.0.0.0]:5020' (ED25519) to the list of known hosts.
sending incremental file list
my-local/
my-local/main.py

sent 188 bytes  received 39 bytes  151.33 bytes/sec
total size is 0  speedup is 0.00
"""

WORKSPACE_UPDATE_EXAMPLE = """\
$ anyscale workspace_v2 update expwrk_6l2ldwbu8299ympgltn725ciak --config-file config.yaml
(anyscale +2.3s) Workspace updated successfully id: expwrk_6l2ldwbu8299ympgltn725ciak
(anyscale +5.6s) Applied dynamic requirements to workspace id: expwrk_6l2ldwbu8299ympgltn725ciak
(anyscale +9.1s) Applied environment variables to workspace id: expwrk_6l2ldwbu8299ympgltn725ciak

$ cat config.yaml
name: my-workspace
idle_termination_minutes: 10
env_vars:
    HUMPDAY: WEDS
requirements: requirements.txt
"""

WORKSPACE_GET_EXAMPLE = """\
$ anyscale workspace_v2 get --name my-workspace
id: my-workspace-id
name: my-workspace
state: RUNNING
"""

WORKSPACE_LIST_EXAMPLE = """\
$ anyscale workspace_v2 list
Listing workspaces with:
• name              = <any>
• project           = <any>
• cloud             = <any>
• created_by_me     = False
• states            = <all>
• tags              = <none>
• include_archived  = False
• sort              = <none>
• mode              = interactive
• per-page limit    = 10
• max-items total   = all

View your Workspaces in the UI at https://console.anyscale.com/workspaces

NAME              STATE     PROJECT           CLOUD        CREATED BY              CREATED AT
my-workspace-1    RUNNING   proj_abc          cld_xyz      user@example.com        2024-01-15 10:30
my-workspace-2    STARTING  proj_abc          cld_xyz      user@example.com        2024-01-16 14:20
Fetched 2 workspaces.
"""

WORKSPACE_TAGS_ADD_EXAMPLE = """\
$ anyscale workspace_v2 tags add --name my-workspace --tag team=ml --tag env=prod
"""

WORKSPACE_TAGS_REMOVE_EXAMPLE = """\
$ anyscale workspace_v2 tags remove --name my-workspace --key team --key env
"""

WORKSPACE_TAGS_LIST_EXAMPLE = """\
$ anyscale workspace_v2 tags list --name my-workspace
Output
Tags:
KEY   VALUE
env   prod
team  ml
"""

MACHINE_POOL_CREATE_EXAMPLE = """\
$ anyscale machine-pool create --name can-testing
Machine pool can-testing has been created successfully (ID mp_8ogdz85mdwxb8a92yo44nn84ox).
"""

MACHINE_POOL_UPDATE_EXAMPLE = """\
$ anyscale machine-pool update --name can-testing --spec-file spec.yaml
Updated machine pool 'can-testing'.
"""

MACHINE_POOL_DESCRIBE_EXAMPLE = """\
$ anyscale machine-pool describe --name can-testing --mode machines
"""

MACHINE_POOL_DELETE_EXAMPLE = """\
$ anyscale machine-pool delete --name can-testing
Deleted machine pool 'can-testing'.
"""

MACHINE_POOL_LIST_EXAMPLE = """\
$ anyscale machine-pool list
MACHINE POOL       ID                             Clouds
can-testing        mp_8ogdz85mdwxb8a92yo44nn84ox
"""

MACHINE_POOL_ATTACH_EXAMPLE = """\
$ anyscale machine-pool attach --name can-testing --cloud my-cloud
Attached machine pool 'can-testing' to cloud 'my-cloud'.
"""

MACHINE_POOL_DETACH_EXAMPLE = """\
$ anyscale machine-pool detach --name can-testing --cloud my-cloud
Detached machine pool 'can-testing' from cloud 'my-cloud'.
"""

RESOURCE_QUOTAS_CREATE_EXAMPLE = """
$ anyscale resource-quota create -n my-resource-quota --cloud my-cloud --project my-project --user-email someone@myorg.com --num-instances 100 --num-cpus 1000 --num-gpus 50 --num-accelerators A10G 10 --num-accelerators A100-80G 0
(anyscale +2.5s) Name: my-resource-quota
Cloud name: my-cloud
Project name: my-project
User email: someone@myorg.com
Number of CPUs: 1000
Number of instances: 100
Number of GPUs: 50
Number of accelerators: {'A10G': 10, 'A100-80G': 0}
(anyscale +2.5s) Resource quota created successfully ID: rsq_abcdef
"""

RESOURCE_QUOTAS_LIST_EXAMPLE = """
$ anyscale resource-quota list --cloud my-cloud
Resource quotas:
ID       NAME               CLOUD ID    PROJECT ID  USER ID     IS ENABLED    CREATED AT    DELETED AT    QUOTA
rsq_123  resource-quota-1   cld_abcdef  prj_abcdef  usr_abcdef  True          09/11/2024                  {'num_accelerators': {'A100-80G': 0, 'A10G': 10},
                                                                                                           'num_cpus': 1000,
                                                                                                           'num_gpus': 50,
                                                                                                           'num_instances': 100}
rsq_456  resource-quota-2   cld_abcdef              usr_abcdef  True          09/10/2024                  {'num_accelerators': {}, 'num_cpus': None, 'num_gpus': None, 'num_instances': 2}
rsq_789  resource-quota-3   cld_abcdef                          False         09/05/2024                  {'num_accelerators': {'A10G': 1},
                                                                                                            'num_cpus': None,
                                                                                                            'num_gpus': None,
                                                                                                            'num_instances': None}
"""

RESOURCE_QUOTAS_ENABLE_EXAMPLE = """
$ anyscale resource-quota enable --id rsq_abcdef
(anyscale +1.2s) Enabled resource quota with ID rsq_abcdef successfully.
"""

RESOURCE_QUOTAS_DISABLE_EXAMPLE = """
$ anyscale resource-quota disable --id rsq_abcdef
(anyscale +1.4s) Disabled resource quota with ID rsq_abcdef successfully.
"""

RESOURCE_QUOTAS_DELETE_EXAMPLE = """
$ anyscale resource-quota delete --id rsq_abcdef
(anyscale +1.0s) Resource quota with ID rsq_abcdef deleted successfully.
"""

COMPUTE_CONFIG_CREATE_EXAMPLE = """
$ anyscale compute-config create -n my-compute-config -f compute_config.yaml
(anyscale +3.7s) Created compute config: 'my-compute-config:1'
(anyscale +3.7s) View the compute config in the UI: 'https://console.anyscale.com/v2/...'

$cat compute_config.yaml
head_node:
  instance_type: m5.8xlarge
worker_nodes:
- instance_type: m5.8xlarge
  min_nodes: 5
  max_nodes: 5
  market_type: ON_DEMAND # (Optional) Defaults to ON_DEMAND
- instance_type: g4dn.xlarge
  min_nodes: 1
  max_nodes: 10
  market_type: PREFER_SPOT # (Optional) Defaults to ON_DEMAND
min_resources: # (Optional) Defaults to no minimum.
  CPU: 1
  GPU: 1
max_resources: # (Optional) Defaults to no maximum.
  CPU: 6
  GPU: 1
"""

COMPUTE_CONFIG_GET_EXAMPLE = """
$ anyscale compute-config get -n my-compute-config
name: my-compute-config:1
id: cpt_buthu4glxj3azv4e287jad3ya3
config:
  cloud: aviary-prod-us-east-1
  head_node:
    instance_type: m5.8xlarge
    resources:
      CPU: 0
      GPU: 0
  worker_nodes:
  - instance_type: m5.8xlarge
    name: m5.8xlarge
    min_nodes: 5
    max_nodes: 5
    market_type: ON_DEMAND
  - instance_type: g4dn.xlarge
    name: g4dn.xlarge
    min_nodes: 1
    max_nodes: 10
    market_type: PREFER_SPOT
  min_resources:
    CPU: 1
    GPU: 1
  max_resources:
    CPU: 6
    GPU: 1
  enable_cross_zone_scaling: false
  flags: {}
"""

COMPUTE_CONFIG_ARCHIVE_EXAMPLE = """
$ anyscale compute-config archive -n my-compute-config
(anyscale +2.3s) Compute config is successfully archived.
"""

COMPUTE_CONFIG_LIST_EXAMPLE = """\
$ anyscale compute-config list --max-items 5
Compute configs:
ID                              NAME                CLOUD        LAST MODIFIED AT
cpt_abc123                      my-compute-config   my-cloud     01/15/2025, 10:30:00
cpt_def456                      prod-config         prod-cloud   01/14/2025, 14:22:00

$ anyscale compute-config list --cloud-name my-cloud --include-shared --json
{
  "results": [
    {
      "id": "cpt_abc123",
      "name": "my-compute-config",
      "cloud_id": "cld_xyz",
      "version": 1,
      "created_at": "2025-01-15T10:30:00",
      "last_modified_at": "2025-01-15T10:30:00",
      "url": "https://console.anyscale.com/configurations/cluster-computes/cpt_abc123"
    }
  ],
  "metadata": {
    "count": 1,
    "next_token": null
  }
}
"""

IMAGE_BUILD_EXAMPLE = """
$ anyscale image build -f my.Dockerfile -n my-image --ray-version 2.21.0
(anyscale +2.8s) Building image. View it in the UI: https://console.anyscale.com/v2/...
(anyscale +1m53.0s) Waiting for image build to complete. Elapsed time: 102 seconds.
(anyscale +1m53.0s) Image build succeeded.
Image built successfully with URI: anyscale/image/my-image:1

$ cat my.Dockerfile
FROM anyscale/ray:2.21.0-py39
RUN pip install --no-cache-dir pandas
"""

IMAGE_GET_EXAMPLE = """
$ anyscale image get -n my-image
uri: anyscale/image/my-image:1
status: SUCCEEDED
ray_version: 2.21.0
"""

IMAGE_LIST_EXAMPLE = """
$ anyscale image list --no-interactive --max-items 2
NAME         LATEST VERSION  LATEST URI                    CREATED BY           CREATED AT
my-image     3               anyscale/image/my-image:3     dhyey@anyscale.com   2024-12-01 18:42
workspace    1               anyscale/image/workspace:1    joshlee@anyscale.com 2024-11-09 09:15
"""

IMAGE_REGISTER_EXAMPLE = """
$ anyscale image register --image-uri docker.io/myrepo/image:v2 --name mycoolimage --ray-version 2.30.0
Image registered successfully with URI: anyscale/image/mycoolimage:1
"""

IMAGE_ARCHIVE_EXAMPLE = """
$ anyscale image archive --name my-old-image
(anyscale +0.5s) Successfully archived image.
Image 'my-old-image' archived successfully.
"""

AGGREGATED_INSTANCE_USAGE_DOWNLOAD_CSV_EXAMPLE = """\
$ anyscale aggregated-instance-usage download-csv --start-date 2024-10-01 --end-date 2024-10-03
(anyscale +0.5s) Downloading aggregated instance usage CSV...
Download complete! File saved to 'aggregated_instance_usage_2024-10-01_2024-10-03.zip'
0:00:01 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.4 kB / 79.4 kB Downloading aggregated instance usage CSV
(anyscale +2.2s) Downloaded CSV to aggregated_instance_usage_2024-10-01_2024-10-03.zip
"""

USER_BATCH_CREATE_EXAMPLE = """\
$ anyscale user batch-create --users-file users.yaml
(anyscale +0.5s) Creating users...
(anyscale +0.8s) 2 users created.

$ cat users_file.yaml
create_users:
  - name: name1
    email: test1@anyscale.com
    password: ''
    is_sso_user: false
    lastname: lastname1
    title: title1
  - name: name2
    email: test2@anyscale.com
    password: ''
    is_sso_user: false
    lastname: lastname2
    title: title2
"""

USER_LIST_EXAMPLE = """\
$ anyscale user list --page-size 5 --no-interactive
Output
USERS:
EMAIL                ID         NAME         ROLE         CREATED AT
owner@anyscale.com   usr_owner  Owner Name   owner        2024-01-01 12:00
service@anyscale.com           Service Bot  collaborator 2024-02-01 08:30
"""

USER_GET_EXAMPLE = """\
$ anyscale user get --email owner@anyscale.com --json
Output
{
  "name": "Owner Name",
  "email": "owner@anyscale.com",
  "created_at": "2024-01-01T12:00:00+00:00",
  "permission_level": "owner",
  "user_id": "usr_owner"
}
"""

ORGANIZATION_INVITATION_CREATE_EXAMPLE = """\
$ anyscale organization-invitation create --emails test1@anyscale.com,test2@anyscale.com
(anyscale +0.5s) Creating organization invitations...
(anyscale +1.7s) Organization invitations sent to: test1@anyscale.com, test2@anyscale.com
"""

ORGANIZATION_INVITATION_LIST_EXAMPLE = """\
$ anyscale organization-invitation list
ID             Email              Created At           Expires At
-------------  -----------------  -------------------  -------------------
orginv_abcedf  test@anyscale.com  11/25/2024 10:24 PM  12/02/2024 10:24 PM
"""

ORGANIZATION_INVITATION_DELETE_EXAMPLE = """\
$ anyscale organization-invitation delete --email test@anyscale.com
(anyscale +0.6s) Organization invitation for test@anyscale.com deleted.
"""


PROJECT_ADD_COLLABORATORS_EXAMPLE = """\
$ anyscale project add-collaborators --cloud cloud_name --project project_name --users-file collaborators.yaml
(anyscale +1.3s) Successfully added 3 collaborators to project project_name.
$ cat collaborators.yaml
collaborators:
  - email: "test1@anyscale.com"
    permission_level: "collaborator"
  - email: "test2@anyscale.com"
    permission_level: "readonly"
  - email: "test3@anyscale.com"
    permission_level: "owner"
"""


PROJECT_GET_EXAMPLE = """\
$ anyscale project get --id my-project-id
"""


PROJECT_LIST_EXAMPLE = """\
$ anyscale project list --include-defaults --non-interactive --max-items 2
"""


PROJECT_CREATE_EXAMPLE = """\
$ anyscale project create --name "my-project" --cloud "my-cloud-id"
"""


PROJECT_DELETE_EXAMPLE = """\
$ anyscale project delete --id project-id
"""


PROJECT_GET_DEFAULT_EXAMPLE = """\
$ anyscale project get-default --cloud my-cloud-id --json
"""


CLOUD_SETUP_K8S_AWS_EXAMPLE = """\
# Set up a Kubernetes cloud on AWS EKS
$ anyscale cloud setup --provider aws --region us-west-2 --name my-k8s-cloud \\
    --stack k8s --cluster-name my-eks-cluster --namespace anyscale-operator \\
    --debug --yes
Output
(anyscale +1.0s) Setting up Kubernetes cloud 'my-k8s-cloud' on AWS
(anyscale +1.5s) Required CLI tools are installed (kubectl, helm, aws)
(anyscale +2.3s) Using namespace: anyscale-operator
(anyscale +5.2s) Cluster discovered: arn:aws:eks:us-west-2:123456789012:cluster/my-eks-cluster
(anyscale +8.4s) Setting up AWS infrastructure...
(anyscale +1m12.3s) CloudFormation stack created
(anyscale +1m15.7s) Cloud registered with ID: cld_abc123
(anyscale +1m18.2s) Adding Anyscale Helm repository...
(anyscale +1m20.5s) Installing Anyscale operator...
(anyscale +1m25.8s) Generated Helm values file: anyscale-helm-values-aws-anyscale-operator-20241015_175602.yaml
(anyscale +2m45.3s) Helm installation completed successfully
(anyscale +2m45.4s) Kubernetes cloud 'my-k8s-cloud' setup completed successfully!
"""

CLOUD_SETUP_K8S_GCP_EXAMPLE = """\
# Set up a Kubernetes cloud on GCP GKE
$ anyscale cloud setup --provider gcp --region us-central1 --name my-gke-cloud \\
    --stack k8s --cluster-name my-gke-cluster --namespace anyscale-operator \\
    --project-id my-project-123 --yes
Output
(anyscale +1.0s) Setting up Kubernetes cloud 'my-gke-cloud' on GCP
(anyscale +1.5s) Required CLI tools are installed (kubectl, helm, gcloud, gsutil)
(anyscale +2.3s) Using namespace: anyscale-operator
(anyscale +5.8s) Cluster discovered: gke_my-project-123_us-central1_my-gke-cluster
(anyscale +9.2s) Setting up GCP infrastructure...
(anyscale +15.4s) Created GCS bucket: anyscale-k8s-my-gke-cloud-a1b2c3d4
(anyscale +18.7s) Created service account: anyscale-operator-e5f6g7h8@my-project-123.iam.gserviceaccount.com
(anyscale +25.3s) GCP resources created successfully
(anyscale +28.1s) Cloud registered with ID: cld_xyz789
(anyscale +30.5s) Installing Anyscale operator...
(anyscale +35.8s) Generated Helm values file: anyscale-helm-values-gcp-anyscale-operator-20241015_180215.yaml
(anyscale +1m55.2s) Helm installation completed successfully
(anyscale +1m55.3s) Kubernetes cloud 'my-gke-cloud' setup completed successfully!
"""

CLOUD_SETUP_K8S_CUSTOM_VALUES_EXAMPLE = """\
# Set up a Kubernetes cloud with a custom Helm values file path
$ anyscale cloud setup --provider aws --region us-west-2 --name my-k8s-cloud \\
    --stack k8s --cluster-name my-eks-cluster \\
    --values-file /path/to/custom-values.yaml --yes
Output
(anyscale +1.0s) Setting up Kubernetes cloud 'my-k8s-cloud' on AWS
...
(anyscale +1m25.8s) Generated Helm values file: /path/to/custom-values.yaml
...
"""

CLOUD_ADD_COLLABORATORS_EXAMPLE = """\
$ anyscale cloud add-collaborators --cloud cloud_name --users-file collaborators.yaml
(anyscale +1.3s) Successfully added 2 collaborators to cloud cloud_name.
$ cat collaborators.yaml
collaborators:
  - email: "test1@anyscale.com"
    permission_level: "collaborator"
  - email: "test2@anyscale.com"
    permission_level: "readonly"
"""


CLOUD_RESOURCE_CREATE_EXAMPLE = """\
$ anyscale cloud resource create --cloud my-cloud -f new-cloud-resource.yaml
Successfully created cloud resource my-new-resource in cloud my-cloud.

$ cat new-cloud-resource.yaml
name: my-new-resource
provider: AWS
compute_stack: VM
region: us-west-2
networking_mode: PUBLIC
object_storage:
  bucket_name: s3://my-bucket
file_storage:
  file_storage_id: fs-123
aws_config:
  vpc_id: vpc-123
  subnet_ids:
  - subnet-123
  security_group_ids:
  - sg-123
  anyscale_iam_role_id: arn:aws:iam::123456789012:role/anyscale-role-123
  cluster_iam_role_id: arn:aws:iam::123456789012:role/cluster-role-123
  memorydb_cluster_name: my-memorydb-cluster
"""

CLOUD_RESOURCE_DELETE_EXAMPLE = """\
$ anyscale cloud resource delete --cloud my-cloud --resource my-resource
Output
Please confirm that you would like to remove resource my-resource from cloud my-cloud. [y/N]: y
(anyscale +3.5s) Successfully removed resource my-resource from cloud my-cloud!
"""

CLOUD_GET_CLOUD_EXAMPLE = """\
$ anyscale cloud get --name my-cloud
name: my-cloud
id: cld_123
resources:
- cloud_resource_id: cldrsrc_123
  name: vm-aws-us-west-2
  provider: AWS
  compute_stack: VM
  region: us-west-2
  networking_mode: PUBLIC
  object_storage:
    bucket_name: s3://my-bucket
  file_storage:
    file_storage_id: fs-123
  aws_config:
    vpc_id: vpc-123
    subnet_ids:
    - subnet-123
    security_group_ids:
    - sg-123
    anyscale_iam_role_id: arn:aws:iam::123456789012:role/anyscale-role-123
    cluster_iam_role_id: arn:aws:iam::123456789012:role/cluster-role-123
    memorydb_cluster_name: my-memorydb-cluster
"""

CLOUD_STATUS_EXAMPLE = """\
$ anyscale cloud status --name my-cloud
name: my-cloud
id: cld_123
created_at: 2022-10-18 05:12:13.335803+00:00
is_default: true
resources:
- cloud_resource_id: cldrsrc_123
  name: vm-aws-us-west-2
  provider: AWS
  compute_stack: VM
  region: us-west-2
  networking_mode: PUBLIC
  operator_status: HEALTHY
  operator_status_details: All components running
  object_storage:
    bucket_name: s3://my-bucket
  aws_config:
    vpc_id: vpc-123
    subnet_ids:
    - subnet-123
"""

CLOUD_GET_DEFAULT_CLOUD_EXAMPLE = """\
$ anyscale cloud get-default
name: anyscale_v2_default_cloud
id: cld_abc
provider: AWS
region: us-west-2
created_at: 2022-10-18 05:12:13.335803+00:00
is_default: true
compute_stack: VM
"""

CLOUD_CONFIG_UPDATE_EXAMPLE = """\
$ anyscale cloud config update --cloud-id cloud_id --enable-log-ingestion --enable-system-cluster
--enable-log-ingestion is specified. [...] If you are sure you want to enable this feature, please type "consent": consent
Output
(anyscale +7.3s) Successfully updated log ingestion configuration for cloud, cloud_id to True
--enable-system-cluster is specified. [...] Are you sure you want to enable system cluster? [y/N]: y
Output
(anyscale +11.4s) Successfully enabled system cluster for cloud cloud_id

$ anyscale cloud config update --cloud-id cloud_id --spec-file iam.yaml
Output
(anyscale +2.1s) Successfully updated cloud configuration for cloud my-cloud (resource: cldrsrc_xyz123)

$ anyscale cloud config update --cloud-id cloud_id --resource shared-usw2 --spec-file iam.yaml
Output
(anyscale +2.1s) Successfully updated cloud configuration for cloud my-cloud (resource: cldrsrc_abc456)

$ anyscale cloud config update --cloud-id cloud_id --cloud-resource-id cldrsrc_xyz123 --spec-file iam.yaml
Output
(anyscale +2.1s) Successfully updated cloud configuration for cloud my-cloud (resource: cldrsrc_xyz123)
"""

CLOUD_TERMINATE_SYSTEM_CLUSTER_EXAMPLE = """\
$ anyscale cloud terminate-system-cluster --cloud-id cloud_id --wait
(anyscale +1.3s) Waiting for system cluster termination............
(anyscale +1m22.9s) System cluster for cloud 'cloud_id' is Terminated.
"""

SERVICE_ARCHIVE_EXAMPLE = """\
$ anyscale service archive --name my_service
"""

SERVICE_DELETE_EXAMPLE = """\
$ anyscale service delete --name my_service
"""

SERVICE_LIST_EXAMPLE = """\
$ anyscale service list --state running --sort -created_at
"""

SERVICE_TAGS_ADD_EXAMPLE = """\
$ anyscale service tags add --name my-service --tag team=platform --tag env=prod
"""

SERVICE_TAGS_REMOVE_EXAMPLE = """\
$ anyscale service tags remove --name my-service --key team --key env
"""

SERVICE_TAGS_LIST_EXAMPLE = """\
$ anyscale service tags list --name my-service --json
"""

# User Group Examples
USER_GROUP_LIST_EXAMPLE = """\
$ anyscale user-group list --max-items 50
ID            Name
------------  ----------------
ug_abc123     Engineering
ug_def456     Data Science
ug_ghi789     Platform Team
"""

USER_GROUP_GET_EXAMPLE = """\
$ anyscale user-group get --id ug_abc123
ID               ug_abc123
Name             Engineering
Organization ID  org_abc123
Created At       2025-01-15 10:30:00 UTC
Updated At       2025-01-15 10:30:00 UTC
"""

USER_GROUP_MEMBERSHIP_LIST_EXAMPLE = """\
$ anyscale user-group membership list
{
  "Engineering": [
    "alice@example.com",
    "charlie@example.com"
  ],
  "Data Science": [
    "bob@example.com"
  ]
}

# Save output to a file
$ anyscale user-group membership list --output memberships.json
"""

# Policy Examples
POLICY_SET_EXAMPLE = """\
# Set policy for a cloud
$ anyscale policy set --resource-type cloud --resource-id cld_abc123 -f policy.yaml
(anyscale +0.5s) Setting policy for cloud cld_abc123...
(anyscale +1.2s) Policy for cloud cld_abc123 has been updated.

# Set policy for your organization (--resource-id is not allowed)
$ anyscale policy set --resource-type organization -f org_policy.yaml
(anyscale +0.5s) Setting policy for organization your organization...
(anyscale +1.2s) Policy for organization your organization has been updated.

$ cat policy.yaml
bindings:
  - role_name: collaborator
    principals:
      - ug_abc123
  - role_name: readonly
    principals:
      - ug_def456
      - ug_ghi789
"""

POLICY_GET_EXAMPLE = """\
# Get policy for a cloud
$ anyscale policy get --resource-type cloud --resource-id cld_abc123
(anyscale +0.5s) Policy for cloud cld_abc123:
Role      Principal (User Group ID)  Process Status
--------  -------------------------  --------------
collaborator  ug_abc123              success
readonly  ug_def456                  success
readonly  ug_ghi789                  success

# Get policy for your organization (--resource-id is not allowed)
$ anyscale policy get --resource-type organization
(anyscale +0.5s) Policy for organization your organization:
Role      Principal (User Group ID)  Process Status
--------  -------------------------  --------------
owner     ug_admins                  success
collaborator  ug_developers          success
"""

POLICY_LIST_EXAMPLE = """\
$ anyscale policy list --resource-type cloud
(anyscale +0.6s) cloud: cld_abc123
Role      Principal (User Group ID)  Process Status
--------  -------------------------  --------------
collaborator  ug_abc123              success
readonly  ug_def456                  success

(anyscale +0.6s) cloud: cld_xyz789
Role      Principal (User Group ID)  Process Status
--------  -------------------------  --------------
collaborator  ug_ghi789              pending
"""

# SCIM Examples
SCIM_ENFORCE_GROUP_PERMISSIONS_EXAMPLE = """\
# Preview permission changes before applying (dry-run mode)
$ anyscale scim enforce-groups --dry-run
(anyscale +0.5s) Running in dry-run mode. Analyzing permission changes...

=== Permission Changes Preview ===

user1@example.com:
  - clouds:
      - prod-cloud: collaborator -> readonly
      - staging-cloud: owner -> (removed)
  - projects:
      - proj-1: collaborator -> readonly
  - organization: owner -> collaborator

user2@example.com:
  - clouds:
      - dev-cloud: collaborator -> (removed)

--- Users to be removed (not in any active user group) ---
  - orphan-user@example.com

(No changes were applied. Remove --dry-run to apply changes.)

# Apply the changes (live mode)
$ anyscale scim enforce-groups
(anyscale +0.5s) Analyzing permission changes...

=== Permission Changes Preview ===

user1@example.com:
  - clouds:
      - prod-cloud: collaborator -> readonly
      - staging-cloud: owner -> (removed)
  - projects:
      - proj-1: collaborator -> readonly
  - organization: owner -> collaborator

user2@example.com:
  - clouds:
      - dev-cloud: collaborator -> (removed)

--- Users to be removed (not in any active user group) ---
  - orphan-user@example.com

╭─────────────────── ⚠️  Confirmation Required ───────────────────╮
│ WARNING: This is a destructive operation that cannot be undone. │
│                                                                  │
│ All role bindings on users will be removed.                      │
│ Role bindings on user groups and service accounts are unchanged. │
╰──────────────────────────────────────────────────────────────────╯
Do you want to proceed? [y/N]: y
(anyscale +1.5s) Starting SCIM permission migration...

=== Applied Permission Changes ===

user1@example.com:
  - clouds:
      - prod-cloud: collaborator -> readonly
      - staging-cloud: owner -> (removed)
  - projects:
      - proj-1: collaborator -> readonly
  - organization: owner -> collaborator

user2@example.com:
  - clouds:
      - dev-cloud: collaborator -> (removed)

--- Users to be removed (not in any active user group) ---
  - orphan-user@example.com

(anyscale +2.0s) SCIM permission migration completed successfully.
"""

SCIM_LIST_USERS_PERMISSIONS_EXAMPLE = """\
# List all users' effective cloud & project permissions, plus org owners
$ anyscale scim list-users-permissions
(anyscale +0.6s) Listing users and their effective permissions...
{
  "organization_id": "org_p72",
  "org_owners": [
    {
      "user_email": "admin1@p72.com",
      "user_id": "usr_aaa"
    }
  ],
  "users": [
    {
      "clouds": [
        {
          "cloud_id": "cld_111",
          "cloud_name": "prod-cloud",
          "role": "collaborator",
          "projects": [
            {
              "project_id": "prj_111",
              "project_name": "prod-project",
              "role": "readonly"
            }
          ]
        }
      ],
      "user_email": "alice@p72.com",
      "user_id": "usr_abc"
    }
  ]
}
"""

USER_LIST_PERMISSIONS_EXAMPLE = """\
# List all users' effective cloud & project permissions
$ anyscale user list-permissions
(anyscale +0.6s) Listing users and their effective permissions...
{
  "organization_id": "org_p72",
  "org_owners": [
    {
      "user_email": "admin1@p72.com",
      "user_id": "usr_aaa"
    }
  ],
  "users": [
    {
      "clouds": [
        {
          "cloud_id": "cld_111",
          "cloud_name": "prod-cloud",
          "role": "collaborator",
          "projects": [
            {
              "project_id": "prj_111",
              "project_name": "prod-project",
              "role": "readonly"
            }
          ]
        }
      ],
      "user_email": "alice@p72.com",
      "user_id": "usr_abc"
    }
  ]
}

# List permissions for a specific user
$ anyscale user list-permissions --user-id usr_abc
(anyscale +0.4s) Listing users and their effective permissions...
{
  "organization_id": "org_p72",
  "org_owners": [
    {
      "user_email": "admin1@p72.com",
      "user_id": "usr_aaa"
    }
  ],
  "users": [
    {
      "clouds": [
        {
          "cloud_id": "cld_111",
          "cloud_name": "prod-cloud",
          "role": "collaborator"
        }
      ],
      "user_email": "alice@p72.com",
      "user_id": "usr_abc"
    }
  ]
}
"""

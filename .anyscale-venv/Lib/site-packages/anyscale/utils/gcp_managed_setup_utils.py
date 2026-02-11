import copy
import os
from string import Template
import time
from typing import Any, Dict, List, Optional

from click import ClickException
from google.api_core.exceptions import BadRequest, Forbidden, NotFound, PermissionDenied
from google.cloud.compute_v1.types import (
    FirewallPolicyAssociation,
    FirewallPolicyRule,
    Operation,
)
from google.iam.v1.iam_policy_pb2 import SetIamPolicyRequest
from google.iam.v1.policy_pb2 import Binding
from googleapiclient.errors import HttpError
import yaml

from anyscale.anyscale_pydantic import BaseModel
from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models import CloudAnalyticsEventCloudResource
import anyscale.conf
import anyscale.shared_anyscale_utils.conf as shared_anyscale_conf
from anyscale.util import (
    confirm,
    GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS_LONG,
    SharedStorageType,
)
from anyscale.utils.gcp_utils import GCP_REQUIRED_APIS, GoogleCloudClientFactory


GCP_DEPLOYMENT_PREVIEW_TIMEOUT_SECONDS = 300  # 5 minutes


GCP_MEMORYSTORE_RESOURCE_CONFIG_TEMPLATE = """
- name: redis-${CLOUD_ID}
  type: gcp-types/redis-v1:projects.locations.instances
  properties:
    parent: projects/${PROJECT_ID}/locations/${REGION}
    instanceId: redis-${CLOUD_ID}
    authorizedNetwork: projects/${PROJECT_ID}/global/networks/$$(ref.vpc-${CLOUD_ID}.name)
    memorySizeGb: 5
    replicaCount: 1
    readReplicasMode: READ_REPLICAS_ENABLED
    tier: STANDARD_HA
    redisVersion: REDIS_7_0
    displayName: redis-${CLOUD_ID}
    redisConfigs:
        maxmemory-policy: allkeys-lru
  metadata:
   dependsOn:
     - subnet-${CLOUD_ID}"""


GCP_FILESTORE_RESOURCE_CONFIG_TEMPLATE = """
- name: filestore-${CLOUD_ID}
  type: gcp-types/file-v1beta1:projects.locations.instances
  properties:
    instanceId: filestore-${CLOUD_ID}
    parent: projects/${PROJECT_ID}/locations/${REGION}-${ZONE}
    tier: STANDARD
    networks:
      - network: projects/${PROJECT_ID}/global/networks/$$(ref.vpc-${CLOUD_ID}.name)
    fileShares:
      - name: anyscale_vol
        capacityGb: ${FILESHARE_CAPACITY_GB}
    labels:
      anyscale-cloud-id: ${CLOUD_ID_UNDERSCORE}
  metadata:
    dependsOn:
      - vpc-${CLOUD_ID}"""


GCP_FILESTORE_RESOURCE_CONFIG_TEMPLATE_OA = """
- name: filestore-${CLOUD_ID}
  type: gcp-types/file-v1beta1:projects.locations.instances
  properties:
    instanceId: filestore-${CLOUD_ID}
    parent: projects/${PROJECT_ID}/locations/${REGION}-${ZONE}
    tier: STANDARD
    networks:
      - network: projects/${PROJECT_ID}/global/networks/${VPC_NAME}
    fileShares:
      - name: anyscale_vol
        capacityGb: ${FILESHARE_CAPACITY_GB}
    labels:
      anyscale-cloud-id: ${CLOUD_ID_UNDERSCORE}
- name: allow-filestore-${CLOUD_ID}
  type: compute.v1.firewall
  properties:
    direction: EGRESS
    network: projects/${PROJECT_ID}/global/networks/${VPC_NAME}
    allowed:
    - IPProtocol: all
    targetServiceAccounts:
    - $$(ref.${CLOUD_ID}.email)
    destinationRanges:
    - $$(ref.filestore-${CLOUD_ID}.networks[0].reservedIpRange)
    priority: 1000
  metadata:
    dependsOn:
      - ${CLOUD_ID}
      - filestore-${CLOUD_ID}"""


class GCPDeployment(BaseModel):
    deployment_name: str
    fingerprint: str
    config_content: str


def get_project_number(factory: GoogleCloudClientFactory, project_id: str):
    project_client = factory.resourcemanager_v3.ProjectsClient()
    try:
        project = project_client.get_project(name=f"projects/{project_id}")
        return project.name  # format: "/projects/{project_number}"
    except (NotFound, PermissionDenied, Forbidden) as e:
        raise ClickException(
            f"Error occurred when trying to access the project {project_id}: {e}"
        )


def append_project_iam_policy(
    factory: GoogleCloudClientFactory, project_id: str, role: str, member: str
):
    policy_client = factory.resourcemanager_v3.ProjectsClient()
    try:
        policy = policy_client.get_iam_policy(resource=f"projects/{project_id}")
        if policy.bindings is None:
            policy.bindings = []

        policy.bindings.append(Binding(role=role, members=[member]))
        updated_policy = policy_client.set_iam_policy(
            SetIamPolicyRequest(policy=policy, resource=f"projects/{project_id}")
        )
        return updated_policy
    except (BadRequest, NotFound, PermissionDenied, Forbidden) as e:
        raise ClickException(f"Failed to set IAM policy for project {project_id}: {e}")


def enable_project_apis(
    factory: GoogleCloudClientFactory,
    project_id: str,
    logger: CloudSetupLogger,
    enable_head_node_fault_tolerance: bool = False,
):
    """ Automatically enable APIs that Anyscale needs.
    """
    try:
        service_usage_client = factory.build("serviceusage", "v1")
        api_list = copy.deepcopy(GCP_REQUIRED_APIS)
        if enable_head_node_fault_tolerance:
            api_list.append("redis.googleapis.com")

        response = (
            service_usage_client.services()
            .batchEnable(
                parent="projects/" + project_id, body={"serviceIds": api_list},
            )
            .execute()
        )

        wait_for_operation_completion(
            service_usage_client,
            {"name": response["name"]},
            f"Enable APIs for project {project_id}",
        )
    except HttpError as e:
        logger.log_resource_exception(CloudAnalyticsEventCloudResource.GCP_PROJECT, e)
        raise ClickException(
            f"Failed to enable APIs for project {project_id}: {e}. "
            f"Please make sure the service usage API is enabled: https://console.cloud.google.com/marketplace/product/google/serviceusage.googleapis.com?project={project_id}"
        )


def get_workload_identity_pool(
    factory: GoogleCloudClientFactory, project_id: str, pool_id: str,
):
    workload_identity_pool_client = (
        factory.build("iam", "v1").projects().locations().workloadIdentityPools()
    )

    try:
        workload_identity_pool_client.get(
            name=f"projects/{project_id}/locations/global/workloadIdentityPools/{pool_id}"
        ).execute()
        return pool_id
    except HttpError as e:
        if e.status_code == 404:
            # workload identity pool not found
            return None
        else:
            raise ClickException(f"Failed to get Workload Identity Provider Pool. {e}")


def get_anyscale_gcp_access_service_acount(
    factory: GoogleCloudClientFactory, anyscale_access_service_account: str,
):
    service_account_client = factory.build("iam", "v1").projects().serviceAccounts()

    try:
        service_account_client.get(
            name=f"projects/-/serviceAccounts/{anyscale_access_service_account}"
        ).execute()
        return anyscale_access_service_account
    except HttpError as e:
        if e.status_code == 404:
            # service account not found
            return None
        else:
            raise ClickException(f"Failed to get service account: {e}")


def get_deployment_resources(
    factory: GoogleCloudClientFactory,
    deployment_name: str,
    project_id: str,
    anyscale_access_service_account_name: str,
) -> Dict[str, str]:
    """
    Get the resources in a deployment.
    """
    gcp_deployment = get_deployment_config(factory, project_id, deployment_name)
    config = yaml.safe_load(gcp_deployment.config_content)

    # TODO (congding): use a pydantic model instead of dict
    cloud_resources = {}
    for resource in config["resources"]:
        resource_props = resource["properties"]
        if resource["name"] == anyscale_access_service_account_name:
            # we should just keep the cluster service account and skip this
            continue
        if "iamMemberBinding" in resource["type"]:
            # skip IAM member binding
            continue
        if resource["type"] == "gcp-types/file-v1beta1:projects.locations.instances":
            # get filestore location and instance
            cloud_resources["filestore_instance"] = resource_props["instanceId"]
            cloud_resources["filestore_location"] = resource_props["parent"].split("/")[
                -1
            ]
        if resource["type"] == "gcp-types/redis-v1:projects.locations.instances":
            # get redis instance
            cloud_resources[
                "memorystore_name"
            ] = f'{resource_props["parent"]}/instances/{resource_props["instanceId"]}'

        cloud_resources[resource["type"]] = resource["name"]

    return cloud_resources


def create_workload_identity_pool(
    factory: GoogleCloudClientFactory,
    project_id: str,
    pool_id: str,
    logger: CloudSetupLogger,
    display_name: str = "a workload identity pool",
    description: str = "a workload identity pool",
):
    """ Create a GCP Workload Identity Provider Pool. The functionality is not
    currently supported by GCP Deployment Manager.
    """
    workload_identity_pool_client = (
        factory.build("iam", "v1").projects().locations().workloadIdentityPools()
    )

    parent = f"projects/{project_id}/locations/global"
    pool = {"displayName": display_name, "description": description}

    try:
        create_workload_identity_pool_operation = workload_identity_pool_client.create(
            parent=parent, workloadIdentityPoolId=pool_id, body=pool
        ).execute()

        wait_for_operation_completion(
            workload_identity_pool_client,
            {"name": create_workload_identity_pool_operation["name"]},
            description="creating workload identity provider pool",
        )

        workload_identity_pool = create_workload_identity_pool_operation["name"].split(
            "/operation"
        )[0]
        logger.info(f"Workload Identity Pool created: {workload_identity_pool}")
        return workload_identity_pool
    except HttpError as e:
        if e.status_code == 409:
            logger.error(
                f"Provider Pool {pool_id} already exists in project {project_id}."
            )
        else:
            logger.error(
                f"Error occurred when trying to build Workload Identity Provider Pool. Detailed: {e}"
            )
        raise ClickException("Failed to create Workload Identity Provider Pool. ")


def create_anyscale_aws_provider(
    factory: GoogleCloudClientFactory,
    organization_id: str,
    pool_id: str,
    provider_id: str,
    aws_account_id: str,
    display_name: str,
    logger: CloudSetupLogger,
):
    """ Create a GCP Workload Identity Provider for Anyscale cross account access.
    The functionality is notcurrently supported by GCP Deployment Manager.
    """
    provider_client = (
        factory.build("iam", "v1")
        .projects()
        .locations()
        .workloadIdentityPools()
        .providers()
    )

    parent = pool_id
    provider = {
        "aws": {"accountId": aws_account_id},
        "name": provider_id,
        "displayName": display_name,
        "description": "provider for Anyscale access",
        "attributeMapping": {
            "attribute.aws_role": "assertion.arn.contains('assumed-role') ? assertion.arn.extract('{account_arn}assumed-role/') + 'assumed-role/' + assertion.arn.extract('assumed-role/{role_name}/') : assertion.arn",
            "google.subject": "assertion.arn",
            "attribute.arn": "assertion.arn",
        },
        "attributeCondition": f"google.subject.startsWith('arn:aws:sts::{aws_account_id}:assumed-role/gcp_if_{organization_id}')",
    }

    try:
        response = provider_client.create(
            parent=parent, workloadIdentityPoolProviderId=provider_id, body=provider
        ).execute()
        wait_for_operation_completion(
            provider_client,
            {"name": response["name"]},
            description="creating workload identity provider",
        )
        workload_identity_provider = response["name"].split("/operation")[0]
        logger.info(f"Anyscale provider created: {workload_identity_provider}")
        return workload_identity_provider
    except HttpError as e:
        if e.status_code == 409:
            logger.error(f"Provider {provider_id} already exists in pool {parent}.")
        else:
            logger.error(
                f"Error occurred when trying to build Workload Identity Provider Pool. Detailed: {e}"
            )
        raise ClickException("Failed to create Anyscale AWS Workload Identity Provider")


def generate_deployment_manager_config(  # noqa: PLR0913
    region: str,
    project_id: str,
    cloud_id: str,
    anyscale_access_service_account_name: str,
    workload_identity_pool_name: str,
    anyscale_aws_account: str,
    organization_id: str,
    enable_head_node_fault_tolerance: bool,
    *,
    vpc_name: Optional[str] = None,
    subnet_cidr: str = "10.0.0.0/20",
    zone: str = "b",  # some regions like us-east1 or europe-west1 don't have zone "a"
    fileshare_capacity_gb: int = 1024,
    control_plane_service_account_email: Optional[str] = None,
    shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
) -> str:
    """
    Generate the deployment manager config for Anyscale cloud setup.
    """
    file_path = (
        os.path.join(anyscale.conf.ROOT_DIR_PATH, "anyscale-cloud-setup-gcp.yaml")
        if not vpc_name
        else os.path.join(
            anyscale.conf.ROOT_DIR_PATH, "anyscale-cloud-setup-gcp-oa.yaml"
        )
    )
    with open(file_path) as f:
        body = f.read()

    if enable_head_node_fault_tolerance:
        body += GCP_MEMORYSTORE_RESOURCE_CONFIG_TEMPLATE

    if shared_storage == SharedStorageType.NFS:
        if vpc_name:
            # Organization admin template - use OA filestore template
            body += GCP_FILESTORE_RESOURCE_CONFIG_TEMPLATE_OA
        else:
            # Regular template - use regular filestore template
            body += GCP_FILESTORE_RESOURCE_CONFIG_TEMPLATE

    deployment_manager_config = Template(body)

    params = {
        "ANYSCALE_HOST": shared_anyscale_conf.ANYSCALE_HOST,
        "ANYSCALE_CORS_ORIGIN": shared_anyscale_conf.ANYSCALE_CORS_ORIGIN,
        "REGION": region,
        # `cloud_id` is used as the deployment name, which cannot contain underscores.
        "CLOUD_ID": cloud_id.replace("_", "-").lower(),
        # `cloud_id_underscore` is used as the the label values, which should be consistent with GCE instances.
        "CLOUD_ID_UNDERSCORE": cloud_id,
        "PROJECT_ID": project_id,
        "ANYSCALE_ACCESS_SERVICE_ACCOUNT": anyscale_access_service_account_name,
        "WORKLOAD_IDENTITY_POOL_NAME": workload_identity_pool_name,
        "ANYSCALE_AWS_ACCOUNT": anyscale_aws_account,
        "ORGANIZATION_ID": organization_id,
        "SUBNET_CIDR": subnet_cidr,
        "ZONE": zone,
        "FILESHARE_CAPACITY_GB": fileshare_capacity_gb,
        "VPC_NAME": vpc_name,
        "ANYSCALE_CONTROL_PLANE_SERVICE_ACCOUNT_EMAIL": control_plane_service_account_email,
    }

    return deployment_manager_config.substitute(params)


def configure_firewall_policy(
    factory: GoogleCloudClientFactory,
    vpc_name: str,
    project_id: str,
    firewall_policy: str,
    subnet_cidr: str = "10.0.0.0/20",
):
    """ Add VPC association and necessary firewall rules to a firewall policy.
    """
    association = FirewallPolicyAssociation(
        name=f"{firewall_policy}-for-{vpc_name}",
        attachment_target=f"projects/{project_id}/global/networks/{vpc_name}",
    )

    rules = [
        FirewallPolicyRule(
            action="allow",
            direction="INGRESS",
            priority=1000,
            match={
                "src_ip_ranges": ["0.0.0.0/0"],
                "layer4_configs": [{"ip_protocol": "TCP", "ports": ["22", "443"]}],
            },
        ),
        FirewallPolicyRule(
            action="allow",
            direction="INGRESS",
            priority=1001,
            match={
                "src_ip_ranges": [subnet_cidr],
                "layer4_configs": [{"ip_protocol": "all"}],
            },
        ),
    ]

    network_firewall_policy_client = factory.compute_v1.NetworkFirewallPoliciesClient()
    client = factory.compute_v1.GlobalOperationsClient()
    try:
        operations = []
        for rule in rules:
            operation = network_firewall_policy_client.add_rule(
                project=project_id,
                firewall_policy=firewall_policy,
                firewall_policy_rule_resource=rule,
            )
            operations.append(operation)
        operation = network_firewall_policy_client.add_association(
            project=project_id,
            firewall_policy=firewall_policy,
            firewall_policy_association_resource=association,
        )
        operations.append(operation)

        for operation in operations:
            response = client.wait(project=project_id, operation=operation.name)
            if response.status != Operation.Status.DONE:
                # timeout
                raise ClickException(
                    "Timeout when trying to configure firewall policy."
                )
            if response.error:
                raise ClickException(
                    f"Failed to configure firewall policy {firewall_policy}. {response.error}"
                )
    except (NotFound, BadRequest) as e:
        raise ClickException(
            f"Failed to configure firewall policy {firewall_policy}. {e}"
        )


def delete_workload_identity_pool(
    factory: GoogleCloudClientFactory, pool_name: str, logger: CloudSetupLogger
):
    service = factory.build("iam", "v1").projects().locations().workloadIdentityPools()

    # we can directly delete the pool even if there're providers in it
    try:
        delete_pool_operation = service.delete(name=pool_name).execute()
        wait_for_operation_completion(
            service,
            {"name": delete_pool_operation["name"]},
            "deleting workload identity provider pool",
        )
        logger.info(f"Deleted workload identity pool: {pool_name}")
    except HttpError as e:
        if e.status_code == 404:
            return
        raise ClickException(
            f"Error occurred when trying to delete workload identity pool {pool_name}: {e}. Please delete the resources by yourself."
        )


def wait_for_operation_completion(
    service: Any,
    request: Dict[str, str],
    description: str = "Operation",
    timeout: int = 300,
    polling_interval: int = 3,
) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_operation = service.operations().get(**request).execute()
        if (
            current_operation.get("done", False)
            or current_operation.get("status", None) == "DONE"
        ):
            if "error" in current_operation:
                raise ClickException(
                    f"{description} encountered an error: {current_operation['error']}"
                )
            break
        time.sleep(polling_interval)
    else:
        raise ClickException(
            f"{description} did not complete within the timeout period ({timeout}s)"
        )


def delete_gcp_deployment(
    factory: GoogleCloudClientFactory, project_id: str, deployment_name: str,
):
    """
    Get the GCP Deployment and try to delete the deployment if it exists.
    """
    deployment_client = factory.build("deploymentmanager", "v2")
    try:
        # get the deployment
        deployment_client.deployments().get(
            project=project_id, deployment=deployment_name
        ).execute()
    except HttpError as e:
        if e.status_code == 404:
            # no deployment found
            return
        raise ClickException(
            f"Failed to get deployment {deployment_name}: {e}. Please delete the resources yourself"
        )

    try:
        # delete the deployment
        response = (
            deployment_client.deployments()
            .delete(project=project_id, deployment=deployment_name)
            .execute()
        )
        wait_for_operation_completion(
            deployment_client,
            {"operation": response["name"], "project": project_id},
            f"Delete deployment {deployment_name}",
            timeout=600,
        )
    except HttpError as e:
        raise ClickException(
            f"Failed to delete deployment {deployment_name}: {e}. Please delete the resources yourself."
        )


def update_deployment_with_bucket_only(
    factory: GoogleCloudClientFactory, project_id: str, deployment_name: str,
):
    """
    Update all resources in the deployment except the bucket.
    Removing the resources in deployment deletes the underlying resource by default.
    See https://cloud.google.com/deployment-manager/docs/reference/latest/deployments/update
    """
    # update the deployment with only the bucket
    gcp_deployment = get_deployment_config(factory, project_id, deployment_name)
    config = yaml.safe_load(gcp_deployment.config_content)

    # build a resource config with only the bucket
    resources = list(
        filter(lambda r: r["type"] == "storage.v1.bucket", config["resources"])
    )

    # update the deployment
    updated_config_content = yaml.dump({"resources": resources})
    update_deployment(
        factory,
        project_id,
        deployment_name,
        gcp_deployment.fingerprint,
        updated_config_content,
    )


def remove_firewall_policy_associations(
    factory: GoogleCloudClientFactory, project_id: str, firewall_policy_name: str,
):
    """
    Remove the firewall policy associations if exist
    """
    # get the firewall first
    network_firewall_policy_client = factory.compute_v1.NetworkFirewallPoliciesClient()
    operation_client = factory.compute_v1.GlobalOperationsClient()
    try:
        firewall_policy = network_firewall_policy_client.get(
            project=project_id, firewall_policy=firewall_policy_name
        )
    except NotFound:
        # no firewall policy
        return
    except (PermissionDenied, Forbidden, BadRequest) as e:
        raise ClickException(f"Failed to remove firewall policy associations. {e}")

    if not firewall_policy.associations:
        # no associations
        return

    operations = []
    associations = firewall_policy.associations
    for association in associations:
        try:
            remove_association_request = {
                "name": association.name,
                "project": project_id,
                "firewall_policy": firewall_policy_name,
            }
            operation = network_firewall_policy_client.remove_association(
                request=remove_association_request
            )
        except (BadRequest, NotFound):
            # BadRequest: no association found, NotFound: no firewall policy found
            continue
        except (PermissionDenied, Forbidden) as e:
            raise ClickException(f"Failed to remove firewall policy association. {e}")
        operations.append(operation)

    for operation in operations:
        response = operation_client.wait(project=project_id, operation=operation.name)
        if response.status != Operation.Status.DONE:
            # timeout
            raise ClickException(
                "Timeout when trying to remove firewall policy association."
            )
        if response.error:
            raise ClickException(
                f"Failed to remove firewall policy association. {response.error}"
            )


def delete_gcp_tls_certificates(
    factory: GoogleCloudClientFactory, project_id: str, cloud_id: str
):

    # Initialize the Compute Engine client
    certificate_manager_client = (
        factory.certificate_manager_v1.CertificateManagerClient()
    )

    try:
        location_path = certificate_manager_client.common_location_path(
            project_id, "global"
        )
    except NotFound:
        # no certificate
        return
    except Exception as e:  # noqa: BLE001
        raise ClickException(f"Failed to delete tls certificate. {e}")

    try:
        certificate_objects = certificate_manager_client.list_certificates(
            parent=location_path
        )

        certificate_maps_objects = certificate_manager_client.list_certificate_maps(
            parent=location_path
        )
    except Exception as e:  # noqa: BLE001
        raise ClickException(f"Failed to delete tls certificate. {e}")

    certificates = list(certificate_objects)
    certificate_maps = list(certificate_maps_objects)

    certificates = filter_resources(certificates, cloud_id)
    certificate_maps = filter_resources(certificate_maps, cloud_id)

    all_certificate_map_entries = []
    for certificate in certificates:
        certificate_name = certificate.split("/")[-1]
        if certificate_name:
            try:
                certificate_map_entry_path = certificate_manager_client.certificate_map_path(
                    project_id, "global", certificate_name
                )
                certificate_map_entry_objects = certificate_manager_client.list_certificate_map_entries(
                    parent=certificate_map_entry_path
                )
            except NotFound:
                # no certificate_map
                continue
            except Exception as e:  # noqa: BLE001
                raise ClickException(f"Failed to delete tls certificate. {e}")

            certificate_map_entries = list(certificate_map_entry_objects)
            all_certificate_map_entries.extend(certificate_map_entries)

    all_certificate_map_entries = filter_resources(
        all_certificate_map_entries, cloud_id
    )

    wait_on_operation(
        [
            certificate_manager_client.delete_certificate_map_entry(
                name=certificate_map_entry
            )
            for certificate_map_entry in all_certificate_map_entries
        ]
    )

    wait_on_operation(
        [
            certificate_manager_client.delete_certificate(name=certificate)
            for certificate in certificates
        ]
    )

    wait_on_operation(
        [
            certificate_manager_client.delete_certificate_map(name=certificate_map)
            for certificate_map in certificate_maps
        ]
    )


def wait_on_operation(operations):
    """
    Waits until all operations are finished before returning
    """
    for operation in operations:
        try:
            operation.result()
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to wait for operation to complete. {e}")


def filter_resources(resources, cloud_id) -> List[str]:
    ret = []
    for resource in resources:
        resource_labels = resource.labels
        if (
            "anyscale-cloud-id" in resource_labels
            and resource_labels["anyscale-cloud-id"] == cloud_id
        ):
            ret.append(resource.name)
    return ret


def filter_resources_on_cloud(resources, cloud_id):
    ret = []
    for resource in resources:
        if cloud_id in resource.name:
            ret.append(resource)

    return ret


def get_or_create_memorystore_gcp(
    factory: GoogleCloudClientFactory,
    cloud_id: str,
    deployment_name: str,
    project_id: str,
    region: str,
    logger: CloudSetupLogger,
    yes: bool = False,
) -> str:
    """
    Get or create Memorystore instance in the deployment.
    Return the memorystore name in the format of projects/<project number>/locations/<location>/instances/<instance id>
    """
    redis_resource_name = f"redis-{deployment_name}"

    # Get redis from deployment
    redis_instance = get_deployment_resource(
        factory, project_id, deployment_name, redis_resource_name
    )
    if redis_instance is not None:
        return f"projects/{project_id}/locations/{region}/instances/{redis_instance['name']}"

    # No redis instance found, create one
    gcp_deployment = get_deployment_config(factory, project_id, deployment_name)

    # Update body with Memorystore
    params = {
        "CLOUD_ID": cloud_id.replace("_", "-").lower(),
        "REGION": region,
        "PROJECT_ID": project_id,
    }
    gcp_memorystore_instance = Template(
        GCP_MEMORYSTORE_RESOURCE_CONFIG_TEMPLATE
    ).substitute(params)
    updated_config_content = gcp_deployment.config_content + gcp_memorystore_instance

    # update the deployment
    if not yes:
        # Preview the deployment update
        with logger.spinner("Creating deployment preview..."):
            update_deployment(
                factory,
                project_id,
                deployment_name,
                gcp_deployment.fingerprint,
                updated_config_content,
                preview=True,
            )
        deployment_url = f"https://console.cloud.google.com/dm/deployments/details/{deployment_name}?project={project_id}"
        logger.info(f"Please review the changes in the deployment at {deployment_url}")
        confirm("Continue?", yes)

    with logger.spinner("Updating deployment..."):
        # Get the fingerprint
        gcp_deployment = get_deployment_config(factory, project_id, deployment_name)
        # To update a deployment in preview, we must set the config_content to None
        update_deployment(
            factory, project_id, deployment_name, gcp_deployment.fingerprint, None
        )

    # Get redis from deployment
    redis_instance = get_deployment_resource(
        factory, project_id, deployment_name, redis_resource_name
    )
    if redis_instance is not None:
        return f"projects/{project_id}/locations/{region}/instances/{redis_instance['name']}"
    else:
        raise ClickException(
            f"Failed to create Memorystore instance in deployment {deployment_name}. Please contact Anyscale support."
        )


def update_deployment(
    factory: GoogleCloudClientFactory,
    project_id: str,
    deployment_name: str,
    fingerprint: str,
    config_content: Optional[str],
    preview: bool = False,
):
    """
    Update a deployment with the given config.
    """
    deployment_client = factory.build("deploymentmanager", "v2")
    description = "create preview for deployment" if preview else "update deployment"
    deployment: Dict[str, Any] = {
        "name": deployment_name,
        "fingerprint": fingerprint,
    }
    if config_content:
        deployment["target"] = {
            "config": {"content": config_content},
        }

    try:
        response = (
            deployment_client.deployments()
            .update(
                project=project_id,
                deployment=deployment_name,
                body=deployment,
                preview=preview,
            )
            .execute()
        )
        wait_for_operation_completion(
            deployment_client,
            {"operation": response["name"], "project": project_id},
            f"{description.capitalize()} {deployment_name}",
            timeout=GCP_DEPLOYMENT_PREVIEW_TIMEOUT_SECONDS
            if preview
            else GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS_LONG,
        )
    except HttpError as e:
        raise ClickException(f"Failed to {description} {deployment_name}. Error: {e}")


def get_deployment_resource(
    factory: GoogleCloudClientFactory,
    project_id: str,
    deployment_name: str,
    resource_name: str,
):
    """
    Get a resource from a deployment. If the resource is not found, return None.
    """
    deployment_client = factory.build("deploymentmanager", "v2")
    try:
        resource = (
            deployment_client.resources()
            .get(project=project_id, deployment=deployment_name, resource=resource_name)
            .execute()
        )
        return resource
    except HttpError as e:
        if e.status_code == 404:
            return None
        raise ClickException(
            f"Failed to get resource {resource_name} from deployment {deployment_name}: {e}. "
        )


def get_deployment_config(
    factory: GoogleCloudClientFactory, project_id: str, deployment_name: str
) -> GCPDeployment:
    """
    Get the fingerprint and manifest config from a deployment.
    """
    deployment_client = factory.build("deploymentmanager", "v2")
    try:
        # get the deployment
        deployment = (
            deployment_client.deployments()
            .get(project=project_id, deployment=deployment_name)
            .execute()
        )
        fingerprint = deployment["fingerprint"]
        manifest = deployment["manifest"]
        manifest_name = manifest.split("/")[-1]

        # get manifest
        manifest = (
            deployment_client.manifests()
            .get(
                project=project_id, deployment=deployment_name, manifest=manifest_name,
            )
            .execute()
        )
    except HttpError as e:
        raise ClickException(
            f"Failed to get deployment config from deployment {deployment_name}: {e}. "
        )
    return GCPDeployment(
        deployment_name=deployment_name,
        fingerprint=fingerprint,
        config_content=manifest["config"]["content"],
    )

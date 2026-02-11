import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from botocore.exceptions import ClientError, NoCredentialsError
from click import ClickException

from anyscale import __version__ as anyscale_version
from anyscale.anyscale_pydantic import BaseModel
from anyscale.aws_iam_policies import (
    ANYSCALE_IAM_POLICIES,
    AnyscaleIAMPolicy,
)
from anyscale.cli_logger import CloudSetupLogger, pad_string
from anyscale.util import (
    _client,
    confirm,
    generate_inline_policy_parameter,
    generate_inline_policy_resource,
    get_anyscale_cross_account_iam_policies,
    get_availability_zones,
    get_memorydb_supported_zones,
    MEMORY_DB_OUTPUT,
    MEMORY_DB_RESOURCE,
)
from anyscale.utils.cloud_utils import modify_memorydb_parameter_group


DETECT_DRIFT_TIMEOUT_SECONDS = 60 * 5  # 5 minutes
CREATE_CHANGE_SET_TIMEOUT_SECONDS = 60 * 5  # 5 minutes
UPDATE_CLOUDFORMATION_STACK_TIMEOUT_SECONDS = 60 * 5  # 5 minutes
# Some resources (e.g. memoryDB) take a long time to create, so we need to increase the timeout.
CLOUDFORMATION_TIMEOUT_SECONDS_LONG = 1800  # 30 minutes


CUSTOMER_DRIFTS_POLICY_NAME = "Customer_Drifts_Policy"
MEMORYDB_REDIS_PORT = "6379"


class PropertyDifference(BaseModel):
    """
    PropertyDifference is a field in the drift response.
    It describes the difference between the expected and actual value of a property.

    Example:
    {
        "DifferenceType": "NOT_EQUAL",
        "PropertyPath": "/Policies/0/PolicyDocument/Statement/0/Sid",
        "ExpectedValue": "AllowAnyScaleCLI",
        "ActualValue": "AllowAnyScaleCLI2"
    }

    We use the PropertyPath to recognize the drift type and identify the policy and statement.
    """

    DifferenceType: str
    PropertyPath: str
    ExpectedValue: Optional[str]
    ActualValue: Optional[str]

    def get_policy_number(self) -> Optional[int]:
        """
        Get the policy number from the PropertyPath.
        If the PropertyPath is not a policy path, return None.
        """
        match = re.search(r"\/Policies\/(\d+)", self.PropertyPath)
        if match is None:
            return None
        return int(match.group(1))

    def get_statement_number(self) -> Optional[int]:
        """
        Get the statement number from the PropertyPath.
        If the PropertyPath is not a statement path, return None.
        """
        match = re.search(r"\/Statement\/(\d+)", self.PropertyPath)
        if match is None:
            return None
        return int(match.group(1))

    def is_add_or_remove_statement(self) -> bool:
        """
        Check if the PropertyPath is a statement path
        and the difference type is ADD or REMOVE.
        """
        if (
            re.match(
                r"\/Policies\/(\d+)\/PolicyDocument\/Statement\/(\d+)$",
                self.PropertyPath,
            )
            is not None
        ):
            if self.DifferenceType in ("ADD", "REMOVE"):
                return True
            else:
                raise ClickException(f"Drift {self.json()} cannot be resolved.")
        return False


class AWSCloudformationHandler:
    def __init__(
        self, aws_cloudformation_stack_id: str, region: str, logger: CloudSetupLogger
    ):
        self.aws_cloudformation_stack_id = aws_cloudformation_stack_id
        self.region = region
        self.logger = logger
        self.cfn_client = _client("cloudformation", self.region)
        try:
            cfn_stacks = self.cfn_client.describe_stacks(
                StackName=self.aws_cloudformation_stack_id
            )["Stacks"]
        except (ClientError, NoCredentialsError) as e:
            if isinstance(e, NoCredentialsError):
                raise ClickException(
                    "Unable to locate AWS credentials. Please make sure you have AWS credentials configured.",
                )
            raise ClickException(
                f"Failed to describe cloudformation stack {self.aws_cloudformation_stack_id}. Error: {e}"
            )

        if len(cfn_stacks) != 1:
            raise ClickException(
                f"Got unexpected number of cloudformation stacks returned with stack ID: {self.aws_cloudformation_stack_id}.\n "
                f"Results returned: {cfn_stacks}\n"
                "Please check your cloudformation stack page to make sure the stack still exists."
            )
        cfn_stack = cfn_stacks[0]

        # Validate the cloudformation stack status
        if not cfn_stack["StackStatus"].endswith("_COMPLETE"):
            raise ClickException(
                f"Cloudformation stack {cfn_stack['StackName']} is in {cfn_stack['StackStatus']} status. Please check your cloudformation stack page to make sure the stack is in a stable state."
            )

        self.stack = cfn_stack

    def get_cloudformation_stack(self):
        return self.stack

    def get_resource(self, logical_resource_id: str):
        try:
            resource = self.cfn_client.describe_stack_resource(
                StackName=self.aws_cloudformation_stack_id,
                LogicalResourceId=logical_resource_id,
            )
            return resource["StackResourceDetail"]
        except ClientError:
            raise ClickException(
                f"Failed to get resource {logical_resource_id} from stack {self.aws_cloudformation_stack_id}."
            )

    def detect_drift(self) -> List[Dict]:
        """
        Detect drifts on cloudformation stack.
        More about drifts on cfn stack: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-stack-drift.html
        """
        stack_name = self.aws_cloudformation_stack_id
        with self.logger.spinner("Detecting drift on cloudformation stack..."):
            # Init drift detection
            try:
                drift_detection_id = self.cfn_client.detect_stack_drift(
                    StackName=stack_name
                )["StackDriftDetectionId"]
            except ClientError as e:
                raise ClickException(
                    f"Failed to detect drift on stack {stack_name}: {e}"
                )

            # Polling drift detection status
            end_time = time.time() + DETECT_DRIFT_TIMEOUT_SECONDS
            while time.time() < end_time:
                time.sleep(1)
                try:
                    response = self.cfn_client.describe_stack_drift_detection_status(
                        StackDriftDetectionId=drift_detection_id
                    )
                except ClientError as e:
                    raise ClickException(
                        f"Failed to get drift detection status for stack {stack_name}: {e}"
                    )
                if response["DetectionStatus"] == "DETECTION_COMPLETE":
                    drift_response = self.cfn_client.describe_stack_resource_drifts(
                        StackName=stack_name,
                        StackResourceDriftStatusFilters=[
                            "MODIFIED",
                            "DELETED",
                            "NOT_CHECKED",
                        ],
                    )
                    drifts = drift_response["StackResourceDrifts"]
                    self.logger.info("Drift detection completed.")
                    return drifts
                elif response["DetectionStatus"] == "DETECTION_FAILED":
                    raise ClickException(
                        f'Drift detection failed. Error: {response["DetectionStatusReason"]}'
                    )
        raise ClickException("Drift detection timeout. Please try again later.")

    def resolve_drift(self, drift: Dict):
        """
        Resolve the drift on the cross account IAM role.

        1) Drifts we care about:
        We only resolve the drifts on the Anyscale-maintained inline policies (defined in ANYSCALE_IAM_POLICIES).
        For other drifts, we'll skip them and keep the drifts as is.

        2) How to resolve the drift:
        We first identify the drifted statements in the Anyscale-maintained inline policies from the drift detection response.
        Then we append the drifted statements to the Customer_Drifts_Policy of the cross account IAM role.
        The reason is that we'll overwrite the Anyscale-maintained inline policies and we don't want to lose the drifted statements.
        If the customer removes some of the statements, they'll need to remove it again manually.
        """
        try:
            drifted_statements = extract_drifted_statements(drift)
            if len(drifted_statements) == 0:
                self.logger.info("No drifted statements found.")
                return True
            role_name = drift["PhysicalResourceId"]
            append_statements_to_customer_drifts_policy(
                self.region, role_name, drifted_statements
            )
            self.logger.info(
                f"Drifted statements have been appended to the policy {CUSTOMER_DRIFTS_POLICY_NAME} of the role {role_name}."
            )
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to resolve drift. Error: {e}")
            return False

    def get_template_body(self) -> str:
        try:
            template_body = self.cfn_client.get_template(
                StackName=self.aws_cloudformation_stack_id, TemplateStage="Original"
            )["TemplateBody"]
            return template_body
        except ClientError as e:
            raise ClickException(
                f"Failed to get template body for stack {self.aws_cloudformation_stack_id}: {e}"
            )

    def get_processed_template_body(self) -> Dict[str, Any]:
        try:
            template_body = self.cfn_client.get_template(
                StackName=self.aws_cloudformation_stack_id, TemplateStage="Processed"
            )["TemplateBody"]
            return template_body
        except ClientError as e:
            raise ClickException(
                f"Failed to get template body for stack {self.aws_cloudformation_stack_id}: {e}"
            )

    def update_cloudformation_stack(
        self,
        updated_template_body: str,
        updated_parameters: List[Dict],
        yes: bool = False,
        timeout_seconds: int = UPDATE_CLOUDFORMATION_STACK_TIMEOUT_SECONDS,
    ):
        """
        Update the cloudformation stack with the updated parameter list and template body.
        Update the stack by creating a change set and executing it.
        """
        stack_name = self.aws_cloudformation_stack_id

        # Create change set
        with self.logger.spinner("Creating change set for cloud update..."):
            response = self.cfn_client.create_change_set(
                StackName=stack_name,
                ChangeSetName=f"AnyscaleCloudUpdate{secrets.token_hex(4)!s}",
                TemplateBody=updated_template_body,
                Parameters=updated_parameters,
                Capabilities=["CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
                ChangeSetType="UPDATE",
            )

            change_set_id = response["Id"]

            # Polling change set status
            end_time = time.time() + CREATE_CHANGE_SET_TIMEOUT_SECONDS
            while time.time() < end_time:
                time.sleep(1)
                response = self.cfn_client.describe_change_set(
                    ChangeSetName=change_set_id, StackName=stack_name
                )
                if response["Status"] == "CREATE_COMPLETE":
                    break
                elif response["Status"] == "FAILED":
                    self.cfn_client.delete_change_set(ChangeSetName=change_set_id)
                    raise ClickException(
                        f"Failed to create change set for cloud update. {response['StatusReason']}"
                    )
            else:
                raise ClickException(
                    "Timeout when creating change set for cloud update. Please try again later."
                )

        # Preview change set
        stack_id = response["StackId"]
        stack_url = f"https://{self.region}.console.aws.amazon.com/cloudformation/home?region={self.region}#/stacks/stackinfo?stackId={stack_id}"
        change_set_url = f"https://{self.region}.console.aws.amazon.com/cloudformation/home?region={self.region}#/stacks/changesets/changes?stackId={stack_id}&changeSetId={change_set_id}"
        self.logger.info(f"Change set created at {change_set_url}")
        confirm(
            "Please review the change set before updating the stack. Do you want to proceed with the update?",
            yes,
        )

        # Execute change set
        with self.logger.spinner("Executing change set for cloud update..."):
            response = self.cfn_client.execute_change_set(ChangeSetName=change_set_id)

            # Polling cfn stack status
            end_time = time.time() + timeout_seconds
            while time.time() < end_time:
                time.sleep(1)
                response = self.cfn_client.describe_stacks(StackName=stack_name)
                stack = response["Stacks"][0]
                if stack["StackStatus"] == "UPDATE_COMPLETE":
                    break
                elif stack["StackStatus"] == "UPDATE_ROLLBACK_COMPLETE":
                    raise ClickException(
                        f"Failed to execute change set. Please check the cloudformation stack events for more details ({stack_url})"
                    )
            else:
                raise ClickException(
                    f"Timeout when executing change set. Please check the cloudformation stack events for more details ({stack_url})"
                )


def validate_stack_version(stack_parameters: List[Dict]) -> bool:
    cfn_stack_version = None
    for parameter in stack_parameters:
        if parameter["ParameterKey"] == "AnyscaleCLIVersion":
            cfn_stack_version = parameter["ParameterValue"]
            break

    # return False if:
    # 1) cfn_stack_version is None
    # 2) cfn_stack_version is the different with the CLI version
    # 3) cfn_stack_version is the same as the CLI version but the CLI version is dev version
    return not (
        cfn_stack_version == anyscale_version and anyscale_version != "0.0.0-dev"
    )


def is_template_policy_documents_up_to_date(stack_parameters: List[Dict]) -> bool:
    """
    Check if the policy documents in the cfn template are up to date.
    """
    parameter_dict = {p["ParameterKey"]: p["ParameterValue"] for p in stack_parameters}
    for policy in ANYSCALE_IAM_POLICIES:
        if policy.parameter_key not in parameter_dict:
            return False
        if parameter_dict[policy.parameter_key] != policy.policy_document:
            return False
    return True


def format_drifts(drifts: List[Dict]) -> str:
    padding_size = 40
    outputs: List[str] = []
    outputs.append(
        f'{pad_string("Resource Type", padding_size)}'
        f'{pad_string("Resource Id", padding_size)}'
        f'{pad_string("Drift status", padding_size)}'
    )
    outputs.append(
        f'{pad_string("-------------", padding_size)}'
        f'{pad_string("-----------", padding_size)}'
        f'{pad_string("------------", padding_size)}'
    )
    for drift in drifts:
        outputs.append(
            f'{pad_string(drift["ResourceType"], padding_size)}'
            f'{pad_string(drift["PhysicalResourceId"], padding_size)}'
            f'{pad_string(drift["StackResourceDriftStatus"], padding_size)}'
        )
    return "\n".join(outputs)


def extract_cross_account_iam_role_drift(drifts: List[Dict]) -> Optional[Dict]:
    """
    Check if the cross account IAM role is drifted.
    If so, return the drift information.
    """
    for drift in drifts:
        # TODO (congding): don't hardcode "customerRole"
        if (
            drift["ResourceType"] == "AWS::IAM::Role"
            and drift["LogicalResourceId"] == "customerRole"
            and drift["StackResourceDriftStatus"] != "IN_SYNC"
        ):
            return drift
    return None


def get_all_sids_from_policy(policy: Dict) -> List[str]:
    """
    Get all the SIDs from the policy.
    SIDs (Statement ID) are used to identify the statements.
    """
    sids: List[str] = []
    for statement in policy["PolicyDocument"]["Statement"]:
        if "Sid" in statement:
            sids.append(statement["Sid"])
        else:
            raise ClickException(
                f"Statement {statement} in policy {policy['PolicyName']} doesn't have a Sid."
            )
    return sids


def get_sids_to_remove(
    diffs: List[Dict], expected_policies: List[Dict],
) -> Dict[str, Set[str]]:
    """
    Get the SIDs from the drifted statements to remove from the expected policies.
    """
    policy_names_to_overwrite = [policy.policy_name for policy in ANYSCALE_IAM_POLICIES]

    sids_to_remove: Dict[str, Set[str]] = {}

    for raw_diff in diffs:
        diff = PropertyDifference(**raw_diff)
        policy_number = diff.get_policy_number()
        if policy_number is None:
            # We'll skip the drift if it's not a policy drift
            # We only update the maintained inline policies
            # If there're other drifts (e.g., AssumeRolePolicyDocument drift)
            # we'll skip them and keep the drifts as is.
            continue
        try:
            policy_name = expected_policies[policy_number]["PolicyName"]
        except IndexError:
            continue
        if policy_name not in policy_names_to_overwrite:
            # Not a policy we care about
            # We only update the maintained inline policies
            # If the drifts are on other policies (e.g., newly created inline policies),
            # we'll skip them and keep the drifts as is.
            continue

        # Drift detected on a maintained policy
        statement_number = diff.get_statement_number()
        if statement_number is None:
            raise ClickException(
                f"Drift {diff} in policy {policy_name} cannot be resolved."
            )
        if diff.is_add_or_remove_statement():
            # Policy statement added or removed
            # No need to append to the drifted policy
            continue
        expected_statements = expected_policies[policy_number]["PolicyDocument"][
            "Statement"
        ]
        try:
            sid = expected_statements[statement_number]["Sid"]
            if policy_name not in sids_to_remove:
                sids_to_remove[policy_name] = set()
            sids_to_remove[policy_name].add(sid)
        except IndexError:
            continue

    return sids_to_remove


def generate_drifted_statements_to_append(
    actual_policies: List[Dict], undrifted_sid: Dict[str, Set[str]]
) -> List[Dict]:
    """
    Generate the drifted statements to append to the drifted policy.

    Iterate through the actual policies and skip the undrifted statements.
    For the drifted statements, we'll append them to the drifted statements list.

    """
    no_sid_cnt = 0
    sid_suffix = str(secrets.token_hex(4))
    drifted_statements: List[Dict] = []

    for actual_policy in actual_policies:
        policy_name = actual_policy["PolicyName"]
        if policy_name not in undrifted_sid:
            # Not a policy we care about
            continue
        statements = actual_policy["PolicyDocument"]["Statement"]
        for statement in statements:
            sid = statement.get("Sid", None)
            if sid is None:
                # No sid in the statement
                # Generate a new sid
                statement["Sid"] = f"Drifted{no_sid_cnt}{sid_suffix}"
                no_sid_cnt += 1
                drifted_statements.append(statement)
            elif sid not in undrifted_sid[policy_name]:
                # drifted statement
                statement["Sid"] = f"{statement['Sid']}Drifted{sid_suffix}"
                drifted_statements.append(statement)

    return drifted_statements


def get_all_sids(expected_policies) -> Dict[str, Set[str]]:
    all_sids: Dict[str, Set[str]] = {}
    policies_to_overwrite = [policy.policy_name for policy in ANYSCALE_IAM_POLICIES]
    for policy in expected_policies:
        if policy["PolicyName"] not in policies_to_overwrite:
            continue
        if policy["PolicyDocument"]["Version"] != "2012-10-17":
            raise ClickException(
                f"Unexpected policy version {policy['PolicyDocument']['Version']} for policy {policy['PolicyName']}"
            )
        all_sids[policy["PolicyName"]] = set(get_all_sids_from_policy(policy))
    return all_sids


def extract_drifted_statements(drift: Dict) -> List[Dict]:  # noqa: PLR0912
    """
    Extract the drifted statements from the role drift information.

    1) About SID (Statement ID):
    Each inline policy contains a list of statements, and each statement has a SID (Statement ID).
    We use the SID to identify the statements.
    Anyscale-maintained policies is guaranteed to have a SID for each statement.

    2) Get drifted statements:
    Step 1: Get all the SIDs from the expected policies.
    Step 2: Leverage the drift detection result to identify the drifted statements.
    If there's a drift on a statement, we will add it to a set of SIDs to remove.
    Step 3: Remove the drifted SIDs and get a set of undrifted SIDs.
    Step 4: Get all the statements from the actual policy and filter out the undrfited SIDs.
    """
    # Get all sids from expected policies
    expected_policies = json.loads(drift["ExpectedProperties"])["Policies"]
    all_sids = get_all_sids(expected_policies)

    # Remove drifted statements
    diffs = drift["PropertyDifferences"]
    sids_to_remove = get_sids_to_remove(diffs, expected_policies)

    undrifted_sid: Dict[str, Set[str]] = {}
    for policy_name, sids in all_sids.items():
        if policy_name not in sids_to_remove:
            undrifted_sid[policy_name] = sids
            continue
        undrifted_sid[policy_name] = sids.difference(sids_to_remove[policy_name])

    # Get drifted statements
    actual_policies = json.loads(drift["ActualProperties"])["Policies"]
    drifted_statements = generate_drifted_statements_to_append(
        actual_policies, undrifted_sid
    )

    return drifted_statements


def append_statements_to_customer_drifts_policy(
    region: str, role_name: str, statements: List[Dict]
):
    """
    Append statements to the customer drfits policy.
    If the inline policy doesn't exist, create it.
    """
    iam = _client("iam", region)
    policy_document = None
    try:
        policy = iam.get_role_policy(
            RoleName=role_name, PolicyName=CUSTOMER_DRIFTS_POLICY_NAME,
        )
        policy_document = policy["PolicyDocument"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise e
    # Create the policy
    if policy_document is None:
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [],
        }

    # Append new statements
    for statement in statements:
        policy_document["Statement"].append(statement)  # type: ignore
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=CUSTOMER_DRIFTS_POLICY_NAME,
            PolicyDocument=json.dumps(policy_document),
        )
    except ClientError as e:
        raise ClickException(
            f"Failed to append statements to the drifted policy. Error: {e}"
        )


def merge_parameters(
    existing_parameters: List[Dict], parameters_to_update: List[Dict]
) -> List[Dict]:
    """
    Overwrite the existing parameters with the parameters to update.
    If the parameter to update does not exist in the existing parameters, add it to the existing parameters.

    The returned parameter list should contain all the combined parameters.
    """
    returned_parameters: Dict = {
        p["ParameterKey"]: p["ParameterValue"] for p in existing_parameters
    }
    for p in parameters_to_update:
        returned_parameters[p["ParameterKey"]] = p["ParameterValue"]
    return [
        {"ParameterKey": k, "ParameterValue": v} for k, v in returned_parameters.items()
    ]


def add_missing_parameters_to_template_body(
    template_body: str, missing_parameters: Set[str]
) -> str:
    """
    Add missing parameters to template body.

    For AnyscaleCLIVersion, we only need to add the parameter part.

    For other parameters for inline IAM policies, we need to add both parameter and resource definitions.
    """
    # Get all the missing parameters' policy information
    policy_dict: Dict[str, AnyscaleIAMPolicy] = {}
    for policy in ANYSCALE_IAM_POLICIES:
        if policy.parameter_key in missing_parameters:
            policy_dict[policy.parameter_key] = policy

    parameter_substitutions = ["Parameters:"]
    resource_substitutions = ["Resources:"]

    for parameter_key in missing_parameters:
        if parameter_key == "AnyscaleCLIVersion":
            parameter_substitutions.append(
                "  AnyscaleCLIVersion:\n    Description: Anyscale CLI version\n    Type: String\n"
            )
        elif parameter_key == "MemoryDBRedisPort":
            parameter_substitutions.append(
                f'  MemoryDBRedisPort:\n    Description: Port for MemoryDB Redis\n    Type: String\n    Default: "{MEMORYDB_REDIS_PORT}"\n'
            )
        else:
            policy = policy_dict[parameter_key]
            parameter_substitutions.append(
                generate_inline_policy_parameter(policy) + "\n"
            )
            resource_substitutions.append(
                generate_inline_policy_resource(policy) + "\n"
            )

    template_body = template_body.replace(
        "Parameters:", "\n".join(parameter_substitutions),
    )

    template_body = template_body.replace(
        "Resources:", "\n".join(resource_substitutions),
    )
    return template_body


def try_delete_customer_drifts_policy(cloud):
    iam_client = _client("iam", cloud.region)
    role_name = cloud.credentials.split("/")[-1]
    try:
        iam_client.delete_role_policy(
            RoleName=role_name, PolicyName=CUSTOMER_DRIFTS_POLICY_NAME
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise ClickException(
                f"Failed to delete inline policy {CUSTOMER_DRIFTS_POLICY_NAME} in role {role_name}: {e}"
            )


def update_template_with_memorydb(
    template_body: str, region: str, zone_name_to_resource_logical_id: Dict[str, str]
) -> str:
    """
    Add MemoryDB resources and outputs to template body.
    """
    # Get the supported zone names for MemoryDB
    zone_ids_to_names = get_availability_zones(region)
    supported_zone_names = get_memorydb_supported_zones(region, zone_ids_to_names)

    # Get the subnet logical IDs for the supported zones
    subnet_logical_ids = sorted(
        zone_name_to_resource_logical_id[zone]
        for zone in supported_zone_names
        if zone in zone_name_to_resource_logical_id
    )

    # Generate the subnet list
    subnet_list = [
        f"        - !Ref {subnet_logical_id}"
        for subnet_logical_id in subnet_logical_ids
    ]
    memory_db_resource_str = MEMORY_DB_RESOURCE.format("\n".join(subnet_list))

    updated_template_body = template_body.replace(
        "Resources:", f"Resources:\n{memory_db_resource_str}\n"
    ).replace("Outputs:", f"Outputs:\n{MEMORY_DB_OUTPUT}\n")

    return updated_template_body


def generate_updated_parameters_and_template(
    cfn_stack_parameters: List[Dict],
    template_body: str,
    parameters_to_update: List[Dict],
) -> Tuple[List[Dict], str]:
    """
    Generate the updated parameter list and updated tempelate body.
    If parameter exists in both cfn_stack_parameters and parameters_to_update,
    we'll override the value in cfn_stack_parameters with the value in parameters_to_update.
    If parameter only exists in parameters_to_update, we'll add it to cfn_stack_parameters.
    We also update the template body with the missing parameters.
    """
    updated_parameters = merge_parameters(cfn_stack_parameters, parameters_to_update)
    missing_parameters = {p["ParameterKey"] for p in parameters_to_update}.difference(
        {p["ParameterKey"] for p in cfn_stack_parameters}
    )
    if len(missing_parameters) > 0:
        updated_template_body = add_missing_parameters_to_template_body(
            template_body, missing_parameters
        )
    else:
        updated_template_body = template_body
    return updated_parameters, updated_template_body


def update_iam_role(
    region: str,
    aws_cloudformation_stack_id: str,
    logger: CloudSetupLogger,
    yes: bool = False,
):
    cfn_handler = AWSCloudformationHandler(aws_cloudformation_stack_id, region, logger)
    cfn_stack = cfn_handler.get_cloudformation_stack()
    cfn_stack_parameters = cfn_stack["Parameters"]

    # Validate cfn stack version is not the same as the CLI version
    if not validate_stack_version(cfn_stack_parameters):
        logger.info(
            "Cloud is already up-to-date. Please make sure your Anyscale CLI is on the latest version."
        )
        return

    # Check if the cfn template are the same as the CLI definition
    if is_template_policy_documents_up_to_date(cfn_stack_parameters):
        logger.info("No inline policy changes detected.")
        return

    drifts = cfn_handler.detect_drift()
    role_drift = extract_cross_account_iam_role_drift(drifts)
    if role_drift:
        inline_policies = [policy.policy_name for policy in ANYSCALE_IAM_POLICIES]
        customer_role_drift_details_url = f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks/drifts/info?stackId={aws_cloudformation_stack_id}&logicalResourceId={role_drift['LogicalResourceId']}"
        logger.info(
            f"Drfits detected on the cross account IAM role: {customer_role_drift_details_url}"
        )
        logger.warning(
            logger.highlight(
                f"Cloud update will create or override the inline policies: {inline_policies}. "
                f"We'll append drifted statements to inline policy {CUSTOMER_DRIFTS_POLICY_NAME} so you won't lose your change."
                "Please note that this will only include the permissions you added to the role. "
                "If you have removed permissions or restricted the resources from the role, "
                "you'll need to manually apply the change again after cloud update."
            )
        )
        confirm(
            "Proceed to resolve the drift?", yes,
        )
        resolved_result = cfn_handler.resolve_drift(role_drift)
        if not resolved_result:
            confirm(
                "[DANGER] If you proceed to update the cloud,"
                f"we'll overwrite the changes you made in inline policies {inline_policies}. "
                "We recommend creating a new inline policy to include your changes. "
                "Continue to overwrite the inline policies?",
                yes,
            )
    template_body = cfn_handler.get_template_body()
    # Get updated parameter list
    # We update the following 2 types of parameters:
    # 1. AnyscaleCLIVersion: the version of CLI
    # 2. Parameters that define the inline policy documents for the cross account IAM role
    # Other parameters should remain unchanged.
    parameters_to_update: List[Dict] = get_anyscale_cross_account_iam_policies()
    parameters_to_update.append(
        {"ParameterKey": "AnyscaleCLIVersion", "ParameterValue": anyscale_version}
    )

    (
        updated_parameters,
        updated_template_body,
    ) = generate_updated_parameters_and_template(
        cfn_stack_parameters, template_body, parameters_to_update
    )

    cfn_handler.update_cloudformation_stack(
        updated_template_body, updated_parameters, yes,
    )


def get_or_create_memorydb(
    region: str,
    aws_cloudformation_stack_id: str,
    logger: CloudSetupLogger,
    yes: bool = False,
) -> str:
    """
    Get or create the memorydb cluster.
    """
    cfn_handler = AWSCloudformationHandler(aws_cloudformation_stack_id, region, logger)

    cfn_stack = cfn_handler.get_cloudformation_stack()
    cfn_stack_parameters = cfn_stack["Parameters"]
    template_body = cfn_handler.get_template_body()
    processed_template = cfn_handler.get_processed_template_body()
    zone_name_to_resource_logical_id = {}
    for resource_logical_id, resource_definition in processed_template.get(
        "Resources", []
    ).items():
        if resource_definition.get("Type") == "AWS::EC2::Subnet":
            zone_name_to_resource_logical_id[
                resource_definition["Properties"]["AvailabilityZone"]
            ] = resource_logical_id

    if "AWS::MemoryDB::Cluster" in template_body:
        logger.info("MemoryDB cluster already exists in cloudformation template.")
        memorydb = cfn_handler.get_resource("MemoryDB")
        return memorydb["PhysicalResourceId"]

    # Update parameters
    parameters_to_update: List[Dict] = [
        {"ParameterKey": "MemoryDBRedisPort", "ParameterValue": MEMORYDB_REDIS_PORT}
    ]
    updated_parameters, template_body = generate_updated_parameters_and_template(
        cfn_stack_parameters, template_body, parameters_to_update
    )

    # Update template
    updated_template_body = update_template_with_memorydb(
        template_body, region, zone_name_to_resource_logical_id
    )

    # Update cfn stack
    cfn_handler.update_cloudformation_stack(
        updated_template_body,
        updated_parameters,
        yes,
        timeout_seconds=CLOUDFORMATION_TIMEOUT_SECONDS_LONG,
    )
    memorydb_parameter_group = cfn_handler.get_resource("MemoryDBParameterGroup")
    modify_memorydb_parameter_group(
        memorydb_parameter_group["PhysicalResourceId"], region
    )

    memorydb = cfn_handler.get_resource("MemoryDB")
    return memorydb["PhysicalResourceId"]

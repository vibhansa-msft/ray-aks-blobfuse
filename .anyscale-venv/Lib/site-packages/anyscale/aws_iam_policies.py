import json
from typing import Any, Dict

from anyscale.anyscale_pydantic import BaseModel


# Used for data-gplane role.
AMAZON_S3_FULL_ACCESS_POLICY_NAME = "AmazonS3FullAccess"
AMAZON_S3_FULL_ACCESS_POLICY_ARN = (
    f"arn:aws:iam::aws:policy/{AMAZON_S3_FULL_ACCESS_POLICY_NAME}"
)

AMAZON_ECR_READONLY_ACCESS_POLICY_NAME = "AmazonEC2ContainerRegistryReadOnly"
AMAZON_ECR_READONLY_ACCESS_POLICY_ARN = (
    f"arn:aws:iam::aws:policy/{AMAZON_ECR_READONLY_ACCESS_POLICY_NAME}"
)
DEFAULT_RAY_IAM_ASSUME_ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": {
        "Effect": "Allow",
        "Principal": {"Service": ["ec2.amazonaws.com"]},
        "Action": "sts:AssumeRole",
    },
}

# Used for control-plane role.
ANYSCALE_IAM_POLICY_NAME_STEADY_STATE = "Anyscale_IAM_Policy_Steady_State"

# Refer to https://docs.anyscale.com/cloud-deployment/aws/manage-clouds#appendix-detailed-resource-requirements
ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE: Dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": ["iam:PassRole", "iam:GetInstanceProfile"],
            "Resource": "*",
        },
        {
            "Sid": "RetrieveGenericAWSResources",
            "Effect": "Allow",
            "Action": [
                # Populates metadata about what is available
                # in the account.
                "ec2:DescribeAvailabilityZones",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeRegions",
                "ec2:DescribeAccountAttributes",
            ],
            "Resource": "*",
        },
        {
            "Sid": "DescribeRunningResources",
            "Effect": "Allow",
            "Action": [
                # Determines cluster/configuration status.
                "ec2:DescribeInstances",
                "ec2:DescribeSubnets",
                "ec2:DescribeRouteTables",
                "ec2:DescribeSecurityGroups",
            ],
            "Resource": "*",
        },
        {
            "Sid": "InstanceManagementCore",
            "Effect": "Allow",
            "Action": [
                # Minimal Permissions to Run Instances on Anyscale.
                "ec2:RunInstances",
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:TerminateInstances",
            ],
            "Resource": "*",
        },
        {
            "Sid": "InstanceTagMangement",
            "Effect": "Allow",
            "Action": ["ec2:CreateTags", "ec2:DeleteTags"],
            "Resource": "*",
        },
        {
            "Sid": "InstanceManagementSpot",
            "Effect": "Allow",
            "Action": [
                # Extended Permissions to Run Instances on Anyscale.
                "ec2:CancelSpotInstanceRequests",
                "ec2:ModifyImageAttribute",
                "ec2:ModifyInstanceAttribute",
                "ec2:RequestSpotInstances",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ResourceManagementExtended",
            "Effect": "Allow",
            "Action": [
                # Volume management
                "ec2:AttachVolume",
                "ec2:CreateVolume",
                "ec2:DescribeVolumes",
                # IAMInstanceProfiles
                "ec2:AssociateIamInstanceProfile",
                "ec2:DisassociateIamInstanceProfile",
                "ec2:ReplaceIamInstanceProfileAssociation",
                # Placement groups
                "ec2:CreatePlacementGroup",
                # Address Management
                "ec2:AllocateAddress",
                "ec2:ReleaseAddress",
                # Additional DescribeResources
                "ec2:DescribeIamInstanceProfileAssociations",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribePlacementGroups",
                "ec2:DescribePrefixLists",
                "ec2:DescribeReservedInstancesOfferings",
                "ec2:DescribeSpotInstanceRequests",
                "ec2:DescribeSpotPriceHistory",
            ],
            "Resource": "*",
        },
        {
            "Action": ["elasticfilesystem:DescribeMountTargets"],
            "Effect": "Allow",
            "Resource": "*",
            "Sid": "EFSManagement",
        },
        {
            "Sid": "CreateSpotRole",
            "Effect": "Allow",
            "Action": ["iam:CreateServiceLinkedRole"],
            "Resource": "arn:aws:iam::*:role/aws-service-role/spot.amazonaws.com/AWSServiceRoleForEC2Spot",
            "Condition": {"StringLike": {"iam:AWSServiceName": "spot.amazonaws.com"}},
        },
    ],
}


def get_anyscale_iam_permissions_ec2_restricted(cloud_id: str) -> Dict[str, Any]:
    # Refer to https://docs.anyscale.com/cloud-deployment/aws/manage-clouds#appendix-detailed-resource-requirements
    return {
        "Version": ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE["Version"],
        "Statement": [
            statement
            for statement in ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE["Statement"]
            if statement["Sid"] != "InstanceManagementCore"
        ]
        + [
            {
                "Sid": "DenyTaggingOnOtherInstances",
                "Effect": "Deny",
                "Action": ["ec2:DeleteTags", "ec2:CreateTags"],
                "Resource": "arn:aws:ec2:*:*:instance/*",
                "Condition": {
                    "StringNotEquals": {
                        "aws:ResourceTag/anyscale-cloud-id": f"{cloud_id}",
                        "ec2:CreateAction": ["RunInstances", "StartInstances"],
                    }
                },
            },
            {
                "Sid": "RestrictedEc2Termination",
                "Effect": "Allow",
                "Action": ["ec2:TerminateInstances", "ec2:StopInstances"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"aws:ResourceTag/anyscale-cloud-id": f"{cloud_id}"}
                },
            },
            {
                "Sid": "RestrictedInstanceStart",
                "Effect": "Allow",
                "Action": ["ec2:StartInstances", "ec2:RunInstances"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"aws:RequestTag/anyscale-cloud-id": f"{cloud_id}"},
                    "ForAnyValue:StringEquals": {"aws:TagKeys": ["anyscale-cloud-id"]},
                },
            },
            {
                "Sid": "AllowRunInstancesForUntaggedResources",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": [
                    "arn:aws:ec2:*::image/*",
                    "arn:aws:ec2:*::snapshot/*",
                    "arn:aws:ec2:*:*:subnet/*",
                    "arn:aws:ec2:*:*:network-interface/*",
                    "arn:aws:ec2:*:*:security-group/*",
                    "arn:aws:ec2:*:*:key-pair/*",
                    "arn:aws:ec2:*:*:volume/*",
                ],
            },
        ],
    }


ANYSCALE_IAM_POLICY_NAME_SERVICE_STEADY_STATE = (
    "Anyscale_IAM_Policy_Service_Steady_State"
)
ANYSCALE_IAM_PERMISSIONS_SERVICE_STEADY_STATE = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CFN",
            "Effect": "Allow",
            "Action": [
                # CRUD for stack resources
                "cloudformation:TagResource",
                "cloudformation:UntagResource",
                "cloudformation:CreateStack",
                "cloudformation:UpdateStack",
                "cloudformation:DeleteStack",
                "cloudformation:DescribeStackEvents",
                "cloudformation:DescribeStackResources",
                "cloudformation:DescribeStacks",
                "cloudformation:GetTemplate",
            ],
            "Resource": "*",
        },
        {
            "Sid": "EC2Describe",
            "Action": ["ec2:DescribeVpcs", "ec2:DescribeInternetGateways"],
            "Effect": "Allow",
            "Resource": "*",
        },
        {
            "Sid": "ELBDescribe",
            "Effect": "Allow",
            "Action": [
                # Describe ELB state information
                "elasticloadbalancing:DescribeListeners",
                "elasticloadbalancing:DescribeLoadBalancers",
                "elasticloadbalancing:DescribeLoadBalancerAttributes",
                "elasticloadbalancing:DescribeRules",
                "elasticloadbalancing:DescribeTargetGroups",
                "elasticloadbalancing:DescribeTargetGroupAttributes",
                "elasticloadbalancing:DescribeTargetHealth",
                "elasticloadbalancing:DescribeListenerCertificates",
                "elasticloadbalancing:DescribeTags",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ELBCerts",
            "Effect": "Allow",
            "Action": [
                # Add/Remove ELB listener certs
                "elasticloadbalancing:AddListenerCertificates",
                "elasticloadbalancing:RemoveListenerCertificates",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ACMList",
            "Effect": "Allow",
            "Action": [
                # List ELB certs
                "acm:ListCertificates"
            ],
            "Resource": "*",
        },
        {
            "Sid": "ACM",
            "Effect": "Allow",
            "Action": [
                # Manage ACM
                "acm:DeleteCertificate",
                "acm:RenewCertificate",
                "acm:RequestCertificate",
                "acm:AddTagsToCertificate",
                "acm:DescribeCertificate",
                "acm:GetCertificate",
                "acm:ListTagsForCertificate",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ELBWrite",
            "Effect": "Allow",
            "Action": [
                # Modify ELB
                "elasticloadbalancing:AddTags",
                "elasticloadbalancing:RemoveTags",
                "elasticloadbalancing:CreateRule",
                "elasticloadbalancing:ModifyRule",
                "elasticloadbalancing:DeleteRule",
                "elasticloadbalancing:SetRulePriorities",
                "elasticloadbalancing:CreateListener",
                "elasticloadbalancing:ModifyListener",
                "elasticloadbalancing:DeleteListener",
                "elasticloadbalancing:CreateLoadBalancer",
                "elasticloadbalancing:DeleteLoadBalancer",
                "elasticloadbalancing:ModifyLoadBalancerAttributes",
                "elasticloadbalancing:CreateTargetGroup",
                "elasticloadbalancing:ModifyTargetGroup",
                "elasticloadbalancing:DeleteTargetGroup",
                "elasticloadbalancing:ModifyTargetGroupAttributes",
                "elasticloadbalancing:RegisterTargets",
                "elasticloadbalancing:DeregisterTargets",
                "elasticloadbalancing:SetIpAddressType",
                "elasticloadbalancing:SetSecurityGroups",
                "elasticloadbalancing:SetSubnets",
            ],
            "Resource": "*",
        },
        {
            "Sid": "CreateELBServiceLinkedRole",
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "iam:AWSServiceName": "elasticloadbalancing.amazonaws.com"
                }
            },
        },
    ],
}

ANYSCALE_IAM_POLICY_NAME_INITIAL_RUN = "Anyscale_IAM_Policy_Initial_Setup"
ANYSCALE_IAM_PERMISSIONS_EC2_INITIAL_RUN = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SetupEC2",
            "Effect": "Allow",
            "Action": [
                # Anyscale runs this on the first time a cloud is configured.
                "ec2:CreateSecurityGroup",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:AuthorizeSecurityGroupEgress",
                # For configuring VPCs
                "ec2:DescribeVpcs",
                "ec2:CreateVpc",
                "ec2:ModifyVpcAttribute",
                "ec2:CreateVpcEndpoint",
                # Add subnets
                "ec2:CreateSubnet",
                "ec2:ModifySubnetAttribute",
                # Add InternetGateway
                "ec2:CreateInternetGateway",
                "ec2:AttachInternetGateway",
                "ec2:DescribeInternetGateways",
                # Connect InternetGateway to Internet
                "ec2:CreateRouteTable",
                "ec2:AssociateRouteTable",
                "ec2:CreateRoute",
                "ec2:ReplaceRoute",
                # NAT Gateway Setup
                "ec2:CreateNatGateway",
                "ec2:DescribeNatGateways",
            ],
            "Resource": "*",
        },
        {
            "Sid": "CleanupEC2",
            "Effect": "Allow",
            "Action": [
                # Anyscale runs this on the first time a cloud is configured.
                "ec2:DeleteSecurityGroup",
                "ec2:RevokeSecurityGroupIngress",
                "ec2:RevokeSecurityGroupEgress",
                # Remove VPC
                "ec2:DeleteVpc",
                "ec2:DeleteVpcEndpoints",
                # Remove subnets
                "ec2:DeleteSubnet",
                # Remove InternetGateway
                "ec2:DeleteInternetGateway",
                "ec2:DetachInternetGateway",
                # Disconnect InternetGateway
                "ec2:DeleteRouteTable",
                "ec2:DisassociateRouteTable",
                "ec2:DeleteRoute",
                # Remove NATGateway
                "ec2:DeleteNatGateway",
            ],
            "Resource": "*",
        },
    ],
}

# Used for experimental readonly access to SSM
ANYSCALE_SSM_READONLY_ACCESS_POLICY_DOCUMENT = {
    "Version": "2012-10-17",
    "Statement": {
        "Sid": "SecretsManagerReadOnly",
        "Effect": "Allow",
        "Action": [
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
            "secretsmanager:ListSecrets",
        ],
        "Resource": "*",
    },
}

# Used for experimental read and write access to SSM
ANYSCALE_SSM_READ_WRITE_ACCESS_POLICY_DOCUMENT = {
    "Version": "2012-10-17",
    "Statement": {
        "Sid": "SecretsManagerReadWrite",
        "Effect": "Allow",
        "Action": [
            "secretsmanager:CreateSecret",
            "secretsmanager:PutSecretValue",
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
            "secretsmanager:ListSecrets",
        ],
        "Resource": "*",
    },
}


class AnyscaleIAMPolicy(BaseModel):
    parameter_key: str
    parameter_description: str
    resource_logical_id: str
    policy_name: str
    policy_document: str


ANYSCALE_IAM_POLICIES = [
    AnyscaleIAMPolicy(
        parameter_key="AnyscaleCrossAccountIAMPolicySteadyState",
        parameter_description="Steady state IAM policy document",
        resource_logical_id="IAMPermissionEC2SteadyState",
        policy_name=ANYSCALE_IAM_POLICY_NAME_STEADY_STATE,
        policy_document=json.dumps(ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE),
    ),
    AnyscaleIAMPolicy(
        parameter_key="AnyscaleCrossAccountIAMPolicyServiceSteadyState",
        parameter_description="Steady state IAM policy document for services",
        resource_logical_id="IAMPermissionServiceSteadyState",
        policy_name=ANYSCALE_IAM_POLICY_NAME_SERVICE_STEADY_STATE,
        policy_document=json.dumps(ANYSCALE_IAM_PERMISSIONS_SERVICE_STEADY_STATE),
    ),
    AnyscaleIAMPolicy(
        parameter_key="AnyscaleCrossAccountIAMPolicyInitialRun",
        parameter_description="Initial run IAM policy document",
        resource_logical_id="IAMPermissionEC2InitialRun",
        policy_name=ANYSCALE_IAM_POLICY_NAME_INITIAL_RUN,
        policy_document=json.dumps(ANYSCALE_IAM_PERMISSIONS_EC2_INITIAL_RUN),
    ),
]


def get_anyscale_aws_iam_assume_role_policy(
    anyscale_aws_account: str,
) -> Dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AnyscaleControlPlaneAssumeRole",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{anyscale_aws_account}:root"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

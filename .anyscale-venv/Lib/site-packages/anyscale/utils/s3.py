import json
from typing import Any

from botocore.exceptions import ClientError

from anyscale.cli_logger import BlockLogger
from anyscale.util import (  # pylint:disable=private-import
    _coerce_to_list,
    filter_actions_associated_with_role,
    filter_actions_from_policy_document,
)


REQUIRED_S3_PERMISSIONS = {"s3:PutObject", "s3:ListBucket", "s3:GetObject"}


def verify_s3_access(
    boto3_session: Any, s3_bucket: Any, iam_role: Any, logger: BlockLogger
) -> bool:
    """This function verifies that either the specified IAM Role has an identity-based policy
    that grants permission to access the S3 bucket OR the S3 bucket has a resource-based policy
    that grants the IAM Role access.

    NOTE: If the necessary permissions are split between the Role's policy and the Bucket's policy (i.e.
    one has head bucket, the other has get/list objects), this function will return `False`.

    TODO: Check for explicit DENY policies
    TODO: Ensure the bucket is in the same account as the role.

    TODO(#16330): Use AWS Policy Simulator or a similar policy simulation library.
    """
    return (
        # black: no
        _verify_resource_based_s3_access(s3_bucket, iam_role, logger)
        or _verify_identity_based_s3_access(boto3_session, s3_bucket, iam_role, logger)
    )


def _verify_resource_based_s3_access(
    s3_bucket: Any, iam_role: Any, logger: BlockLogger
) -> bool:
    try:
        policy = s3_bucket.Policy().policy
        policy_document = json.loads(policy)
        allow_actions = filter_actions_from_policy_document(
            policy_document,
            lambda statement: statement["Effect"] == "Allow"
            and iam_role.name in str(statement["Principal"]),
        )
        # TODO(#16330) ListBucket must be on `BUCKET_ARN`, while GetObject must be on `BUCKET_ARN/*`
        if len(allow_actions) > 0:
            missing_actions = REQUIRED_S3_PERMISSIONS - allow_actions
            if "s3:*" in allow_actions or len(missing_actions) == 0:
                return True
            logger.info(
                f"Bucket {s3_bucket.name} grants {iam_role.name} some permissions, "
                "but is missing the following permissions:\n[{}]".format(
                    ",".join(missing_actions)
                )
            )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchBucketPolicy":
            raise e

    return False


def _verify_identity_based_s3_access(
    boto3_session: Any, s3_bucket: Any, iam_role: Any, logger: BlockLogger
) -> bool:
    # TODO(#16330) ListBucket must be on `BUCKET_ARN`, while GetObject must be on `BUCKET_ARN/*`
    allow_actions_on_role = filter_actions_associated_with_role(
        boto3_session,
        iam_role,
        lambda statement: statement["Effect"] == "Allow"
        and (
            "*" in _coerce_to_list(statement.get("Resource"))
            or s3_bucket.name in str(statement.get("Resource"))
        ),
    )
    if len(allow_actions_on_role) == 0:
        return False
    missing_actions = REQUIRED_S3_PERMISSIONS - allow_actions_on_role
    if "s3:*" in allow_actions_on_role or len(missing_actions) == 0:
        return True
    logger.info(
        f"Role {iam_role.name} has some permissions to access {s3_bucket.name}, "
        "but is missing the following permissions:\n[{}]".format(
            ",".join(missing_actions)
        )
    )
    return False

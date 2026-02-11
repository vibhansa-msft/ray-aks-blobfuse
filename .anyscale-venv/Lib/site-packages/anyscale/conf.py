import os
from typing import Optional


AWS_PROFILE = None


# Global variable that contains the server session token.
CLI_TOKEN: Optional[str] = None

TEST_V2 = False


ANYSCALE_IAM_ROLE_NAME = "anyscale-iam-role"

# Minimum default Ray version to return when a user either asks for `anyscale.connect required_ray_version`
# or when a Default Cluster Env is used
# TODO(ilr/nikita) Convert this to a backend call for the most recent Ray Version in the Dataplane!
MINIMUM_RAY_VERSION = "2.7.0"

IDLE_TIMEOUT_DEFAULT_MINUTES = 120

ROOT_DIR_PATH = os.path.dirname(__file__)

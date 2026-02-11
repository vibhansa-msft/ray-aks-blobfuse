import json
import os
import time


IDLE_TERMINATION_DIR = os.environ.get(
    "IDLE_TERMINATION_DIR", "/tmp/anyscale/idle_termination_reports"
)


def get_activity(file_path: str, reason: str):
    """
    A generic method to parse a JSON file and returns an object compatible with the Component Activity endpoint in Ray.

    The input file is of the following format
    {
        error: "some optional error string, if present the last_activity_timestamp is ignored",
        last_activity_timestamp: 1666226726.043797
    }

    If an error string exists, the is_active state is set to error with the error string is passed as a reason. Otherwise, the
    last_activity_timestamp & reason are passed as it, and is_active is set to INACTIVE.

    Parameters
    ----------
    file_path: str
    The file path of the activity file.

    reason: str
    The reason for the activity if the file exists.

    Returns
    -------
    dict[str, Any]
    The activity object
    """

    # If the file is not there return error
    if not os.path.exists(file_path):
        return {
            "is_active": "ERROR",
            "reason": "status file does not exist",
            "timestamp": time.time(),
        }
    try:
        with open(file_path) as f:
            result = json.loads(f.read())
            # check if an error exists
            error = result.get("error")
            if error:
                return {
                    "is_active": "ERROR",
                    "reason": error,
                    "timestamp": time.time(),
                }
            else:
                return {
                    "is_active": "INACTIVE",
                    "reason": reason,
                    "timestamp": time.time(),
                    "last_activity_at": result["last_activity_timestamp"],
                }
    except Exception:  # noqa: BLE001
        return {
            "is_active": "ERROR",
            "reason": "error parsing the status file",
            "timestamp": time.time(),
        }


def env_hook():
    """
    This is to be used by the RAY_CLUSTER_ACTIVITY_HOOK in order to report additional activities
    via the /api/component_activities API.
    """
    return {
        "workspace": get_activity(
            f"{IDLE_TERMINATION_DIR}/workspace.json", "workspace last snapshot"
        ),
        "web_terminal": get_activity(
            f"{IDLE_TERMINATION_DIR}/web_terminal.json", "web terminal command finished"
        ),
    }

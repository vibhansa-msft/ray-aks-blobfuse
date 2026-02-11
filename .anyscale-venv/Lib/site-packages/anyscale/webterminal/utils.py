from datetime import datetime, timezone
import os
import pathlib
import re
import shutil
from typing import Any, List, Optional, Tuple

from anyscale.shared_anyscale_utils.util import execution_log_name
from anyscale.shared_anyscale_utils.utils.id_gen import generate_id, IDTypes


# *_DELIMITER are used within preexec / precmd zsh hooks to so that
# we can parse out command information for logging commands, their outputs
# and their statuses.
START_COMMAND_DELIMITER = "71DbDeuXGpbEn8KO93V4kH56xr992Zu3RoUAW0lWesqPWCFff9PVr1RE"
START_OUTPUT_DELIMITER = "YJipyPN4Quh41VbKKHoYqS6bUzw8d0soTo8W61jeBTQUm9F4IRZvQoqg"
END_OUTPUT_DELIMITER = "jjVvK9ydZGXScXs5G89zgqVtwCSYMGdmV2Za9VaTVV3jraSP6hCB5F"
END_COMMAND_DELIMITER = "FwCsfKV6Xfg7PbUJD9sJfZWRBgfAG9uQBj4BBx6uWnss5k2HA3VXtuwb"

ANYSCALE_PREFIX = "7JexUxYaAJfstcUh5rN5CD3nAP24FbpfDAP4Uk8NP879H2N2CntNKk"
CMD_MATCH_TEXT = f"{ANYSCALE_PREFIX}_command="
RET_STATUS_MATCH_TEXT = f"{ANYSCALE_PREFIX}_return_status="
BASH_PREEXEC = "/tmp/bash-preexec.sh"
BASH_PREEXEC_CONFIG = "/tmp/bash-preexec-config.sh"

# The default bashrc file that will be used by our bash terminal.
# This includes output delimiters
bashrc_preexec_content = """
source {BASH_PREEXEC}
print_start_output_delimiter() {{
  echo "{START_COMMAND_DELIMITER}"
  # the command
  echo {CMD_MATCH_TEXT}$1
  echo "{START_OUTPUT_DELIMITER}"
}}

print_end_output_delimiter() {{
  # save the command exit code, then emit it.
  RET_STAT=$?
  echo "{END_OUTPUT_DELIMITER}"
  echo {RET_STATUS_MATCH_TEXT}$RET_STAT
  echo "{END_COMMAND_DELIMITER}"
}}
preexec_functions=(print_start_output_delimiter)
precmd_functions=(print_end_output_delimiter)

""".format(
    START_COMMAND_DELIMITER=START_COMMAND_DELIMITER,
    START_OUTPUT_DELIMITER=START_OUTPUT_DELIMITER,
    END_COMMAND_DELIMITER=END_COMMAND_DELIMITER,
    END_OUTPUT_DELIMITER=END_OUTPUT_DELIMITER,
    CMD_MATCH_TEXT=CMD_MATCH_TEXT,
    RET_STATUS_MATCH_TEXT=RET_STATUS_MATCH_TEXT,
    BASH_PREEXEC=BASH_PREEXEC,
)


def configure_bash_preexec():
    """
    Stores the bash-preexec support which will allow us to parse commands and submit them to anyscale backend.
    """
    dir = pathlib.Path(__file__).parent.resolve()  # noqa: A001
    shutil.copyfile(os.path.join(dir, "bash-preexec.sh"), BASH_PREEXEC)

    with open(BASH_PREEXEC_CONFIG, "w") as f:
        f.write(bashrc_preexec_content)


class Command:
    def __init__(  # noqa: PLR0913
        self,
        term_name: str,
        scid: str = "",
        finished: bool = False,
        output: str = "",
        command: str = "",
        status_code: Optional[int] = None,
        exec_command: bool = False,
    ):
        self.scid = scid
        self.finished = finished
        self.output = output
        self.command = command
        self.status_code = status_code
        self.created_at = datetime.now(timezone.utc)
        self.finished_at = datetime.now(timezone.utc) if finished else None
        self.term_name = term_name
        self.exec_command = exec_command

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, other.__class__):
            return bool(
                self.finished == other.finished
                and self.output == other.output
                and self.command == other.command
                and self.status_code == other.status_code
            )
        return False

    __hash__ = None  # type: ignore

    def finish(self, status_code: int) -> None:
        self.status_code = status_code
        self.finished = True
        self.finished_at = datetime.now(timezone.utc)


def extract_commands(
    stdout: str, last_command: Optional[Command], term_name: str
) -> Tuple[str, List[Command]]:
    """
    extract_commands parse the stdout into commands.
    It also removes any emitted anyscale values like delimiters.
    This functions as follows:
    -- split out all delimiters. All anyscale values are wrapped
    by delimiters so they will be a separate value in the resulting array
    for example:
        "DELIM CMD DELIM STDOUT STDOUT DELIM RET_VAL DELIM" =>
        ["CMD", "STDOUT STDOUT", "RET_VAL"]
    -- for each value
    -- If its a command: Create a new CMD
    -- If its a return status: Mark the last CMD as finished if its not
    -- If its output: Associate with any running commands

    Assumption:
    -- Both DELIMITERS wrapping an emitted value will always be present.
    e.g. "START_CMD cmd START_OUTPUT" will be emitted togeter. This is empirically
    how zsh emits the precmd / preexec function fortunately.
    """
    values = re.split(
        f"{START_COMMAND_DELIMITER}\r?\n?|{START_OUTPUT_DELIMITER}\r?\n?|{END_COMMAND_DELIMITER}\r?\n?|{END_OUTPUT_DELIMITER}\r?\n?",
        stdout,
    )
    output_to_user = ""
    commands = [last_command] if last_command is not None else []
    for val in values:
        if CMD_MATCH_TEXT in val:
            # remove white space from end of command. Then remove the cmd_match_text prefix.
            cmd_text = val.rstrip().split(CMD_MATCH_TEXT)[-1]
            command = Command(
                term_name=term_name,
                command=cmd_text,
                scid=generate_id(IDTypes.session_commands),
            )
            commands.append(command)
        elif RET_STATUS_MATCH_TEXT in val:
            # We emit a return status even for entered new lines
            # So check to make sure there is a command running
            if len(commands) > 0 and commands[-1].finished is False:
                status_code = val.rstrip().split(RET_STATUS_MATCH_TEXT)[-1]
                cmd = commands[-1]
                cmd.finish(int(status_code))
        else:
            output_to_user += val
            # If there is a running command, add this output to it to be logged.
            if len(commands) > 0 and commands[-1].finished is False:
                cmd = commands[-1]
                cmd.output += val
    return (output_to_user, commands)


# log_commands effectively flushes the commands' output
# to their respective logs.
# Then, resets the command output value to an empty string
# Add then logs the value.
def log_commands(commands: List[Command]) -> None:
    for c in commands:
        log_name = f"{execution_log_name(c.scid)}.out"
        output = c.output
        log_output(log_name, output)
        c.output = ""


# log_output writes output to a file
# it used by log_commands
def log_output(file: str, output: str) -> None:
    with open(file, "a") as log_file:
        log_file.write(output)

from collections import deque
import json
from typing import Any, cast, Dict, Optional, Tuple

from terminado import NamedTermManager, TermSocket
import tornado.web

from anyscale.shared_anyscale_utils.utils.id_gen import generate_id, IDTypes
from anyscale.webterminal.command_persister import CommandPersister
from anyscale.webterminal.utils import (
    BASH_PREEXEC_CONFIG,
    configure_bash_preexec,
    extract_commands,
    log_commands,
)


BASH_CMD = f"source {BASH_PREEXEC_CONFIG} && clear\n"
TERMINAL_READ_BUFFER_MAXLEN = 1000


class TermState:
    def __init__(self) -> None:
        self.curr_cmd: Any = None


class AnyscaleTermManager(NamedTermManager):  # type: ignore
    def __init__(self, command_persister: CommandPersister, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # For each terminal, record if it is a debug terminal
        # (keyed by terminal name).
        self.command_persister = command_persister
        self.is_debugger: Dict[str, bool] = {}
        self.is_exec_command: Dict[str, Optional[str]] = {}
        self.terminal_state: Dict[str, TermState] = {}

    def clean_up_terminal(self, term_name: str) -> None:
        if term_name in self.terminals:
            del self.terminals[term_name]
        del self.terminal_state[term_name]
        del self.is_debugger[term_name]
        del self.is_exec_command[term_name]

    def get_terminal(
        self,
        term_name: str,
        is_debugger: bool = False,
        exec_command_id: Optional[str] = None,
    ) -> Any:
        terminal = super().get_terminal(term_name)
        self.is_debugger[term_name] = is_debugger
        self.is_exec_command[term_name] = exec_command_id
        if term_name not in self.terminal_state:
            # We're requesting a new terminal
            self.terminal_state[term_name] = TermState()
        return terminal

    def new_named_terminal(self, **kwargs: Any) -> Tuple[str, Any]:
        name, term = super().new_named_terminal(**kwargs)
        self.is_debugger[name] = False
        self.is_exec_command[name] = None
        self.terminal_state[name] = TermState()
        term.ptyproc.write(BASH_CMD)

        # Overwrite the read buffer's maxlen to TERMINAL_READ_BUFFER_MAXLEN, so terminal history can be longer.
        # terminado==0.12.1 set read_buffer maxlen to 10, but in newer version, it's set to 1000.
        # We should consider to upgrade terminado in the future and remove the overwriting.
        # The source that control the maxlen is here:
        # https://github.com/jupyter/terminado/blob/f586aca16a70bbb0d36bc5475345022d2b9ca587/terminado/management.py#LL48C14-L48C25
        term.read_buffer = deque([], maxlen=TERMINAL_READ_BUFFER_MAXLEN)

        return name, term

    def terminal_count(self, include_all: bool = False) -> int:
        return sum(
            self.is_exec_command[term_name] is None or include_all
            for term_name in self.terminals
        )

    def pty_read(self, fd: Any, events: Any = None) -> None:  # noqa: ARG002
        """
        Called by the event loop when there is pty data ready to read.

        This is a vendored version from terminado
        https://github.com/jupyter/terminado/blob/master/terminado/management.py#L200
        """
        ptywclients = self.ptys_by_fd[fd]
        try:
            text = ptywclients.ptyproc.read(65536)
            client_list = ptywclients.clients

            ###############################
            # START ANYSCALE SPECIFIC CODE
            ###############################
            term_state = self.terminal_state[ptywclients.term_name]
            output_to_user, commands = extract_commands(
                text, term_state.curr_cmd, ptywclients.term_name
            )
            ptywclients.read_buffer.append(output_to_user)

            if len(commands) > 0:
                last_command = commands[-1]
                term_state.curr_cmd = (
                    last_command if not last_command.finished else None
                )
                if self.is_exec_command[ptywclients.term_name]:
                    # Override the id of the command to use the id the backend created
                    # We only do this for exec command because exec command returns
                    # the command id immediately to the client.
                    last_command.scid = cast(
                        str, self.is_exec_command[ptywclients.term_name]
                    )
                    last_command.exec_command = True
                    if last_command.finished:
                        # If the terminal is a single command terminal and the command is finished
                        # we can clean up the terminal.
                        loop = tornado.ioloop.IOLoop.current()
                        loop.spawn_callback(self.terminate, ptywclients.term_name, True)
                        self.clean_up_terminal(ptywclients.term_name)
            # Mark the command as either WT or SDK
            log_commands(commands)
            for command in commands:
                self.command_persister.enqueue_command(command)
            ###############################
            # END ANYSCALE SPECIFIC CODE
            ###############################

            if not client_list:
                # No one to consume our output: buffer it.
                ptywclients.preopen_buffer.append(output_to_user)
                return
            for client in ptywclients.clients:
                client.on_pty_read(output_to_user)
        except EOFError:
            self.on_eof(ptywclients)
            for client in ptywclients.clients:
                client.on_pty_died()


class AnyscaleTermSocket(TermSocket):  # type: ignore
    def initialize(self, **kwargs: Any) -> None:
        super().initialize(**kwargs)

    def check_origin(self, origin: Any) -> bool:
        if self.application.settings["deploy_environment"] == "test":
            return True
        else:
            return cast(bool, origin == self.application.settings["host"])

    def open(self, url_component=None):
        # call the parent method that handles creation of the terminal
        super().open(url_component)

        preopen_buffer = self.terminal.read_buffer.copy()

        # send the full buffer to the client
        while len(preopen_buffer) > 0:
            self.on_pty_read(preopen_buffer.popleft())


class TerminalPageHandler(tornado.web.RequestHandler):
    """Render the /ttyX pages"""

    async def delete(self, term_name: str) -> None:
        term_manager = self.application.settings["term_manager"]
        command_persister = self.application.settings["command_persister"]
        await term_manager.terminate(term_name, force=True)
        command_persister.kill_running_commands_for_terminal(term_name)
        term_manager.clean_up_terminal(term_name)


class TerminalListHandler(tornado.web.RequestHandler):
    """List active terminals."""

    def get(self) -> None:
        term_manager = self.application.settings["term_manager"]
        include_all = json.loads(
            self.get_query_argument("all_terms", default="false")
            or "false"  # this additional false is added bc mypy complains.
        )
        create_if_none = json.loads(
            self.get_query_argument("create_if_none", default="false")
            or "false"  # this additional false is added bc mypy complains.
        )
        terminals = [
            {
                "id": term_name,
                "debugger": term_manager.is_debugger[term_name],
                "exec_command": term_manager.is_exec_command[term_name] is not None,
            }
            for term_name in term_manager.terminals
            if (term_manager.is_exec_command[term_name] is None or include_all)
        ]
        if term_manager.terminal_count() == 0 and create_if_none:
            term_name, _ = term_manager.new_named_terminal()
            terminals.append(
                {
                    "id": term_name,
                    "debugger": term_manager.is_debugger[term_name],
                    "exec_command": term_manager.is_exec_command[term_name] is not None,
                }
            )
        data = {"terminals": terminals}

        self.write(data)


class NewTerminalHandler(tornado.web.RequestHandler):
    """Create new unused terminal"""

    def get(self) -> None:
        term_name, _ = self.application.settings["term_manager"].new_named_terminal()
        self.write({"id": term_name})


class ExecTerminalHandler(tornado.web.RequestHandler):
    """Create new unused terminal"""

    def prepare(self) -> None:
        # Incorporate request JSON into arguments dictionary.
        if self.request.body:
            try:
                json_data = json.loads(self.request.body)
                self.request.body_arguments = json_data
            except ValueError:
                message = "Unable to parse JSON."
                self.send_error(400, message=message)  # Bad Request

    async def post(self) -> None:
        """
        POST /exec/ accepts { shell_command, command_id }
          - shell_command, the command to run
          - command_id, the id of the session_command entry created on the backend

          returns: { web_terminal_id }
          this is the web_terminal in which this command was run in.
          We can use this to kill the command.
        """
        # if default is not specified, the argument is considered required and an exeception will
        # be raised by tornado.
        shell_command = self.request.body_arguments.get("shell_command")
        command_id = self.request.body_arguments.get("command_id")
        term_manager = self.application.settings["term_manager"]
        term_id = generate_id(IDTypes.session_commands)
        term = term_manager.get_terminal(
            term_id, is_debugger=False, exec_command_id=command_id
        )
        term.ptyproc.write(BASH_CMD)

        # Write command
        term.ptyproc.write(f"{shell_command}\n")
        self.write({"web_terminal_tab_id": term_id})


def make_application(
    deploy_environment: str, cwd: str, command_persister: CommandPersister, host: str,
) -> tornado.web.Application:
    configure_bash_preexec()
    term_manager = AnyscaleTermManager(
        command_persister=command_persister,
        shell_command=["bash"],
        max_terminals=100,
        term_settings={"cwd": cwd},
    )
    handlers: Any = [
        (
            r"/webterminal/_websocket/(\w+)",
            AnyscaleTermSocket,
            {"term_manager": term_manager},
        ),
        (r"/webterminal/new/?", NewTerminalHandler),
        (r"/webterminal/list", TerminalListHandler),
        (r"/webterminal/exec", ExecTerminalHandler),
        (r"/webterminal/(\w+)/?", TerminalPageHandler),
    ]
    return tornado.web.Application(
        handlers,
        term_manager=term_manager,
        deploy_environment=deploy_environment,
        host=host,
        command_persister=command_persister,
    )


def main(
    deploy_environment: str,
    use_debugger: bool,  # noqa: ARG001
    cli_token: str,
    host: str,
    working_dir: str,
    session_id: str,
) -> None:
    # Write the .bashrc file before starting the server.
    loop = tornado.ioloop.IOLoop.instance()
    command_persister = CommandPersister(cli_token, host, session_id)
    application = make_application(
        deploy_environment, working_dir, command_persister, host
    )
    port = 8700
    application.listen(port, "localhost")
    print(f"Listening on localhost:{port}")
    loop.spawn_callback(command_persister.persist_commands)

    try:
        loop.start()
    except KeyboardInterrupt:
        print(" Shutting down on SIGINT")
    finally:
        application.settings["term_manager"].shutdown()
        command_persister.shutdown()
        loop.close()

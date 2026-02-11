"""
Telemetry for Anyscale CLI commands.

Patches Click to capture execution metrics for _leaf_ commands,
including command path, flags, timing, and errors. Emits via
HTTP POST (best-effort) or debug print.

Supports session-based distributed tracing for interactive commands:
- Each command gets a unique trace_id for backend correlation
- Interactive sessions get a session_id to group related operations
- Page fetches get new trace_ids but share the session_id
"""

from contextvars import ContextVar
import functools
import json
import os
import random
import secrets
import sys
import threading
import time
from typing import List, Optional

import click

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models.cli_usage_payload import CLIUsagePayload


# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = float(os.getenv("ANYSCALE_TELEMETRY_SAMPLE_RATE", "1.0"))
TELEMETRY_DEBUG = os.getenv("ANYSCALE_DEBUG") == "1"

# ContextVar automatically propagates into asyncio tasks if you ever go async.
# (Each CLI invocation gets its own interpreter, so this never crosses commands.)
_trace_id_var: ContextVar[Optional[str]] = ContextVar("_trace_id_var", default=None)
_session_id_var: ContextVar[Optional[str]] = ContextVar("_session_id_var", default=None)
_skip_click_patch_var: ContextVar[bool] = ContextVar(
    "_skip_click_patch_var", default=False
)

logger = BlockLogger()

# ─── Trace Context Helpers ───────────────────────────────────────────────────


def _setup_trace_context() -> str:
    """Ensure we have a trace ID in the ContextVar, and return it."""
    try:
        tid = _trace_id_var.get()
        if tid is None:
            tid = secrets.token_hex(16)
            _trace_id_var.set(tid)
        logger.debug(f"[TRACE DEBUG] trace-id={tid}")
        return tid
    except Exception:  # noqa: BLE001
        # Fallback to a default trace ID if anything goes wrong
        return secrets.token_hex(16)


def get_traceparent() -> Optional[str]:
    """Return a W3C-style traceparent header, or None if not initialized."""
    try:
        tid = _trace_id_var.get()
        if not tid:
            return None
        return f"00-{tid}-{'0'*16}-01"
    except Exception:  # noqa: BLE001
        return None


def start_interactive_session() -> str:
    """Start an interactive session and return the session ID."""
    try:
        session_id = secrets.token_hex(8)
        _session_id_var.set(session_id)
        logger.debug(f"[TRACE DEBUG] session-id={session_id}")
        return session_id
    except Exception:  # noqa: BLE001
        # Return a fallback session ID
        return secrets.token_hex(8)


def new_trace_for_page() -> str:
    """Generate a new trace ID for the next page in an interactive session."""
    try:
        new_trace_id = secrets.token_hex(16)
        _trace_id_var.set(new_trace_id)
        logger.debug(f"[TRACE DEBUG] new-trace-id={new_trace_id}")
        return new_trace_id
    except Exception:  # noqa: BLE001
        # Return a fallback trace ID
        return secrets.token_hex(16)


# ─── CLI Arg Extraction ───────────────────────────────────────────────────────


def _get_user_flags() -> List[str]:
    """Return all `-x`/`--long` flags from the raw argv (no values)."""
    try:
        args = sys.argv[1:]
        # Strip off the program name if Click added it
        if args and args[0] in ("anyscale", "main"):
            args = args[1:]
        return [a for a in args if a.startswith("-")]
    except Exception:  # noqa: BLE001
        return []


def _get_user_options(ctx: click.Context) -> List[str]:
    """Return the names of parameters explicitly set via the CLI."""
    try:
        opts: List[str] = []
        for name in ctx.params:
            try:
                if (
                    ctx.get_parameter_source(name)
                    is click.core.ParameterSource.COMMANDLINE
                ):
                    opts.append(name)
            except Exception:  # noqa: BLE001
                opts.append(name)
        return opts
    except Exception:  # noqa: BLE001
        return []


# ─── Page Fetch Tracking ─────────────────────────────────────────────────────

_page_fetch_start_time: ContextVar[Optional[float]] = ContextVar(
    "_page_fetch_start_time", default=None
)


def mark_page_fetch_start(page_number: int) -> None:
    """
    Mark the start of a page fetch operation. This will:
    1. Generate a new trace ID for this page
    2. Start timing the fetch operation

    Args:
        page_number: The page number being fetched (1-indexed)
    """
    try:
        if SAMPLE_RATE <= 0 or random.random() > SAMPLE_RATE:
            return

        # Generate new trace ID for this page BEFORE making the API request
        new_trace_for_page()

        # Start timing
        _page_fetch_start_time.set(time.perf_counter())

        logger.debug(f"[TRACE DEBUG] page-fetch-start page={page_number}")
    except Exception:  # noqa: BLE001
        # Telemetry should never crash the CLI
        pass


def mark_page_fetch_complete(page_number: int) -> None:
    """
    Mark the completion of a page fetch operation and emit telemetry.
    This calculates the duration and sends the page_fetch event.

    Args:
        page_number: The page number that was fetched (1-indexed)
    """
    try:
        if SAMPLE_RATE <= 0 or random.random() > SAMPLE_RATE:
            return

        # Calculate duration
        start_time = _page_fetch_start_time.get()
        if start_time is None:  # noqa: SIM108 - comment explains fallback reasoning
            # Fallback if timing wasn't started properly
            duration_ms = 0.0
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000

        # Get current click context
        try:
            ctx = click.get_current_context()
        except RuntimeError:
            return  # No active context

        # Get current trace ID (should be the one we generated in mark_page_fetch_start)
        trace_id = _trace_id_var.get()
        if not trace_id:
            return

        # Emit page fetch telemetry
        body = _create_payload(
            trace_id=trace_id,
            ctx=ctx,
            duration_ms=duration_ms,
            exit_code=0,
            exception_type=None,
            event_type="page_fetch",
            page_number=page_number,
        )
        _emit_telemetry(body)

        # Reset timing
        _page_fetch_start_time.set(None)

        logger.debug(
            f"[TRACE DEBUG] page-fetch-complete page={page_number} duration={duration_ms:.2f}ms"
        )
    except Exception:  # noqa: BLE001
        # Telemetry should never crash the CLI
        pass


# ─── Payload Construction ────────────────────────────────────────────────────


def _create_payload(
    trace_id: str,
    ctx: click.Context,
    duration_ms: float,
    exit_code: int,
    exception_type: Optional[str],
    event_type: str = "command",
    page_number: Optional[int] = None,
) -> CLIUsagePayload:
    """
    Build a Typed CLIUsagePayload (from the generated OpenAPI models)
    so we get IDE/type-checker support on all the fields.

    Args:
        trace_id: Unique trace identifier for this operation
        ctx: Click context containing command information
        duration_ms: Command/operation duration in milliseconds
        exit_code: Command exit code (0 for success, 1 for error)
        exception_type: Exception class name if command failed
        event_type: Type of event ("command" or "page_fetch")
        page_number: Page number for page_fetch events
    """
    try:
        # Get session ID if available
        session_id = _session_id_var.get()

        data = {
            "trace_id": trace_id,
            "session_id": session_id,
            "event_type": event_type,
            "page_number": page_number,
            "cmd_path": ctx.command_path,
            "options": sorted(_get_user_options(ctx)),
            "flags_used": sorted(_get_user_flags()),
            "duration_ms": round(duration_ms, 2),
            "exit_code": exit_code,
            "exception_type": exception_type,
            "cli_version": getattr(sys.modules.get("anyscale"), "__version__", None),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "timestamp": int(time.time()),
        }
        return CLIUsagePayload(**data)
    except Exception:  # noqa: BLE001
        # Fallback payload with minimal data if construction fails
        fallback_data = {
            "trace_id": trace_id,
            "cmd_path": "unknown",
            "duration_ms": duration_ms,
            "exit_code": exit_code,
            "timestamp": int(time.time()),
        }
        return CLIUsagePayload(**fallback_data)


def mark_command_complete() -> None:
    """
    Mark that the command logic has completed and emit telemetry immediately.
    For interactive commands, call this when data is ready but before user interaction.
    This will prevent the Click patch from double-emitting.
    """
    try:
        trace_id = _trace_id_var.get()
        if not trace_id:
            return

        # Get current click context
        try:
            ctx = click.get_current_context()
        except RuntimeError:
            return  # No active context

        # Calculate duration from the click context if available
        # For interactive commands, we want the time up to this point
        start_time = getattr(ctx, "telemetry_start_time", None)
        if start_time is None:  # noqa: SIM108 - comment explains fallback reasoning
            # Fallback: use a minimal duration
            duration_ms = 0.0
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit the command completion event
        body = _create_payload(
            trace_id=trace_id,
            ctx=ctx,
            duration_ms=duration_ms,
            exit_code=0,
            exception_type=None,
            event_type="command",
        )
        _emit_telemetry(body)

        # Prevent Click patch from emitting again
        _skip_click_patch_var.set(True)
    except Exception:  # noqa: BLE001
        # Telemetry should never crash the CLI
        pass


# ─── Emission (fire-&-forget) ─────────────────────────────────────────────────


def _emit_telemetry(body: CLIUsagePayload) -> None:
    """
    Send the payload to the console API. Runs in a short-lived thread
    so we never block the CLI for more than ~3 seconds.
    """
    try:
        logger.debug(json.dumps(body.to_dict(), indent=2))

        traceparent = get_traceparent()

        def _worker():
            try:
                from anyscale.authenticate import (  # noqa: PLC0415 - codex_reason("gpt5.2", "lazy import to avoid auth client setup unless telemetry emits")
                    get_auth_api_client,
                )

                if traceparent:
                    _trace_id_var.set(body.trace_id)

                api = get_auth_api_client().api_client
                api.receive_cli_usage_api_v2_cli_usage_post(
                    cli_usage_payload=body, _request_timeout=2
                )

                logger.debug("[TELEMETRY] POST completed successfully")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[TELEMETRY] POST failed: {e}")
                # Best-effort only - never crash the CLI

        thread = threading.Thread(target=_worker, daemon=False)
        thread.start()
        thread.join(timeout=3)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[TELEMETRY] Failed to emit: {e}")
        # Telemetry should never crash the CLI


# ─── Click Patch ─────────────────────────────────────────────────────────────


def _patch_click() -> None:
    """Monkey-patch Click so that each leaf command emits telemetry."""
    try:
        if getattr(click, "_anyscale_telemetry_patched", False):
            return

        original_invoke = click.Command.invoke

        @functools.wraps(original_invoke)
        def instrumented_invoke(self, ctx, *args, **kwargs):
            if (
                isinstance(self, click.Group)
                or SAMPLE_RATE <= 0
                or random.random() > SAMPLE_RATE
            ):
                return original_invoke(self, ctx, *args, **kwargs)

            try:
                trace_id = _setup_trace_context()
            except Exception:  # noqa: BLE001
                return original_invoke(self, ctx, *args, **kwargs)

            start = time.perf_counter()
            ctx.telemetry_start_time = start
            exit_code, exc_name = 0, None

            try:
                result = original_invoke(self, ctx, *args, **kwargs)
                return result
            except Exception as e:  # noqa: BLE001
                exit_code, exc_name = 1, e.__class__.__name__
                raise
            finally:
                # Only emit telemetry once per command invocation
                if not _skip_click_patch_var.get():
                    try:
                        duration_ms = (time.perf_counter() - start) * 1000
                        body = _create_payload(
                            trace_id=trace_id,
                            ctx=ctx,
                            duration_ms=duration_ms,
                            exit_code=exit_code,
                            exception_type=exc_name,
                            event_type="command",
                        )
                        _emit_telemetry(body)
                        # Prevent Click patch from emitting again
                        _skip_click_patch_var.set(True)
                    except Exception:  # noqa: BLE001
                        pass

        click.Command.invoke = instrumented_invoke
        click._anyscale_telemetry_patched = (  # noqa: SLF001  # type: ignore[attr-defined]
            True
        )
    except Exception:  # noqa: BLE001
        # If patching fails, telemetry just won't work - don't crash the CLI
        pass


# Auto-patch on import
_patch_click()

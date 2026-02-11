import itertools
from json import dumps as json_dumps
from typing import Any, Callable, Dict, Iterator, List, Optional

import click
from rich.console import Console
from rich.table import Table

from anyscale.util import AnyscaleJSONEncoder, validate_non_negative_arg


MAX_PAGE_SIZE = 50
NON_INTERACTIVE_DEFAULT_MAX_ITEMS = 10


def validate_page_size(ctx, param, value):
    """Click callback to validate page size argument."""
    value = validate_non_negative_arg(ctx, param, value)
    if value is not None and value > MAX_PAGE_SIZE:
        raise click.BadParameter(f"must be less than or equal to {MAX_PAGE_SIZE}.")
    return value


def create_table(
    columns: List[tuple[str, Optional[str], bool]], is_first: bool = True
) -> Table:
    """Create a Rich table with specified columns.

    Args:
        columns: List of (column_name, style, no_wrap) tuples.
                 E.g., [("ID", "cyan", True), ("Name", None, False)]
        is_first: Show headers (True for first page, False for subsequent pages).

    Returns:
        Configured Rich Table ready for adding rows.

    Example:
        >>> columns = [("ID", "cyan", True), ("Name", None, False)]
        >>> table = create_table(columns, is_first=True)
        >>> table.add_row("123", "My Job")
    """
    table = Table(show_header=is_first, header_style="bold")
    for name, style, no_wrap in columns:
        table.add_column(name, style=style, no_wrap=no_wrap)
    return table


def _paginate(iterator: Iterator[Any], page_size: Optional[int]) -> Iterator[List[Any]]:
    if page_size is None:
        yield list(iterator)
    else:
        while True:
            page = list(itertools.islice(iterator, page_size))
            if not page:
                return
            yield page


def _render_page(
    page: List[Any],
    item_formatter: Callable[[Any], Dict[str, Any]],
    table_creator: Callable[[bool], Table],
    json_output: bool,
    is_first: bool,
    page_num: int,
    console: Console,
) -> int:
    """Render a single page of items."""
    if page_num > 1:  # Only show page number for pages after first
        console.print(f"[dim]Page {page_num}[/dim]")

    rows = [item_formatter(item) for item in page]
    if json_output:
        json_str = json_dumps(rows, indent=2, cls=AnyscaleJSONEncoder)
        console.print_json(json=json_str)
    else:
        tbl = table_creator(is_first)
        for row in rows:
            tbl.add_row(*row.values())
        console.print(tbl)

    return len(page)


def _should_continue_pagination(
    page_size: int, current_page_size: int, console: Console
) -> bool:
    """Prompt user to continue pagination if needed."""
    if current_page_size < page_size:
        return False  # Last page, no need to prompt

    console.print()
    console.print(
        "[dim]Press [bold]Enter[/bold] to continue, [bold]q[/bold] to quit…[/]"
    )
    return input("> ").strip().lower() != "q"


def display_list(  # noqa: PLR0913, PLR0912
    iterator: Iterator[Any],
    item_formatter: Callable[[Any], Dict[str, Any]],
    table_creator: Callable[[bool], Table],
    json_output: bool,
    page_size: int,
    interactive: bool,
    max_items: Optional[int],
    console: Console,
) -> int:
    """Displays a list of items from an iterator, handling pagination and output format.

    Args:
        iterator: The iterator yielding items to display.
        item_formatter: A callable that takes an item and returns a dictionary
            representing the row data (for table) or the JSON object.
        table_creator: A callable that takes a boolean (is_first_page) and
            returns a rich.Table instance. Used only if json_output is False.
        json_output: If True, output items as a JSON list. Otherwise, display
            them in a table created by table_creator.
        page_size: The number of items to display per page in interactive mode.
        interactive: If True, enables interactive pagination. If False, displays
            up to max_items (or all items if max_items is None) without prompting.
        max_items: The maximum total number of items to display when interactive
            is False. If None, all items are displayed.
        console: The rich.Console object to use for output.

    Returns:
        The total number of items displayed.
    """
    total_count = 0
    pages = _paginate(iterator, page_size if interactive else max_items)

    # Start interactive session if needed
    if interactive:
        try:
            from anyscale.telemetry import (  # noqa: PLC0415 - codex_reason("gpt5.2", "lazy import to avoid telemetry startup unless interactive")
                start_interactive_session,
            )

            start_interactive_session()
        except Exception:  # noqa: BLE001
            pass

    # Fetch and render first page
    with console.status("Retrieving items…", spinner="dots"):
        try:
            first_page = next(pages)
        except StopIteration:
            first_page = []

    if first_page:
        total_count += _render_page(
            first_page, item_formatter, table_creator, json_output, True, 1, console
        )

    # For interactive commands, mark when command logic completes
    if interactive:
        try:
            from anyscale.telemetry import (  # noqa: PLC0415 - codex_reason("gpt5.2", "lazy import to avoid telemetry startup unless interactive")
                mark_command_complete,
            )

            mark_command_complete()
        except Exception:  # noqa: BLE001
            pass

    # Non-interactive: stop after first page
    if not interactive:
        return total_count

    # Interactive: check if user wants to continue
    if not _should_continue_pagination(page_size, len(first_page), console):
        return total_count

    # Render remaining pages with correct telemetry timing
    page_num = 2
    while True:
        # Start page fetch timing and generate new trace ID BEFORE fetching
        try:
            from anyscale.telemetry import (  # noqa: PLC0415 - codex_reason("gpt5.2", "lazy import to avoid telemetry startup unless interactive")
                mark_page_fetch_start,
            )

            mark_page_fetch_start(page_num)
        except Exception:  # noqa: BLE001
            pass

        # Now fetch the page (with the new trace ID)
        try:
            page = next(pages)
        except StopIteration:
            break

        # Render the page
        total_count += _render_page(
            page, item_formatter, table_creator, json_output, False, page_num, console
        )

        # Complete page fetch telemetry
        try:
            from anyscale.telemetry import (  # noqa: PLC0415 - codex_reason("gpt5.2", "lazy import to avoid telemetry startup unless interactive")
                mark_page_fetch_complete,
            )

            mark_page_fetch_complete(page_num)
        except Exception:  # noqa: BLE001
            pass

        # Check if user wants to continue or if this was the last page
        if not _should_continue_pagination(page_size, len(page), console):
            break

        page_num += 1

    return total_count

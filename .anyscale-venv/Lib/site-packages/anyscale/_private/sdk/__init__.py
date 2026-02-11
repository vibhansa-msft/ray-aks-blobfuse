from datetime import date, datetime
from functools import wraps
import sys
from typing import Callable, Dict, Optional, Type, TypeVar, Union

import colorama
from typing_extensions import ParamSpec


P = ParamSpec("P")
T = TypeVar("T")


_LAZY_SDK_SINGLETONS: Dict[str, Callable] = {}


def sdk_command(
    key: str, sdk_cls: Type, *, doc_py_example: str, arg_docstrings: Dict[str, str],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically inject an `_sdk` arg into the wrapped function.

    The arguments to this class are a unique key for the singleton and its type
    (the constructor will be called with no arguments).
    """

    # The P and T type hints allow f's type hints to pass through this decorator.
    # Without them, f's type hints would not be visible to the developer.
    # See https://github.com/anyscale/product/pull/27738.
    def _inject_typed_sdk_singleton(f: Callable[P, T]) -> Callable[P, T]:
        if not doc_py_example:
            raise ValueError(
                f"SDK command '{f.__name__}' must provide a non-empty 'doc_py_example'."
            )

        # TODO: validate docstrings.

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # We disable the mypy linter here because it treats kwargs as a
            # P.kwargs object. mypy wrongly thinks kwargs can't be indexed.
            if "_private_sdk" not in kwargs:  # type: ignore
                if key not in _LAZY_SDK_SINGLETONS:
                    _LAZY_SDK_SINGLETONS[key] = sdk_cls()

                kwargs["_private_sdk"] = _LAZY_SDK_SINGLETONS[key]  # type: ignore

            return f(*args, **kwargs)

        # TODO(edoakes): move to parsing docstrings instead.
        wrapper.__doc_py_example__ = doc_py_example  # type: ignore
        wrapper.__arg_docstrings__ = arg_docstrings  # type: ignore

        return wrapper

    return _inject_typed_sdk_singleton


def sdk_docs(
    *, doc_py_example: str, arg_docstrings: Dict[str, str],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add documentation for an SDK command."""

    # The P and T type hints allow f's type hints to pass through this decorator.
    # Without them, f's type hints would not be visible to the developer.
    # See https://github.com/anyscale/product/pull/27738.
    def _add_doc_magic_attrs(f: Callable[P, T]) -> Callable[P, T]:
        if not doc_py_example:
            raise ValueError(
                f"SDK command '{f.__name__}' must provide a non-empty 'doc_py_example'."
            )

        # TODO(edoakes): validate docstrings.
        # TODO(edoakes): move to parsing docstrings instead.
        f.__doc_py_example__ = doc_py_example  # type: ignore
        f.__arg_docstrings__ = arg_docstrings  # type: ignore

        return f

    return _add_doc_magic_attrs


def sdk_command_v2(
    *, doc_py_example: str, arg_docstrings: Dict[str, str],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Similar to `@sdk_command`, but relies on the SDK function initializing `sdk()` in the function body.

    Decorates the function with the provided `doc_py_example` and `arg_docstrings`.
    """

    def _wrap_sdk_function(f: Callable[P, T]) -> Callable[P, T]:
        if not doc_py_example:
            raise ValueError(
                f"SDK command '{f.__name__}' must provide a non-empty 'doc_py_example'."
            )

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return f(*args, **kwargs)

        wrapper.__doc_py_example__ = doc_py_example  # type: ignore
        wrapper.__arg_docstrings__ = arg_docstrings  # type: ignore

        return wrapper

    return _wrap_sdk_function


def deprecated_sdk_command(
    key: str,
    sdk_cls: Type,
    *,
    doc_py_example: str,
    arg_docstrings: Dict[str, str],
    deprecation_message: Optional[str] = None,
    removal_date: Optional[Union[str, date, datetime]] = None,
    alternative: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """SDK command decorator with deprecation warning support.

    Similar to sdk_command but adds deprecation warnings when the function is called.

    Args:
        key: Unique key for the SDK singleton
        sdk_cls: SDK class type
        doc_py_example: Documentation example
        arg_docstrings: Argument documentation
        deprecation_message: Custom deprecation message
        removal_date: When the command will be removed (YYYY-MM-DD string, date, or datetime)
        alternative: Suggested alternative command
    """

    def _inject_typed_sdk_singleton_with_deprecation(
        f: Callable[P, T]
    ) -> Callable[P, T]:
        if not doc_py_example:
            raise ValueError(
                f"SDK command '{f.__name__}' must provide a non-empty 'doc_py_example'."
            )

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # noqa: PLR0912
            # Print a visual separator for attention
            print("\n" + "=" * 80, file=sys.stderr)
            print(
                f"{colorama.Style.BRIGHT}{colorama.Fore.YELLOW}⚠️  SDK DEPRECATION WARNING ⚠️{colorama.Style.RESET_ALL}",
                file=sys.stderr,
            )
            print("=" * 80, file=sys.stderr)

            # Build deprecation message
            base_msg = deprecation_message or f"Function '{f.__name__}' is deprecated"

            # Add removal date information with grammar-aware connector
            date_msg = None
            if removal_date:
                date_str = None
                try:
                    if isinstance(removal_date, str):
                        parsed_date = datetime.strptime(removal_date, "%Y-%m-%d").date()
                    elif isinstance(removal_date, datetime):
                        parsed_date = removal_date.date()
                    elif isinstance(removal_date, date):
                        parsed_date = removal_date
                    else:
                        parsed_date = None

                    if parsed_date:
                        date_str = parsed_date.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    date_str = str(removal_date)

                if date_str:
                    ends_with_punct = base_msg.strip().endswith((".", "!", "?"))
                    if ends_with_punct:
                        date_msg = f"It will be removed on {date_str}"
                    else:
                        date_msg = f"and will be removed on {date_str}"

            # Add alternative suggestion
            alternative_msg = None
            if alternative:
                alternative_msg = f"\n\n➡️  {colorama.Style.BRIGHT}Please {alternative}{colorama.Style.RESET_ALL}"

            msg_parts = {
                "deprecation_message": base_msg,
                "date_msg": date_msg,
                "alternative_msg": alternative_msg,
            }

            # Join the main line with a space, then append the alternative on its own line
            main_line_parts = [
                part
                for part in [msg_parts["deprecation_message"], msg_parts["date_msg"]]
                if part
            ]
            deprecation_msg = " ".join(main_line_parts)
            if msg_parts["alternative_msg"]:
                deprecation_msg += msg_parts["alternative_msg"]

            # Print the warning
            print(
                f"\n{colorama.Fore.YELLOW}{deprecation_msg}{colorama.Style.RESET_ALL}",
                file=sys.stderr,
            )
            print("=" * 80 + "\n", file=sys.stderr)

            # Then handle SDK injection like normal sdk_command
            if "_private_sdk" not in kwargs:  # type: ignore
                if key not in _LAZY_SDK_SINGLETONS:
                    _LAZY_SDK_SINGLETONS[key] = sdk_cls()

                kwargs["_private_sdk"] = _LAZY_SDK_SINGLETONS[key]  # type: ignore

            return f(*args, **kwargs)

        wrapper.__doc_py_example__ = doc_py_example  # type: ignore
        wrapper.__arg_docstrings__ = arg_docstrings  # type: ignore
        wrapper.__deprecated__ = True  # type: ignore
        wrapper.__deprecation_message__ = deprecation_message  # type: ignore
        wrapper.__removal_date__ = removal_date  # type: ignore
        wrapper.__alternative__ = alternative  # type: ignore

        return wrapper

    return _inject_typed_sdk_singleton_with_deprecation

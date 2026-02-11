from functools import wraps
from typing import Any, Callable, Optional
import warnings


def deprecated(message: Optional[str] = None) -> Callable[..., Any]:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    From https://stackoverflow.com/a/30253848
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def decorated_func(*args, **kwargs) -> Any:
            warnings.simplefilter("always", DeprecationWarning)  # turn on filter
            deprecated_message = (
                message
                if message
                else f"Call to deprecated function `{func.__name__}`."
            )
            # stacklevel=2 logs at the caller level of func
            # eg. DeprecationWarning: Call to deprecated function `foo()`
            #   foo()
            warnings.warn(deprecated_message, DeprecationWarning, stacklevel=2)
            warnings.filters = warnings.filters[1:]  # reset filter
            return func(*args, **kwargs)

        return decorated_func

    return decorator

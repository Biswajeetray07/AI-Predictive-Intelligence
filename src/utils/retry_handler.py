"""
Retry Handler Module
====================
Decorator-based retry mechanism with exponential backoff
for resilient API interactions.
"""

import time
import logging
import functools
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_failure: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for sleep duration between retries.
        exceptions: Tuple of exception types to catch and retry on.
        on_failure: Optional callback invoked on each failure (exception, attempt).

    Returns:
        Decorated function with retry logic.

    Example:
        @retry(max_retries=3, backoff_factor=2.0)
        def fetch_data(url):
            return requests.get(url).json()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if on_failure:
                        on_failure(e, attempt)

                    if attempt < max_retries:
                        sleep_time = backoff_factor ** (attempt - 1)
                        logger.warning(
                            f"[{func.__name__}] Attempt {attempt}/{max_retries} "
                            f"failed: {e}. Retrying in {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"[{func.__name__}] All {max_retries} attempts "
                            f"exhausted. Last error: {e}"
                        )

            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def retry_call(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """
    Retry a function call without using the decorator pattern.

    Args:
        func: The function to call.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        max_retries: Maximum retry attempts.
        backoff_factor: Backoff multiplier.
        exceptions: Exception types to catch.

    Returns:
        The function's return value on success.

    Raises:
        The last exception if all retries fail.
    """
    if kwargs is None:
        kwargs = {}

    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                sleep_time = backoff_factor ** (attempt - 1)
                logger.warning(
                    f"[{func.__name__}] Attempt {attempt}/{max_retries} "
                    f"failed: {e}. Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    f"[{func.__name__}] All {max_retries} attempts exhausted."
                )

    raise last_exception  # type: ignore[misc]

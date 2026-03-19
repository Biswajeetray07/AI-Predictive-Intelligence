"""
Retry Handler
==============
Decorators and helpers for retrying failed API calls with exponential backoff.
"""

import time
import logging
import functools
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable] = None,
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries (seconds).
        backoff_factor: Multiply delay by this factor after each retry.
        exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback(attempt, exception) called before each retry.

    Usage:
        @retry(max_retries=3, delay=1.0)
        def fetch_data(url):
            return requests.get(url).json()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_retries + 2):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt > max_retries:
                        break

                    logger.warning(
                        f"{func.__name__}: attempt {attempt} failed ({e}). "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            logger.error(f"{func.__name__}: all {max_retries} retries exhausted.")
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__}: failed after {max_retries} retries")

        return wrapper
    return decorator


def retry_call(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
):
    """
    Call a function with retry logic (non-decorator version).

    Args:
        func: Function to call.
        args: Positional arguments.
        kwargs: Keyword arguments.
        max_retries: Maximum retries.
        delay: Initial delay.
        backoff_factor: Backoff multiplier.
        exceptions: Retryable exception types.

    Returns:
        Function return value.
    """
    kwargs = kwargs or {}
    current_delay = delay
    last_exception = None

    for attempt in range(1, max_retries + 2):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt > max_retries:
                break
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {current_delay:.1f}s...")
            time.sleep(current_delay)
            current_delay *= backoff_factor

    if last_exception:
        raise last_exception
    raise RuntimeError(f"{func.__name__}: failed after {max_retries} retries")

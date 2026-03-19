"""
Rate Limiter Module
===================
Prevents API quota violations through token-bucket style rate limiting
with dynamic sleep capabilities.
"""

import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter that enforces maximum API calls
    within a configurable time window.

    Thread-safe implementation for concurrent collector usage.
    """

    def __init__(
        self,
        max_calls: int = 10,
        period: float = 60.0,
        name: str = "default",
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period.
            period: Time window in seconds.
            name: Identifier for logging purposes.
        """
        self.max_calls = max_calls
        self.period = period
        self.name = name
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """
        Acquire permission to make an API call.
        Blocks (sleeps) if the rate limit has been reached.
        """
        with self._lock:
            now = time.time()
            # Purge expired timestamps
            self._calls = [
                ts for ts in self._calls if now - ts < self.period
            ]

            if len(self._calls) >= self.max_calls:
                oldest = self._calls[0]
                sleep_time = self.period - (now - oldest)
                if sleep_time > 0:
                    logger.info(
                        f"[{self.name}] Rate limit reached "
                        f"({self.max_calls}/{self.period}s). "
                        f"Sleeping {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)

            self._calls.append(time.time())

    def wait(self, seconds: Optional[float] = None) -> None:
        """
        Dynamic sleep for custom throttling.

        Args:
            seconds: Duration to sleep. If None, sleeps for
                     period / max_calls (even spacing).
        """
        if seconds is None:
            seconds = self.period / self.max_calls
        logger.debug(f"[{self.name}] Dynamic wait: {seconds:.2f}s")
        time.sleep(seconds)

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"RateLimiter(name='{self.name}', "
            f"max_calls={self.max_calls}, period={self.period}s)"
        )

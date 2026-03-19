"""
Rate Limiter
=============
Token-bucket rate limiter to prevent exceeding API quotas.
"""

import time
import threading
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter for API call throttling.

    Usage:
        limiter = RateLimiter(calls_per_second=5)
        for url in urls:
            limiter.wait()
            requests.get(url)
    """

    def __init__(self, calls_per_second: float = 1.0, burst_size: int = 1):
        """
        Args:
            calls_per_second: Maximum sustained request rate.
            burst_size: Maximum burst of requests allowed.
        """
        self.rate = calls_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_time = time.monotonic()
        self._lock = threading.Lock()

    def wait(self):
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.last_time = now

            # Replenish tokens
            self.tokens += elapsed * self.rate
            if self.tokens > self.burst_size:
                self.tokens = float(self.burst_size)

            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                logger.debug(f"Rate limiter sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

    def reset(self):
        """Reset the rate limiter state."""
        with self._lock:
            self.tokens = float(self.burst_size)
            self.last_time = time.monotonic()


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that backs off when 429 errors are detected.
    """

    def __init__(self, calls_per_second: float = 1.0, burst_size: int = 1):
        super().__init__(calls_per_second, burst_size)
        self.backoff_multiplier = 1.0

    def report_rate_limit(self):
        """Call this when a 429 response is received."""
        with self._lock:
            self.backoff_multiplier = min(self.backoff_multiplier * 2.0, 32.0)
            logger.warning(f"Rate limit hit. Backoff multiplier: {self.backoff_multiplier}x")

    def report_success(self):
        """Gradually recover from backoff on successful requests."""
        with self._lock:
            self.backoff_multiplier = max(self.backoff_multiplier * 0.9, 1.0)

    def wait(self):
        """Block with adaptive delay based on backoff state."""
        with self._lock:
            delay = (1.0 / self.rate) * self.backoff_multiplier
        time.sleep(delay)

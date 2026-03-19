"""
Async HTTP Fetcher
==================
Concurrent HTTP client using aiohttp for high-throughput API collection.
Supports rate limiting, retries, and batch operations.
"""

import asyncio
import aiohttp
import logging
from typing import List, Optional, Dict, Any, Tuple, cast

logger = logging.getLogger(__name__)


class AsyncFetcher:
    """Async HTTP fetcher with concurrency control and rate limiting."""

    def __init__(
        self,
        max_concurrency: int = 20,
        rate_limit_delay: float = 0.1,
        timeout: int = 15,
        max_retries: int = 3,
    ):
        self.max_concurrency = max_concurrency
        self.rate_limit_delay = rate_limit_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _fetch_one(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[Any]]:
        """Fetch a single URL with retries."""
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            await asyncio.sleep(self.rate_limit_delay)
                            return (url, data)
                        elif resp.status == 429:
                            wait = 2 ** attempt
                            logger.warning(f"Rate limited on {url}, waiting {wait}s")
                            await asyncio.sleep(wait)
                        else:
                            logger.warning(f"HTTP {resp.status} for {url}")
                            return (url, None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} retries: {url} — {e}")
                    return (url, None)
                await asyncio.sleep(0.5 * (attempt + 1))
        return (url, None)

    async def fetch_all(
        self,
        urls: List[str],
        params_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[Tuple[str, Optional[Any]]]:
        """Fetch all URLs concurrently."""
        if params_list is None:
            params_list = cast(List[Optional[Dict[str, Any]]], [None] * len(urls))
        
        params_list = cast(List[Optional[Dict[str, Any]]], params_list)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(timeout=self.timeout, connector=connector) as session:
            tasks = [
                self._fetch_one(session, url, params)
                for url, params in zip(urls, params_list)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Task exception: {r}")
                final.append(("unknown", None))
            else:
                final.append(r)
        return final

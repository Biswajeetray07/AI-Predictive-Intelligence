"""
API Client — Reusable HTTP Request Wrapper
============================================
Built-in retries, pagination, timeout handling, and rate limit awareness.
"""

import time
import requests
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class APIClient:
    """
    Reusable HTTP client with retry logic, timeout, and pagination support.

    Usage:
        client = APIClient(base_url="https://api.example.com", headers={...})
        data = client.get("/endpoint", params={"key": "value"})
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get(
        self,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Send a GET request with automatic retries.

        Args:
            endpoint: API endpoint path (appended to base_url).
            params: Query parameters.
            url: Full URL (overrides base_url + endpoint).

        Returns:
            Parsed JSON response, or None on failure.
        """
        target = url if url else f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(target, params=params, timeout=self.timeout)

                if response.status_code == 200:
                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)
                    return response.json()

                elif response.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {wait:.1f}s... (attempt {attempt})")
                    time.sleep(wait)

                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}. Retrying... (attempt {attempt})")
                    time.sleep(self.retry_delay * attempt)

                else:
                    logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {target}. Retrying... (attempt {attempt})")
                time.sleep(self.retry_delay * attempt)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error: {e}. Retrying... (attempt {attempt})")
                time.sleep(self.retry_delay * attempt)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

        logger.error(f"Failed after {self.max_retries} attempts: {target}")
        return None

    def get_paginated(
        self,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        page_key: str = "page",
        max_pages: int = 10,
        results_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> List[Any]:
        """
        Paginate through an API endpoint collecting all results.

        Args:
            endpoint: API endpoint path.
            params: Base query parameters.
            page_key: Name of the page parameter.
            max_pages: Maximum pages to fetch.
            results_key: Key in response JSON containing results list.
            url: Full URL override.

        Returns:
            List of all collected items.
        """
        all_items = []
        params = dict(params or {})

        for page in range(1, max_pages + 1):
            params[page_key] = page
            data = self.get(endpoint=endpoint, params=params, url=url)

            if data is None:
                break

            if results_key:
                items = data.get(results_key, [])
            elif isinstance(data, list):
                items = data
            else:
                items = [data]

            if not items:
                break

            all_items.extend(items)
            logger.debug(f"Page {page}: {len(items)} items (total: {len(all_items)})")

        return all_items

    def close(self):
        """Close the underlying session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

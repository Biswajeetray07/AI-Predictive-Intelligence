"""
Generic API Client Module
=========================
Reusable HTTP client with session management, retry logic,
automatic pagination, timeout handling, and JSON parsing.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIClient:
    """Generic HTTP client with built-in retry, timeout, and pagination support."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        rate_limit_calls: int = 10,
        rate_limit_period: float = 60.0,
    ) -> None:
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API.
            headers: Default headers to include in every request.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            backoff_factor: Backoff multiplier between retries.
            rate_limit_calls: Maximum calls allowed per rate_limit_period.
            rate_limit_period: Time window (seconds) for rate limiting.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._call_timestamps: List[float] = []

        # Configure session with retry strategy
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by sleeping if necessary."""
        now = time.time()
        # Remove timestamps outside the current window
        self._call_timestamps = [
            ts for ts in self._call_timestamps
            if now - ts < self.rate_limit_period
        ]
        if len(self._call_timestamps) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self._call_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._call_timestamps.append(time.time())

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """
        Send a GET request with rate limiting.

        Args:
            endpoint: URL path (appended to base_url).
            params: Query parameters.
            headers: Additional headers for this request.

        Returns:
            requests.Response object.

        Raises:
            requests.exceptions.RequestException: On request failure.
        """
        self._enforce_rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        logger.debug(f"GET {url} params={params}")
        response = self.session.get(
            url, params=params, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        return response

    def get_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Send a GET request and return parsed JSON.

        Args:
            endpoint: URL path.
            params: Query parameters.
            headers: Additional headers.

        Returns:
            Parsed JSON response (dict or list).
        """
        response = self.get(endpoint, params=params, headers=headers)
        return response.json()

    def post_json(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Send a POST request with JSON body and return parsed JSON.

        Args:
            endpoint: URL path.
            json_data: JSON body payload.
            params: Query parameters.
            headers: Additional headers.

        Returns:
            Parsed JSON response.
        """
        self._enforce_rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        logger.debug(f"POST {url}")
        response = self.session.post(
            url, json=json_data, params=params, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        page_param: str = "page",
        per_page_param: str = "per_page",
        per_page: int = 100,
        max_pages: int = 50,
        results_key: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch all pages from a paginated API endpoint.

        Args:
            endpoint: URL path.
            params: Base query parameters.
            page_param: Name of the page number parameter.
            per_page_param: Name of the page size parameter.
            per_page: Number of results per page.
            max_pages: Maximum number of pages to fetch.
            results_key: JSON key containing the results list (None if root is list).

        Returns:
            Combined list of all results across pages.
        """
        all_results: List[Any] = []
        params = dict(params) if params else {}
        params[per_page_param] = per_page

        for page in range(1, max_pages + 1):
            params[page_param] = page
            try:
                data = self.get_json(endpoint, params=params)
                if results_key:
                    page_results = data.get(results_key, [])
                else:
                    page_results = data if isinstance(data, list) else []

                if not page_results:
                    break

                all_results.extend(page_results)
                logger.info(
                    f"Page {page}: fetched {len(page_results)} records "
                    f"(total: {len(all_results)})"
                )

                if len(page_results) < per_page:
                    break  # Last page
            except requests.exceptions.RequestException as e:
                logger.error(f"Pagination stopped at page {page}: {e}")
                break

        return all_results

    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

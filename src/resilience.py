"""
Resilience module for graceful degradation.

Provides fallback mechanisms when external APIs are unavailable.
"""

import logging
from typing import Dict, Any, Optional
from functools import wraps
import time

logger = logging.getLogger(__name__)

# Cache for fallback data
_fallback_cache: Dict[str, Any] = {}


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


class APIUnavailableError(Exception):
    """Raised when external API is unavailable."""
    pass


def retry(max_attempts: int = 2, delay: float = 1.0):
    """
    Decorator for retrying failed API calls.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, APIUnavailableError) as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def with_fallback(fallback_value: Any):
    """
    Decorator that returns a fallback value on failure.

    Args:
        fallback_value: Value to return if function fails
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed, using fallback: {e}"
                )
                return fallback_value
        return wrapper
    return decorator


@retry(max_attempts=2)
def get_ip_geo(ip: str) -> Dict[str, Any]:
    """
    Get geolocation data for an IP address.

    Uses ipapi.co free tier with graceful fallback.

    Args:
        ip: IP address to lookup

    Returns:
        Dict with geolocation data
    """
    # TODO: Implement actual API call
    # try:
    #     return ipapi_request(ip)
    # except RateLimitError:
    #     logger.warning("API rate limit - using cached mock")
    #     return {"mock": True, "country": "US", "city": "Unknown"}

    # Placeholder returning mock data
    return {
        "mock": True,
        "country": "US",
        "city": "Unknown",
        "ip": ip,
    }


def format_with_indicator(
    value: Any,
    is_live: bool,
    label: Optional[str] = None
) -> str:
    """
    Format a value with a live/cached indicator.

    Args:
        value: The value to display
        is_live: Whether the data is live or cached
        label: Optional label for the value

    Returns:
        Formatted string with indicator

    Example:
        >>> format_with_indicator(0.87, True, "Risk")
        'Risk: 0.87'
        >>> format_with_indicator(0.87, False, "Risk")
        'Risk: 0.87 [cached]'
    """
    indicator = "" if is_live else " [cached]"
    if label:
        return f"{label}: {value}{indicator}"
    return f"{value}{indicator}"


def get_cached_or_default(
    key: str,
    default: Any,
    cache: Optional[Dict] = None
) -> tuple[Any, bool]:
    """
    Get a cached value or return default.

    Args:
        key: Cache key
        default: Default value if not in cache
        cache: Optional cache dict (uses global if None)

    Returns:
        Tuple of (value, is_cached)
    """
    cache = cache or _fallback_cache
    if key in cache:
        return cache[key], True
    return default, False


def set_cache(key: str, value: Any, cache: Optional[Dict] = None) -> None:
    """
    Set a value in the cache.

    Args:
        key: Cache key
        value: Value to cache
        cache: Optional cache dict (uses global if None)
    """
    cache = cache or _fallback_cache
    cache[key] = value

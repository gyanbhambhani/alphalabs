"""
Rate Limiter

Thread-safe rate limiting for API calls with exponential backoff retry support.
"""
import logging
import time
import random
from functools import wraps
from threading import Lock
from typing import Callable, TypeVar, Optional, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and retries are exhausted."""
    pass


class RateLimiter:
    """
    Thread-safe rate limiter with token bucket algorithm.
    
    Supports:
    - Configurable calls per second
    - Burst allowance for short spikes
    - Thread-safe operation
    - Decorator usage
    """
    
    def __init__(
        self,
        calls_per_second: float = 10.0,
        burst_size: int = 5
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum sustained rate
            burst_size: Number of calls allowed in quick succession
        """
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size
        self.min_interval = 1.0 / calls_per_second
        
        self._lock = Lock()
        self._tokens = float(burst_size)
        self._last_update = time.time()
        
        logger.debug(
            f"RateLimiter initialized: {calls_per_second}/s, burst={burst_size}"
        )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time (called with lock held)."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.calls_per_second
        )
        self._last_update = now
    
    def acquire(self, timeout: float = None) -> bool:
        """
        Acquire permission to make a call.
        
        Blocks until a token is available or timeout is reached.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            True if acquired, False if timeout reached
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_tokens()
                
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self.calls_per_second
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            # Wait outside the lock
            time.sleep(wait_time)
    
    def try_acquire(self) -> bool:
        """
        Try to acquire without blocking.
        
        Returns:
            True if acquired, False if no tokens available
        """
        with self._lock:
            self._refill_tokens()
            
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for rate-limited functions.
        
        Usage:
            limiter = RateLimiter(calls_per_second=5)
            
            @limiter
            def my_api_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            self.acquire()
            return func(*args, **kwargs)
        return wrapper
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens (for debugging)."""
        with self._lock:
            self._refill_tokens()
            return self._tokens


class RetryWithBackoff:
    """
    Retry decorator with exponential backoff.
    
    Useful for handling transient API failures.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,)
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Multiplier for exponential backoff
            jitter: Add random jitter to prevent thundering herd
            exceptions: Tuple of exceptions to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for functions with retry logic.
        
        Usage:
            @RetryWithBackoff(max_retries=3)
            def flaky_api_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/"
                            f"{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after "
                            f"{self.max_retries + 1} attempts: {e}"
                        )
            
            raise last_exception
        return wrapper


def rate_limited_call(
    func: Callable[..., T],
    rate_limiter: RateLimiter,
    *args,
    max_retries: int = 3,
    retry_exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """
    Execute a function with rate limiting and retry logic.
    
    Args:
        func: Function to call
        rate_limiter: RateLimiter instance
        *args: Positional arguments for func
        max_retries: Number of retries on failure
        retry_exceptions: Exceptions to retry on
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func call
    """
    retry = RetryWithBackoff(
        max_retries=max_retries,
        exceptions=retry_exceptions
    )
    
    @rate_limiter
    @retry
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped()

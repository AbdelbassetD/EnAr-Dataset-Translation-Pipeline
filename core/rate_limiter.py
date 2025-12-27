"""
Simple rate limiter for API calls.
"""

import time
from threading import Lock
from collections import deque


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Seconds between requests
        self.timestamps = deque()
        self.lock = Lock()
    
    def acquire(self):
        """
        Wait if necessary to respect rate limit, then proceed.
        This method blocks until it's safe to make a request.
        """
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()
            
            # If we've hit the limit, wait
            if len(self.timestamps) >= self.rpm:
                sleep_time = 60 - (now - self.timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    # Clean up old timestamps again
                    while self.timestamps and now - self.timestamps[0] > 60:
                        self.timestamps.popleft()
            
            # Record this request
            self.timestamps.append(now)
    
    def get_wait_time(self) -> float:
        """
        Get estimated wait time before next request.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            
            # Remove old timestamps
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()
            
            if len(self.timestamps) >= self.rpm:
                return max(0, 60 - (now - self.timestamps[0]))
            
            return 0.0


if __name__ == "__main__":
    # Test rate limiter
    print("Testing rate limiter (5 RPM)...")
    limiter = RateLimiter(5)
    
    for i in range(7):
        print(f"Request {i+1} - waiting {limiter.get_wait_time():.2f}s")
        limiter.acquire()
        print(f"Request {i+1} - executed at {time.time():.2f}")

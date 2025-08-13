"""
Retry handler for failed trading operations with exponential backoff.
"""
import time
import random
from typing import Any, Callable, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


class FailureType(Enum):
    """Classification of failure types."""
    TEMPORARY_NETWORK = "temporary_network"
    TEMPORARY_API = "temporary_api"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    MARKET_CLOSED = "market_closed"
    INVALID_ORDER = "invalid_order"
    RATE_LIMITED = "rate_limited"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    error_message: str
    failure_type: FailureType
    delay_used: float


class RetryHandler:
    """Handles retry logic for trading operations with smart backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5, 
                 max_delay: float = 30.0, backoff_factor: float = 2.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        # Track retry attempts for analysis
        self.retry_history: List[RetryAttempt] = []
        
        # Failure type mapping for Alpaca errors
        self.error_patterns = {
            'network': FailureType.TEMPORARY_NETWORK,
            'timeout': FailureType.TEMPORARY_NETWORK,
            'connection': FailureType.TEMPORARY_NETWORK,
            'rate limit': FailureType.RATE_LIMITED,
            'too many requests': FailureType.RATE_LIMITED,
            'insufficient': FailureType.INSUFFICIENT_FUNDS,
            'buying power': FailureType.INSUFFICIENT_FUNDS,
            'market is closed': FailureType.MARKET_CLOSED,
            'invalid': FailureType.INVALID_ORDER,
            'rejected': FailureType.INVALID_ORDER,
            'forbidden': FailureType.PERMANENT,
            'unauthorized': FailureType.PERMANENT,
        }
    
    def classify_error(self, error_message: str) -> FailureType:
        """
        Classify error type based on error message.
        
        Args:
            error_message: Error message from API
            
        Returns:
            FailureType classification
        """
        error_lower = error_message.lower()
        
        for pattern, failure_type in self.error_patterns.items():
            if pattern in error_lower:
                return failure_type
        
        return FailureType.UNKNOWN
    
    def should_retry(self, failure_type: FailureType, attempt_number: int) -> bool:
        """
        Determine if operation should be retried based on failure type.
        
        Args:
            failure_type: Type of failure that occurred
            attempt_number: Current attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        # Never retry certain failure types
        if failure_type in [FailureType.PERMANENT, FailureType.INVALID_ORDER]:
            return False
        
        # Don't exceed max retries
        if attempt_number >= self.max_retries:
            return False
        
        # Special handling for insufficient funds (retry once after small delay)
        if failure_type == FailureType.INSUFFICIENT_FUNDS and attempt_number >= 1:
            return False
        
        # Special handling for market closed (don't retry immediately)
        if failure_type == FailureType.MARKET_CLOSED:
            return False
        
        return True
    
    def calculate_delay(self, attempt_number: int, failure_type: FailureType) -> float:
        """
        Calculate delay before next retry attempt.
        
        Args:
            attempt_number: Current attempt number
            failure_type: Type of failure
            
        Returns:
            Delay in seconds
        """
        # Base exponential backoff
        delay = self.base_delay * (self.backoff_factor ** attempt_number)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        delay += jitter
        
        # Special handling for rate limiting (longer delays)
        if failure_type == FailureType.RATE_LIMITED:
            delay *= 3  # 3x longer for rate limits
        
        # Cap at maximum delay
        delay = min(delay, self.max_delay)
        
        return round(delay, 2)
    
    def execute_with_retry(self, operation: Callable, operation_name: str, 
                          *args, **kwargs) -> Any:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            operation_name: Name of operation for logging
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of successful operation or None if all retries failed
        """
        attempt_number = 0
        
        while attempt_number <= self.max_retries:
            try:
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # If we get here, operation succeeded
                if attempt_number > 0:
                    print(f"✅ {operation_name} succeeded after {attempt_number} retries")
                
                return result
                
            except Exception as e:
                error_message = str(e)
                failure_type = self.classify_error(error_message)
                
                # Log the attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    timestamp=datetime.now(),
                    error_message=error_message,
                    failure_type=failure_type,
                    delay_used=0.0
                )
                self.retry_history.append(retry_attempt)
                
                print(f"❌ {operation_name} failed (attempt {attempt_number + 1}): {error_message}")
                print(f"   Failure type: {failure_type.value}")
                
                # Check if we should retry
                if not self.should_retry(failure_type, attempt_number):
                    print(f"🛑 Not retrying {operation_name} - {failure_type.value}")
                    return None
                
                attempt_number += 1
                
                # Calculate and apply delay
                if attempt_number <= self.max_retries:
                    delay = self.calculate_delay(attempt_number - 1, failure_type)
                    retry_attempt.delay_used = delay
                    
                    print(f"⏳ Retrying {operation_name} in {delay}s...")
                    time.sleep(delay)
        
        print(f"💥 {operation_name} failed after {self.max_retries} retries")
        return None
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get statistics about retry attempts."""
        if not self.retry_history:
            return {"total_attempts": 0}
        
        # Count failures by type
        failure_counts = {}
        for attempt in self.retry_history:
            failure_type = attempt.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        # Calculate success rate
        total_operations = len(set(attempt.timestamp.strftime("%Y%m%d%H%M%S") 
                                 for attempt in self.retry_history))
        
        return {
            "total_attempts": len(self.retry_history),
            "total_operations": total_operations,
            "failure_breakdown": failure_counts,
            "avg_delay": sum(a.delay_used for a in self.retry_history) / len(self.retry_history),
            "recent_failures": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.failure_type.value,
                    "message": a.error_message[:100]
                }
                for a in self.retry_history[-5:]  # Last 5 failures
            ]
        }
    
    def reset_history(self):
        """Clear retry history (useful for testing or long-running bots)."""
        self.retry_history.clear()
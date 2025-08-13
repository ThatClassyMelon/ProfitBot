"""
Mathematical utility functions for trading calculations.
"""
from typing import Union


def percent_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change as decimal (e.g., 0.05 for 5% increase)
    """
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def format_currency(amount: float, symbol: str = "$") -> str:
    """
    Format a number as currency with proper decimals.
    
    Args:
        amount: Amount to format
        symbol: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{symbol}{amount:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a decimal percentage as a readable string.
    
    Args:
        value: Decimal percentage (e.g., 0.05 for 5%)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def round_to_precision(value: float, precision: int = 8) -> float:
    """
    Round a value to specified decimal places.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    return round(value, precision)


def calculate_trade_amount(balance: float, percentage: float, min_amount: float = 0.0) -> float:
    """
    Calculate trade amount based on balance percentage with minimum threshold.
    
    Args:
        balance: Available balance
        percentage: Percentage of balance to use (as decimal)
        min_amount: Minimum trade amount
        
    Returns:
        Trade amount
    """
    trade_amount = balance * percentage
    return max(trade_amount, min_amount) if trade_amount >= min_amount else 0.0
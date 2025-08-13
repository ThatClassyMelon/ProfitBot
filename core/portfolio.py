"""
Portfolio management for tracking USDT balance and cryptocurrency holdings.
"""
from typing import Dict, Any, Optional
from utils.math_tools import round_to_precision, format_currency


class Portfolio:
    """Manages portfolio balance and holdings for the trading bot."""
    
    def __init__(self, initial_balance: float):
        """
        Initialize the portfolio.
        
        Args:
            initial_balance: Starting USDT balance
        """
        self.usdt_balance = initial_balance
        self.holdings: Dict[str, float] = {}  # coin -> quantity
        self.last_trade_prices: Dict[str, float] = {}  # coin -> last trade price
        self.initial_balance = initial_balance
    
    def get_usdt_balance(self) -> float:
        """
        Get current USDT balance.
        
        Returns:
            Current USDT balance
        """
        return self.usdt_balance
    
    def get_holdings(self) -> Dict[str, float]:
        """
        Get current cryptocurrency holdings.
        
        Returns:
            Dictionary of coin holdings
        """
        return self.holdings.copy()
    
    def get_holding(self, coin: str) -> float:
        """
        Get holding amount for a specific coin.
        
        Args:
            coin: Coin symbol
            
        Returns:
            Amount of coin held
        """
        return self.holdings.get(coin, 0.0)
    
    def add_usdt(self, amount: float) -> None:
        """
        Add USDT to balance.
        
        Args:
            amount: Amount to add
        """
        self.usdt_balance = round_to_precision(self.usdt_balance + amount, 2)
    
    def deduct_usdt(self, amount: float) -> bool:
        """
        Deduct USDT from balance.
        
        Args:
            amount: Amount to deduct
            
        Returns:
            True if successful, False if insufficient balance
        """
        if self.usdt_balance >= amount:
            self.usdt_balance = round_to_precision(self.usdt_balance - amount, 2)
            return True
        return False
    
    def add_holding(self, coin: str, quantity: float) -> None:
        """
        Add cryptocurrency to holdings.
        
        Args:
            coin: Coin symbol
            quantity: Quantity to add
        """
        current_holding = self.holdings.get(coin, 0.0)
        self.holdings[coin] = round_to_precision(current_holding + quantity, 8)
        
        # Remove from holdings if quantity becomes effectively zero
        if self.holdings[coin] < 1e-8:
            self.holdings.pop(coin, None)
    
    def deduct_holding(self, coin: str, quantity: float) -> bool:
        """
        Deduct cryptocurrency from holdings.
        
        Args:
            coin: Coin symbol
            quantity: Quantity to deduct
            
        Returns:
            True if successful, False if insufficient holding
        """
        current_holding = self.holdings.get(coin, 0.0)
        if current_holding >= quantity:
            new_holding = round_to_precision(current_holding - quantity, 8)
            if new_holding < 1e-8:  # Effectively zero
                self.holdings.pop(coin, None)
            else:
                self.holdings[coin] = new_holding
            return True
        return False
    
    def execute_buy(self, coin: str, quantity: float, price: float) -> bool:
        """
        Execute a buy order.
        
        Args:
            coin: Coin to buy
            quantity: Quantity to buy
            price: Price per coin
            
        Returns:
            True if successful, False if insufficient balance
        """
        total_cost = quantity * price
        if self.deduct_usdt(total_cost):
            self.add_holding(coin, quantity)
            self.last_trade_prices[coin] = price
            return True
        return False
    
    def execute_sell(self, coin: str, quantity: float, price: float) -> bool:
        """
        Execute a sell order.
        
        Args:
            coin: Coin to sell
            quantity: Quantity to sell
            price: Price per coin
            
        Returns:
            True if successful, False if insufficient holding
        """
        if self.deduct_holding(coin, quantity):
            total_value = quantity * price
            self.add_usdt(total_value)
            self.last_trade_prices[coin] = price
            return True
        return False
    
    def get_last_trade_price(self, coin: str) -> Optional[float]:
        """
        Get the last trade price for a coin.
        
        Args:
            coin: Coin symbol
            
        Returns:
            Last trade price or None if never traded
        """
        return self.last_trade_prices.get(coin)
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including USDT and holdings.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Total portfolio value in USDT
        """
        total_value = self.usdt_balance
        
        for coin, quantity in self.holdings.items():
            price = current_prices.get(coin, 0.0)
            total_value += quantity * price
        
        return round_to_precision(total_value, 2)
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Portfolio summary dictionary
        """
        total_value = self.calculate_portfolio_value(current_prices)
        profit_loss = total_value - self.initial_balance
        profit_loss_percent = (profit_loss / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        holdings_value = {}
        for coin, quantity in self.holdings.items():
            price = current_prices.get(coin, 0.0)
            value = quantity * price
            holdings_value[coin] = {
                'quantity': quantity,
                'price': price,
                'value': value,
                'last_trade_price': self.last_trade_prices.get(coin)
            }
        
        return {
            'usdt_balance': self.usdt_balance,
            'total_value': total_value,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'holdings': holdings_value
        }
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.usdt_balance = self.initial_balance
        self.holdings.clear()
        self.last_trade_prices.clear()
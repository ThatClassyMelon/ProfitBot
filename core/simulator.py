"""
Realistic cryptocurrency price simulation with spikes, drops, and market dynamics.
"""
import random
import time
import math
from typing import Dict, Any, List
from utils.math_tools import round_to_precision


class PriceSimulator:
    """Simulates realistic cryptocurrency price movements with spikes, drops, momentum, and market correlation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the price simulator.
        
        Args:
            config: Configuration dictionary containing coin settings
        """
        self.coins = config['coins']
        self.price_update_factor = config['simulation']['price_update_factor']
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, list] = {}
        
        # Market state tracking for realistic behavior
        self.momentum: Dict[str, float] = {}  # Current momentum for each coin
        self.trend_strength: Dict[str, float] = {}  # Trend strength (-1 to 1)
        self.volatility_multiplier: Dict[str, float] = {}  # Dynamic volatility
        self.last_spike_time: Dict[str, int] = {}  # Time since last major move
        self.market_sentiment = 0.0  # Overall market sentiment (-1 to 1)
        self.tick_counter = 0
        
        # Initialize prices and state
        for coin, settings in self.coins.items():
            initial_price = settings['initial_price']
            self.current_prices[coin] = initial_price
            self.price_history[coin] = [initial_price]
            self.momentum[coin] = 0.0
            self.trend_strength[coin] = 0.0
            self.volatility_multiplier[coin] = 1.0
            self.last_spike_time[coin] = 0
    
    def update_prices(self) -> Dict[str, float]:
        """
        Update all coin prices using realistic market dynamics with spikes, drops, and momentum.
        Each tick represents 1 minute of market activity.
        
        Returns:
            Dictionary of updated prices
        """
        self.tick_counter += 1
        
        # Update overall market sentiment (changes slowly)
        if random.random() < 0.05:  # 5% chance to shift market sentiment
            self.market_sentiment += random.normalvariate(0, 0.2)
            self.market_sentiment = max(-1.0, min(1.0, self.market_sentiment))
        
        for coin, settings in self.coins.items():
            current_price = self.current_prices[coin]
            base_volatility = settings['volatility']
            
            # Update coin-specific state
            self._update_coin_momentum(coin)
            self._update_volatility_multiplier(coin)
            
            # Calculate price change components
            normal_change = self._calculate_normal_price_change(coin, base_volatility)
            momentum_change = self._calculate_momentum_change(coin)
            spike_change = self._calculate_spike_change(coin, base_volatility)
            
            # Combine all price change factors
            total_change = normal_change + momentum_change + spike_change
            
            # Apply market sentiment influence
            sentiment_influence = self.market_sentiment * 0.001  # 0.1% max influence per tick
            total_change += sentiment_influence
            
            # Apply the price change
            new_price = current_price * (1 + total_change)
            
            # Ensure price doesn't go negative or unrealistically low
            new_price = max(new_price, current_price * 0.01)  # Minimum 1% of current price
            
            # Update price and history
            self.current_prices[coin] = round_to_precision(new_price, 8)
            self.price_history[coin].append(self.current_prices[coin])
            
            # Update momentum based on actual price change
            actual_change = (new_price - current_price) / current_price
            self.momentum[coin] = self.momentum[coin] * 0.95 + actual_change * 0.05
            
            # Keep history manageable (last 1000 prices)
            if len(self.price_history[coin]) > 1000:
                self.price_history[coin] = self.price_history[coin][-1000:]
        
        return self.current_prices.copy()
    
    def _update_coin_momentum(self, coin: str) -> None:
        """Update momentum for a specific coin."""
        # Momentum decay
        self.momentum[coin] *= 0.98
        
        # Random momentum shifts (like news events)
        if random.random() < 0.02:  # 2% chance per minute
            momentum_shift = random.normalvariate(0, 0.01)
            self.momentum[coin] += momentum_shift
        
        # Clamp momentum
        self.momentum[coin] = max(-0.05, min(0.05, self.momentum[coin]))
    
    def _update_volatility_multiplier(self, coin: str) -> None:
        """Update dynamic volatility for a coin."""
        # Volatility tends to cluster (high vol periods followed by high vol)
        if abs(self.momentum[coin]) > 0.02:  # High momentum = higher volatility
            self.volatility_multiplier[coin] = min(3.0, self.volatility_multiplier[coin] * 1.1)
        else:
            self.volatility_multiplier[coin] = max(0.5, self.volatility_multiplier[coin] * 0.99)
    
    def _calculate_normal_price_change(self, coin: str, base_volatility: float) -> float:
        """Calculate normal random price movement."""
        effective_volatility = base_volatility * self.volatility_multiplier[coin]
        return random.normalvariate(0, effective_volatility * 0.01)  # Scale for 1-minute intervals
    
    def _calculate_momentum_change(self, coin: str) -> float:
        """Calculate momentum-based price change."""
        return self.momentum[coin] * 0.5  # Apply 50% of momentum
    
    def _calculate_spike_change(self, coin: str, base_volatility: float) -> float:
        """Calculate potential spike or drop."""
        time_since_spike = self.tick_counter - self.last_spike_time[coin]
        
        # Probability of spike increases over time, but is generally low
        spike_probability = min(0.008, time_since_spike * 0.0001)  # Max 0.8% chance
        
        if random.random() < spike_probability:
            # Determine spike direction and magnitude
            is_pump = random.random() > 0.4  # 60% chance of pump vs 40% dump
            
            # Spike magnitude (larger for smaller coins)
            base_magnitude = base_volatility * random.uniform(3, 8)  # 3x to 8x normal volatility
            
            if is_pump:
                spike_change = base_magnitude
                print(f"🚀 {coin} PUMP: +{spike_change*100:.1f}%")
            else:
                spike_change = -base_magnitude * 0.8  # Dumps slightly smaller than pumps
                print(f"📉 {coin} DUMP: {spike_change*100:.1f}%")
            
            self.last_spike_time[coin] = self.tick_counter
            
            # Create follow-through momentum
            self.momentum[coin] += spike_change * 0.3
            
            return spike_change
        
        return 0.0
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all coins.
        
        Returns:
            Dictionary of current prices
        """
        return self.current_prices.copy()
    
    def get_price(self, coin: str) -> float:
        """
        Get current price for a specific coin.
        
        Args:
            coin: Coin symbol
            
        Returns:
            Current price
        """
        return self.current_prices.get(coin, 0.0)
    
    def get_price_history(self, coin: str, limit: int = 100) -> list:
        """
        Get price history for a specific coin.
        
        Args:
            coin: Coin symbol
            limit: Number of historical prices to return
            
        Returns:
            List of historical prices
        """
        history = self.price_history.get(coin, [])
        return history[-limit:] if len(history) > limit else history
    
    def reset_prices(self) -> None:
        """Reset all prices and market state to initial values."""
        self.tick_counter = 0
        self.market_sentiment = 0.0
        
        for coin, settings in self.coins.items():
            initial_price = settings['initial_price']
            self.current_prices[coin] = initial_price
            self.price_history[coin] = [initial_price]
            self.momentum[coin] = 0.0
            self.trend_strength[coin] = 0.0
            self.volatility_multiplier[coin] = 1.0
            self.last_spike_time[coin] = 0
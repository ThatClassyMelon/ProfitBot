"""
Enhanced Threshold-Based Mean Reversion Strategy with Adaptive Logic.
Features: Adaptive thresholds, smart baseline updates, momentum filter, 
tiered DCA, timed sells, and profit rebalancing.
"""
import time
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import deque
from utils.math_tools import percent_change, calculate_trade_amount, round_to_precision


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REBALANCE = "REBALANCE"


class TradeSignal:
    """Represents a trading signal with enhanced metadata."""
    
    def __init__(self, coin: str, action: TradeAction, quantity: float = 0.0, 
                 price: float = 0.0, reason: str = "", tier: int = 0):
        self.coin = coin
        self.action = action
        self.quantity = quantity
        self.price = price
        self.reason = reason
        self.tier = tier  # DCA tier level
        self.timestamp = time.time()
        self.explanation = ""  # Simple explanation for users
    
    def __repr__(self) -> str:
        if self.action == TradeAction.HOLD:
            return f"TradeSignal({self.coin}: {self.action.value} - {self.reason})"
        tier_str = f" [T{self.tier}]" if self.tier > 0 else ""
        return f"TradeSignal({self.coin}: {self.action.value} {self.quantity:.6f} @ ${self.price:.2f}{tier_str} - {self.reason})"


class CoinState:
    """Tracks state for individual coins."""
    
    def __init__(self, coin: str, initial_price: float, volatility: float):
        self.coin = coin
        self.baseline_price = initial_price
        self.volatility = volatility
        self.price_history = deque(maxlen=10)  # Last 10 prices for momentum
        self.dca_level = 0  # Current DCA tier (0-3)
        self.last_buy_price = 0.0
        self.last_buy_time = 0.0
        self.total_buy_count = 0
        self.last_sell_time = 0.0
        
        # Add initial price to history
        self.price_history.append(initial_price)
    
    def update_price_history(self, price: float) -> None:
        """Update price history for momentum calculation."""
        self.price_history.append(price)
    
    def get_momentum(self) -> float:
        """
        Calculate momentum over last 3 ticks.
        Positive = upward momentum, Negative = downward momentum.
        """
        if len(self.price_history) < 4:
            return 0.0
        
        current_price = self.price_history[-1]
        price_3_ticks_ago = self.price_history[-4]
        
        return (current_price - price_3_ticks_ago) / price_3_ticks_ago
    
    def update_baseline_weighted(self, current_price: float, weight: float = 0.3) -> None:
        """
        Update baseline using weighted average.
        new_baseline = (1-weight) * old_baseline + weight * current_price
        """
        self.baseline_price = (1 - weight) * self.baseline_price + weight * current_price
    
    def get_adaptive_threshold(self, k_factor: float) -> float:
        """Calculate adaptive threshold based on volatility."""
        return self.volatility * k_factor
    
    def reset_dca_level(self) -> None:
        """Reset DCA level after profitable exit."""
        self.dca_level = 0


class EnhancedThresholdStrategy:
    """
    Advanced threshold-based trading strategy with adaptive features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced trading strategy."""
        self.config = config
        strategy_config = config['strategy']
        
        # Core parameters
        self.base_trade_percentage = strategy_config['trade_amount_percentage']
        self.min_trade_amount = strategy_config['min_trade_amount']
        
        # Enhanced parameters
        self.k_factor = strategy_config.get('adaptive_k_factor', 1.5)
        self.baseline_weight = strategy_config.get('baseline_update_weight', 0.3)
        self.momentum_threshold = strategy_config.get('momentum_threshold', 0.0)
        self.timed_sell_threshold = strategy_config.get('timed_sell_threshold', 0.015)  # 1.5%
        self.timed_sell_percentage = strategy_config.get('timed_sell_percentage', 0.05)  # 5%
        self.max_hold_ticks = strategy_config.get('max_hold_ticks', 300)  # 5 minutes
        
        # DCA tier multipliers
        self.dca_multipliers = strategy_config.get('dca_multipliers', [0.10, 0.15, 0.20])
        self.max_dca_levels = len(self.dca_multipliers)
        
        # Profit rebalancing
        self.rebalance_profit_threshold = strategy_config.get('rebalance_profit_threshold', 0.01)  # 1%
        self.rebalance_percentage = strategy_config.get('rebalance_percentage', 0.10)  # 10%
        self.trades_since_rebalance = 0
        self.last_portfolio_value = 0.0
        
        # Initialize coin states
        self.coin_states: Dict[str, CoinState] = {}
        for coin, coin_config in config['coins'].items():
            self.coin_states[coin] = CoinState(
                coin, 
                coin_config['initial_price'], 
                coin_config['volatility']
            )
    
    def analyze_market(self, current_prices: Dict[str, float], 
                      portfolio, last_trade_prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Analyze market with enhanced logic and generate trading signals.
        """
        signals = []
        current_time = time.time()
        
        # Update price histories
        for coin, price in current_prices.items():
            if coin in self.coin_states:
                self.coin_states[coin].update_price_history(price)
        
        # Check for profit rebalancing first
        rebalance_signal = self._check_profit_rebalancing(current_prices, portfolio)
        if rebalance_signal:
            signals.append(rebalance_signal)
        
        # Analyze each coin
        for coin in self.coin_states.keys():
            current_price = current_prices.get(coin, 0.0)
            if current_price <= 0:
                continue
            
            signal = self._analyze_coin_enhanced(coin, current_price, portfolio, current_time)
            signals.append(signal)
        
        return signals
    
    def _analyze_coin_enhanced(self, coin: str, current_price: float, 
                              portfolio, current_time: float) -> TradeSignal:
        """Enhanced coin analysis with all new features."""
        coin_state = self.coin_states[coin]
        adaptive_threshold = coin_state.get_adaptive_threshold(self.k_factor)
        
        # Calculate price change from baseline
        price_change = percent_change(coin_state.baseline_price, current_price)
        
        # Check for BUY signal (price dropped by adaptive threshold)
        if price_change <= -adaptive_threshold:
            return self._generate_enhanced_buy_signal(coin, current_price, portfolio, coin_state, abs(price_change))
        
        # Check for SELL signal (price increased by adaptive threshold)
        elif price_change >= adaptive_threshold:
            return self._generate_enhanced_sell_signal(coin, current_price, portfolio, coin_state, price_change)
        
        # Check for timed sell fallback
        elif self._should_timed_sell(coin_state, current_price, current_time):
            return self._generate_timed_sell_signal(coin, current_price, portfolio, coin_state)
        
        # Hold signal
        else:
            movement_needed = adaptive_threshold - abs(price_change)
            movement_needed_percent = movement_needed * 100
            
            # Update baseline even during hold (gradual drift compensation)
            if abs(price_change) > 0.005:  # If price moved > 0.5%
                coin_state.update_baseline_weighted(current_price, self.baseline_weight * 0.1)
            
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Need {movement_needed_percent:.2f}% more (adaptive: {adaptive_threshold*100:.1f}%)"
            )
    
    def _generate_enhanced_buy_signal(self, coin: str, current_price: float, 
                                    portfolio, coin_state: CoinState, price_drop: float) -> TradeSignal:
        """Generate enhanced buy signal with momentum filter and DCA logic."""
        
        # 1. Momentum filter check
        momentum = coin_state.get_momentum()
        if momentum < self.momentum_threshold:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Momentum too negative ({momentum*100:.1f}%) - price still falling"
            )
        
        # 2. Check DCA level limits
        if coin_state.dca_level >= self.max_dca_levels:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Max DCA level reached ({coin_state.dca_level}/{self.max_dca_levels})"
            )
        
        # 3. Calculate tiered trade amount
        usdt_balance = portfolio.get_usdt_balance()
        trade_multiplier = self.dca_multipliers[coin_state.dca_level]
        trade_amount_usdt = usdt_balance * trade_multiplier
        
        if trade_amount_usdt < self.min_trade_amount:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Insufficient balance for DCA T{coin_state.dca_level + 1} (need ${self.min_trade_amount})"
            )
        
        # 4. Calculate quantity and execute
        quantity = trade_amount_usdt / current_price
        
        # Update coin state
        coin_state.dca_level += 1
        coin_state.last_buy_price = current_price
        coin_state.last_buy_time = time.time()
        coin_state.total_buy_count += 1
        
        # Smart baseline update after buy
        coin_state.update_baseline_weighted(current_price, self.baseline_weight)
        
        return TradeSignal(
            coin, TradeAction.BUY, quantity, current_price,
            f"DCA T{coin_state.dca_level} - Price dropped {price_drop*100:.1f}%",
            tier=coin_state.dca_level
        )
    
    def _generate_enhanced_sell_signal(self, coin: str, current_price: float, 
                                     portfolio, coin_state: CoinState, price_increase: float) -> TradeSignal:
        """Generate enhanced sell signal with intelligent position sizing."""
        holding = portfolio.get_holding(coin)
        
        if holding <= 0:
            # Update baseline when we can't sell (pump with no holdings)
            coin_state.update_baseline_weighted(current_price, self.baseline_weight)
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Updated baseline to ${current_price:.2f} - pump with no holdings"
            )
        
        # Calculate sell percentage based on DCA level
        # Higher DCA level = sell more to lock in gains
        base_sell_percentage = self.base_trade_percentage
        dca_bonus = coin_state.dca_level * 0.05  # Extra 5% per DCA level
        sell_percentage = min(base_sell_percentage + dca_bonus, 0.5)  # Cap at 50%
        
        sell_quantity = holding * sell_percentage
        sell_value = sell_quantity * current_price
        
        # Ensure minimum trade size
        if sell_value < self.min_trade_amount:
            sell_quantity = holding  # Sell all if below minimum
        
        # Update coin state
        coin_state.last_sell_time = time.time()
        
        # Smart baseline update after sell
        coin_state.update_baseline_weighted(current_price, self.baseline_weight)
        
        # Reset DCA level if profitable exit
        if current_price > coin_state.last_buy_price:
            coin_state.reset_dca_level()
        
        return TradeSignal(
            coin, TradeAction.SELL, sell_quantity, current_price,
            f"Price up {price_increase*100:.1f}% - selling {sell_percentage*100:.0f}%"
        )
    
    def _should_timed_sell(self, coin_state: CoinState, current_price: float, current_time: float) -> bool:
        """Check if timed sell conditions are met."""
        if coin_state.last_buy_time == 0:
            return False
        
        # Check if held long enough
        hold_duration = current_time - coin_state.last_buy_time
        if hold_duration < self.max_hold_ticks:
            return False
        
        # Check if price is up modestly from baseline
        price_change = percent_change(coin_state.baseline_price, current_price)
        return price_change >= self.timed_sell_threshold
    
    def _generate_timed_sell_signal(self, coin: str, current_price: float, 
                                  portfolio, coin_state: CoinState) -> TradeSignal:
        """Generate timed sell signal to avoid bag-holding."""
        holding = portfolio.get_holding(coin)
        
        if holding <= 0:
            return TradeSignal(coin, TradeAction.HOLD, 0.0, current_price, "No holdings for timed sell")
        
        # Sell small percentage to cut losses
        sell_quantity = holding * self.timed_sell_percentage
        
        # Update state
        coin_state.last_sell_time = time.time()
        coin_state.update_baseline_weighted(current_price, self.baseline_weight)
        
        price_change = percent_change(coin_state.baseline_price, current_price)
        
        return TradeSignal(
            coin, TradeAction.SELL, sell_quantity, current_price,
            f"Timed sell - held too long, up {price_change*100:.1f}%"
        )
    
    def _check_profit_rebalancing(self, current_prices: Dict[str, float], portfolio) -> Optional[TradeSignal]:
        """Check if profit rebalancing should occur."""
        current_portfolio_value = portfolio.calculate_portfolio_value(current_prices)
        
        # Initialize last portfolio value if not set
        if self.last_portfolio_value == 0:
            self.last_portfolio_value = current_portfolio_value
            return None
        
        # Check profit threshold
        profit_ratio = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        
        # Trigger rebalancing if profit threshold met or every 10 trades
        should_rebalance = (
            profit_ratio >= self.rebalance_profit_threshold or 
            self.trades_since_rebalance >= 10
        )
        
        if should_rebalance:
            self.last_portfolio_value = current_portfolio_value
            self.trades_since_rebalance = 0
            
            return TradeSignal(
                "PORTFOLIO", TradeAction.REBALANCE, self.rebalance_percentage, 0.0,
                f"Rebalancing - profit: {profit_ratio*100:.1f}%"
            )
        
        return None
    
    def on_trade_executed(self, trade_signal: TradeSignal) -> None:
        """Callback when a trade is executed."""
        if trade_signal.action in [TradeAction.BUY, TradeAction.SELL]:
            self.trades_since_rebalance += 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get enhanced strategy information."""
        return {
            'strategy_type': 'enhanced_threshold_based',
            'adaptive_thresholds': True,
            'k_factor': self.k_factor,
            'dca_levels': self.max_dca_levels,
            'dca_multipliers': self.dca_multipliers,
            'momentum_filter': True,
            'timed_sells': True,
            'profit_rebalancing': True,
            'baseline_weighting': self.baseline_weight
        }
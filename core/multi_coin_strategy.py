"""
Multi-Coin Strategy System - Different optimized strategies for each cryptocurrency.
Each coin gets its own strategy parameters based on its unique characteristics.
"""
import time
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd

from core.enhanced_strategy import TradeAction, TradeSignal, CoinState
from utils.math_tools import percent_change, calculate_trade_amount, round_to_precision


class CoinStrategy:
    """Base class for individual coin strategies."""
    
    def __init__(self, coin: str, config: Dict[str, Any]):
        self.coin = coin
        self.config = config
        
        # Default parameters (overridden by specific strategies)
        self.ema_fast = 5
        self.ema_slow = 13
        self.momentum_threshold = 0.001
        self.take_profit_pct = 0.01
        self.stop_loss_pct = 0.005
        self.max_hold_periods = 15
        self.volume_filter = 1.2
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Position tracking
        self.position = {
            'entry_price': 0.0,
            'entry_time': 0,
            'quantity': 0.0,
            'periods_held': 0,
            'position_type': None
        }
        
        # History tracking
        self.price_history = deque(maxlen=50)
        self.volume_history = deque(maxlen=50)
        self.ema_fast_history = deque(maxlen=20)
        self.ema_slow_history = deque(maxlen=20)
    
    def calculate_ema(self, values: deque, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return sum(values) / len(values) if values else 0.0
        
        values_list = list(values)[-period:]
        multiplier = 2 / (period + 1)
        ema = values_list[0]
        
        for value in values_list[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, prices: deque, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        prices_list = list(prices)[-period-1:]
        gains = []
        losses = []
        
        for i in range(1, len(prices_list)):
            change = prices_list[i] - prices_list[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def update_indicators(self, price: float, volume: float):
        """Update technical indicators."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) >= 2:
            ema_fast = self.calculate_ema(self.price_history, self.ema_fast)
            ema_slow = self.calculate_ema(self.price_history, self.ema_slow)
            
            self.ema_fast_history.append(ema_fast)
            self.ema_slow_history.append(ema_slow)
    
    def check_volume_filter(self) -> bool:
        """Check if current volume meets filter requirement."""
        if len(self.volume_history) < 10:
            return True
        
        current_volume = self.volume_history[-1]
        avg_volume = sum(list(self.volume_history)[-10:]) / 10
        
        return current_volume >= avg_volume * self.volume_filter
    
    def get_signal_strength(self, current_price: float) -> float:
        """Get signal strength for this coin's strategy."""
        raise NotImplementedError("Subclasses must implement get_signal_strength")
    
    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        if self.position['quantity'] == 0:
            return None
        
        entry_price = self.position['entry_price']
        periods_held = self.position['periods_held']
        
        # Time-based exit
        if periods_held >= self.max_hold_periods:
            return f"Max hold time reached ({self.max_hold_periods} periods)"
        
        # Profit/loss exits
        price_change = (current_price - entry_price) / entry_price
        
        if price_change >= self.take_profit_pct:
            return f"Take profit: {price_change*100:.2f}% gain"
        elif price_change <= -self.stop_loss_pct:
            return f"Stop loss: {price_change*100:.2f}% loss"
        
        return None


class BTCStrategy(CoinStrategy):
    """Bitcoin-specific strategy - Conservative, trend-following."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('BTC', config)
        
        # BTC-specific parameters (optimized from backtest)
        self.ema_fast = 8  # Slower EMAs for less noise
        self.ema_slow = 21
        self.momentum_threshold = 0.0015  # 0.15% - lowered from 0.2%
        self.take_profit_pct = 0.015      # 1.5% - larger profits
        self.stop_loss_pct = 0.008        # 0.8% - wider stop loss
        self.max_hold_periods = 15        # Shorter hold time (was 20)
        self.volume_filter = 1.5          # Higher volume requirement
        self.rsi_oversold = 25            # More extreme RSI levels
        self.rsi_overbought = 75
        
        print(f"🟡 BTC Strategy: Conservative trend-following (OPTIMIZED)")
        print(f"   Momentum: {self.momentum_threshold*100:.2f}% | Profit: {self.take_profit_pct*100:.1f}% | Hold: {self.max_hold_periods}p")
    
    def get_signal_strength(self, current_price: float) -> float:
        """BTC signal strength calculation."""
        if len(self.ema_fast_history) < 2 or len(self.ema_slow_history) < 2:
            return 0.0
        
        ema_fast_current = self.ema_fast_history[-1]
        ema_slow_current = self.ema_slow_history[-1]
        ema_fast_prev = self.ema_fast_history[-2]
        ema_slow_prev = self.ema_slow_history[-2]
        
        # Momentum signal
        momentum = (ema_fast_current - ema_slow_current) / ema_slow_current if ema_slow_current > 0 else 0
        
        # Strong trend confirmation
        trend_strength = 0.0
        if ema_fast_prev <= ema_slow_prev and ema_fast_current > ema_slow_current:
            trend_strength = 0.4  # Strong bullish crossover
        
        # RSI confirmation
        rsi = self.calculate_rsi(self.price_history)
        rsi_signal = 0.0
        if rsi < self.rsi_oversold:
            rsi_signal = 0.3  # Strong oversold
        elif rsi < 40:
            rsi_signal = 0.1  # Mild oversold
        
        return momentum + trend_strength + rsi_signal


class ETHStrategy(CoinStrategy):
    """Ethereum-specific strategy - Balanced momentum."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('ETH', config)
        
        # ETH-specific parameters (balanced approach)
        self.ema_fast = 5
        self.ema_slow = 15
        self.momentum_threshold = 0.0015  # 0.15%
        self.take_profit_pct = 0.012      # 1.2%
        self.stop_loss_pct = 0.006        # 0.6%
        self.max_hold_periods = 18
        self.volume_filter = 1.3
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        print(f"💙 ETH Strategy: Balanced momentum (OPTIMAL - NO CHANGES)")
        print(f"   Momentum: {self.momentum_threshold*100:.2f}% | Profit: {self.take_profit_pct*100:.1f}% | Hold: {self.max_hold_periods}p")
    
    def get_signal_strength(self, current_price: float) -> float:
        """ETH signal strength calculation."""
        if len(self.ema_fast_history) < 2 or len(self.ema_slow_history) < 2:
            return 0.0
        
        ema_fast_current = self.ema_fast_history[-1]
        ema_slow_current = self.ema_slow_history[-1]
        ema_fast_prev = self.ema_fast_history[-2]
        ema_slow_prev = self.ema_slow_history[-2]
        
        # Momentum calculation
        momentum = (ema_fast_current - ema_slow_current) / ema_slow_current if ema_slow_current > 0 else 0
        
        # Crossover signals
        crossover_strength = 0.0
        if ema_fast_prev <= ema_slow_prev and ema_fast_current > ema_slow_current:
            crossover_strength = 0.25  # Bullish crossover
        
        # RSI with balanced levels
        rsi = self.calculate_rsi(self.price_history)
        rsi_signal = 0.0
        if rsi < 30:
            rsi_signal = 0.2
        elif rsi < 45:
            rsi_signal = 0.1
        
        # Volume momentum (ETH often leads with volume)
        volume_signal = 0.0
        if len(self.volume_history) >= 3:
            recent_vol = sum(list(self.volume_history)[-3:]) / 3
            avg_vol = sum(list(self.volume_history)[-10:]) / 10
            if recent_vol > avg_vol * 1.5:
                volume_signal = 0.1
        
        return momentum + crossover_strength + rsi_signal + volume_signal


class SOLStrategy(CoinStrategy):
    """Solana-specific strategy - High-frequency, volatility-adapted."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('SOL', config)
        
        # SOL-specific parameters (optimized from backtest)
        self.ema_fast = 3   # Very fast EMAs for quick reactions
        self.ema_slow = 8
        self.momentum_threshold = 0.0012  # 0.12% - raised from 0.08%
        self.take_profit_pct = 0.008      # 0.8% - quick profits
        self.stop_loss_pct = 0.006        # 0.6% - wider stop loss (was 0.4%)
        self.max_hold_periods = 10        # Quick exits
        self.volume_filter = 1.0          # No volume filter (volatile volume)
        self.rsi_oversold = 35            # Less extreme RSI
        self.rsi_overbought = 65
        
        print(f"🟣 SOL Strategy: High-frequency volatility scalping (OPTIMIZED)")
        print(f"   Momentum: {self.momentum_threshold*100:.2f}% | Stop: {self.stop_loss_pct*100:.1f}% | Hold: {self.max_hold_periods}p")
    
    def get_signal_strength(self, current_price: float) -> float:
        """SOL signal strength calculation - optimized for volatility."""
        if len(self.ema_fast_history) < 2 or len(self.ema_slow_history) < 2:
            return 0.0
        
        ema_fast_current = self.ema_fast_history[-1]
        ema_slow_current = self.ema_slow_history[-1]
        ema_fast_prev = self.ema_fast_history[-2] if len(self.ema_fast_history) > 1 else ema_fast_current
        ema_slow_prev = self.ema_slow_history[-2] if len(self.ema_slow_history) > 1 else ema_slow_current
        
        # Fast momentum for quick moves
        momentum = (ema_fast_current - ema_slow_current) / ema_slow_current if ema_slow_current > 0 else 0
        
        # Quick reversal signals
        momentum_accel = 0.0
        if len(self.price_history) >= 3:
            recent_change = (self.price_history[-1] - self.price_history[-3]) / self.price_history[-3]
            if recent_change > 0.005:  # 0.5% move in 3 periods
                momentum_accel = 0.2
        
        # RSI mean reversion (SOL often bounces quickly)
        rsi = self.calculate_rsi(self.price_history)
        rsi_signal = 0.0
        if rsi < 35:
            rsi_signal = 0.25  # Strong mean reversion signal
        elif rsi < 50:
            rsi_signal = 0.1
        
        # Volatility boost (SOL loves volatility)
        volatility_signal = 0.0
        if len(self.price_history) >= 5:
            recent_prices = list(self.price_history)[-5:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            if volatility > 0.02:  # High volatility
                volatility_signal = 0.1
        
        return momentum + momentum_accel + rsi_signal + volatility_signal


class XRPStrategy(CoinStrategy):
    """XRP-specific strategy - Trend-following with breakouts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('XRP', config)
        
        # XRP-specific parameters (optimized from backtest)
        self.ema_fast = 7
        self.ema_slow = 18
        self.momentum_threshold = 0.001   # 0.1%
        self.take_profit_pct = 0.015      # 1.5% - lowered from 2.0%
        self.stop_loss_pct = 0.005        # 0.5% - tighter (was 0.7%)
        self.max_hold_periods = 25        # Patient holding
        self.volume_filter = 1.8          # High volume requirement for breakouts
        self.rsi_oversold = 28
        self.rsi_overbought = 72
        
        print(f"🟠 XRP Strategy: Patient trend-following with breakouts (OPTIMIZED)")
        print(f"   Profit: {self.take_profit_pct*100:.1f}% | Stop: {self.stop_loss_pct*100:.1f}% | Hold: {self.max_hold_periods}p")
    
    def get_signal_strength(self, current_price: float) -> float:
        """XRP signal strength - focus on strong trends and breakouts."""
        if len(self.ema_fast_history) < 2 or len(self.ema_slow_history) < 2:
            return 0.0
        
        ema_fast_current = self.ema_fast_history[-1]
        ema_slow_current = self.ema_slow_history[-1]
        ema_fast_prev = self.ema_fast_history[-2]
        ema_slow_prev = self.ema_slow_history[-2]
        
        # Strong trend momentum
        momentum = (ema_fast_current - ema_slow_current) / ema_slow_current if ema_slow_current > 0 else 0
        
        # Breakout signals (XRP often has explosive moves)
        breakout_signal = 0.0
        if len(self.price_history) >= 10:
            recent_high = max(list(self.price_history)[-10:])
            if current_price >= recent_high * 1.005:  # Breaking recent high
                breakout_signal = 0.3
        
        # Strong RSI oversold (XRP bounces hard)
        rsi = self.calculate_rsi(self.price_history)
        rsi_signal = 0.0
        if rsi < 28:
            rsi_signal = 0.4  # Very strong signal
        elif rsi < 40:
            rsi_signal = 0.15
        
        # Volume surge detection
        volume_surge = 0.0
        if len(self.volume_history) >= 5:
            recent_vol = self.volume_history[-1]
            avg_vol = sum(list(self.volume_history)[-10:]) / 10
            if recent_vol > avg_vol * 2.0:  # 2x volume surge
                volume_surge = 0.2
        
        return momentum + breakout_signal + rsi_signal + volume_surge


class MultiCoinStrategy:
    """Main strategy manager that routes each coin to its specific strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-coin strategy system."""
        self.config = config
        
        # Initialize individual coin strategies
        self.strategies = {
            'BTC': BTCStrategy(config),
            'ETH': ETHStrategy(config),
            'SOL': SOLStrategy(config),
            'XRP': XRPStrategy(config)
        }
        
        print("🚀 Multi-Coin Strategy System Initialized")
        print(f"   Tracking {len(self.strategies)} coin-specific strategies")
    
    def analyze_market(self, market_data: Dict[str, Any], portfolio=None, last_trade_prices=None) -> List[TradeSignal]:
        """Analyze market using coin-specific strategies."""
        signals = []
        current_time = time.time()
        
        for coin, data in market_data.items():
            if coin not in self.strategies:
                continue
            
            try:
                current_price = data.get('price', 0)
                volume = data.get('volume_24h', 0)
                
                if current_price <= 0:
                    continue
                
                strategy = self.strategies[coin]
                
                # Update indicators
                strategy.update_indicators(current_price, volume)
                
                # Update position tracking
                if strategy.position['quantity'] != 0:
                    strategy.position['periods_held'] += 1
                
                # Check exit conditions first
                exit_reason = strategy.check_exit_conditions(current_price)
                if exit_reason and strategy.position['quantity'] != 0:
                    signal = TradeSignal(
                        coin=coin,
                        action=TradeAction.SELL,
                        quantity=abs(strategy.position['quantity']),
                        price=current_price,
                        reason=f"{coin} Strategy Exit: {exit_reason}",
                        tier=0
                    )
                    signal.strength = 0.8
                    signals.append(signal)
                    
                    # Clear position
                    strategy.position = {
                        'entry_price': 0.0,
                        'entry_time': 0,
                        'quantity': 0.0,
                        'periods_held': 0,
                        'position_type': None
                    }
                    continue
                
                # Skip if already in position
                if strategy.position['quantity'] != 0:
                    continue
                
                # Check volume filter
                if not strategy.check_volume_filter():
                    continue
                
                # Get coin-specific signal strength
                signal_strength = strategy.get_signal_strength(current_price)
                
                # Debug output
                if len(strategy.price_history) % 20 == 0:  # Every 20 cycles
                    rsi = strategy.calculate_rsi(strategy.price_history)
                    print(f"🔍 {coin}: strength={signal_strength:.4f}, threshold={strategy.momentum_threshold:.4f}, RSI={rsi:.1f}")
                
                # Generate entry signals
                if signal_strength > strategy.momentum_threshold:
                    if portfolio:
                        balance = portfolio.get_usdt_balance()
                        
                        # Use coin-specific position sizing
                        position_pct = 0.08  # 8% default
                        if coin == 'BTC':
                            position_pct = 0.12  # Larger BTC positions
                        elif coin == 'SOL':
                            position_pct = 0.06  # Smaller SOL positions (higher risk)
                        
                        trade_amount = balance * position_pct
                        
                        if trade_amount >= 10.0:
                            quantity = trade_amount / current_price
                            
                            signal = TradeSignal(
                                coin=coin,
                                action=TradeAction.BUY,
                                quantity=quantity,
                                price=current_price,
                                reason=f"{coin} Strategy Buy: {signal_strength*100:.2f}% strength",
                                tier=0
                            )
                            signal.strength = min(0.9, 0.6 + abs(signal_strength))
                            signals.append(signal)
                            
                            # Track position
                            strategy.position = {
                                'entry_price': current_price,
                                'entry_time': current_time,
                                'quantity': quantity,
                                'periods_held': 0,
                                'position_type': 'LONG'
                            }
                
            except Exception as e:
                print(f"Error analyzing {coin}: {e}")
                continue
        
        return signals
    
    def on_trade_executed(self, signal: TradeSignal):
        """Handle post-trade execution updates."""
        coin = signal.coin
        if coin in self.strategies:
            strategy = self.strategies[coin]
            
            if signal.action == TradeAction.SELL:
                if strategy.position['entry_price'] > 0:
                    profit_pct = (signal.price - strategy.position['entry_price']) / strategy.position['entry_price'] * 100
                    print(f"📊 {coin} {strategy.__class__.__name__} closed: {profit_pct:+.2f}% | "
                          f"Held: {strategy.position['periods_held']} periods")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get current strategy state summary."""
        active_positions = {}
        for coin, strategy in self.strategies.items():
            if strategy.position['quantity'] != 0:
                active_positions[coin] = strategy.position
        
        return {
            'strategy_name': 'Multi-Coin Optimized Strategies',
            'active_positions': len(active_positions),
            'positions': active_positions,
            'coin_strategies': {
                coin: {
                    'name': strategy.__class__.__name__,
                    'momentum_threshold': strategy.momentum_threshold,
                    'take_profit': strategy.take_profit_pct,
                    'stop_loss': strategy.stop_loss_pct
                }
                for coin, strategy in self.strategies.items()
            }
        }
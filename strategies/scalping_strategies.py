"""
Collection of scalping and small-win trading strategies.
Designed for frequent, consistent profits rather than large gains.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class StrategyType(Enum):
    """Types of trading strategies."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM_SCALP = "momentum_scalp"
    RSI_BOUNCE = "rsi_bounce"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"
    VOLUME_BREAKOUT = "volume_breakout"
    GRID_TRADING = "grid_trading"


@dataclass
class StrategyParams:
    """Parameters for a trading strategy."""
    name: str
    type: StrategyType
    params: Dict[str, Any]
    description: str


class ScalpingStrategies:
    """Collection of scalping strategies for small consistent wins."""
    
    @staticmethod
    def get_all_strategies() -> List[StrategyParams]:
        """Get all available scalping strategies."""
        return [
            # 1. RSI Mean Reversion (Quick bounces)
            StrategyParams(
                name="RSI_MeanReversion_Quick",
                type=StrategyType.RSI_BOUNCE,
                params={
                    'rsi_period': 14,
                    'rsi_oversold': 25,
                    'rsi_overbought': 75,
                    'take_profit_pct': 0.015,  # 1.5% profit target
                    'stop_loss_pct': 0.008,    # 0.8% stop loss
                    'min_volume_ratio': 1.2    # Volume > 1.2x average
                },
                description="Buy oversold, sell overbought with tight stops"
            ),
            
            # 2. Bollinger Band Squeeze Breakout
            StrategyParams(
                name="Bollinger_Squeeze_Scalp",
                type=StrategyType.BOLLINGER_SQUEEZE,
                params={
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'squeeze_threshold': 0.02,  # 2% band width
                    'breakout_threshold': 0.005, # 0.5% breakout
                    'take_profit_pct': 0.012,   # 1.2% profit
                    'stop_loss_pct': 0.006,     # 0.6% stop
                    'volume_confirmation': True
                },
                description="Trade breakouts from tight Bollinger bands"
            ),
            
            # 3. Volume Spike Momentum
            StrategyParams(
                name="Volume_Momentum_Scalp",
                type=StrategyType.VOLUME_BREAKOUT,
                params={
                    'volume_spike_ratio': 2.0,  # 2x average volume
                    'price_move_threshold': 0.008, # 0.8% price move
                    'momentum_period': 5,        # 5-period momentum
                    'take_profit_pct': 0.018,    # 1.8% profit
                    'stop_loss_pct': 0.010,      # 1.0% stop
                    'max_hold_periods': 10       # Max 10 periods
                },
                description="Trade volume spikes with price momentum"
            ),
            
            # 4. Grid Trading (Range-bound markets)
            StrategyParams(
                name="Grid_Trading_Scalp",
                type=StrategyType.GRID_TRADING,
                params={
                    'grid_spacing': 0.01,       # 1% grid spacing
                    'num_grid_levels': 5,       # 5 levels up/down
                    'base_order_size': 0.1,     # 10% of capital per trade
                    'take_profit_pct': 0.01,    # 1% profit per grid
                    'volatility_filter': 0.03,  # Only trade if vol < 3%
                    'trend_filter': True        # Avoid strong trends
                },
                description="Grid trading for sideways markets"
            ),
            
            # 5. Quick Momentum Scalp
            StrategyParams(
                name="Quick_Momentum_Scalp",
                type=StrategyType.MOMENTUM_SCALP,
                params={
                    'ema_fast': 5,
                    'ema_slow': 13,
                    'momentum_threshold': 0.005, # 0.5% momentum
                    'take_profit_pct': 0.01,     # 1% profit
                    'stop_loss_pct': 0.005,      # 0.5% stop
                    'max_hold_periods': 8,       # Quick exits
                    'volume_filter': 1.5         # Volume > 1.5x avg
                },
                description="Quick momentum trades with fast exits"
            ),
            
            # 6. RSI Divergence Scalp
            StrategyParams(
                name="RSI_Divergence_Scalp",
                type=StrategyType.RSI_BOUNCE,
                params={
                    'rsi_period': 14,
                    'divergence_periods': 10,    # Look back 10 periods
                    'rsi_extreme_low': 30,
                    'rsi_extreme_high': 70,
                    'take_profit_pct': 0.02,     # 2% profit
                    'stop_loss_pct': 0.01,       # 1% stop
                    'confirmation_periods': 2    # Wait 2 periods
                },
                description="Trade RSI divergences for quick reversals"
            ),
            
            # 7. Support/Resistance Bounce
            StrategyParams(
                name="SR_Bounce_Scalp",
                type=StrategyType.MEAN_REVERSION,
                params={
                    'lookback_periods': 50,      # Find S/R over 50 periods
                    'touch_threshold': 0.002,    # 0.2% from S/R level
                    'bounce_confirmation': 0.003, # 0.3% bounce required
                    'take_profit_pct': 0.015,    # 1.5% profit
                    'stop_loss_pct': 0.008,      # 0.8% stop
                    'min_touches': 2             # Min 2 prior touches
                },
                description="Trade bounces off support/resistance levels"
            ),
            
            # 8. VWAP Mean Reversion
            StrategyParams(
                name="VWAP_MeanReversion",
                type=StrategyType.MEAN_REVERSION,
                params={
                    'vwap_period': 20,
                    'deviation_threshold': 0.01,  # 1% from VWAP
                    'return_threshold': 0.003,    # 0.3% return to VWAP
                    'take_profit_pct': 0.012,     # 1.2% profit
                    'stop_loss_pct': 0.008,       # 0.8% stop
                    'volume_confirmation': True   # Need volume confirmation
                },
                description="Trade returns to VWAP with volume confirmation"
            )
        ]


class StrategyBacktester:
    """Backtest trading strategies using vectorbt."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy backtester."""
        self.config = config
        self.strategies = ScalpingStrategies.get_all_strategies()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        
        return upper_band, sma, lower_band
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    def calculate_vwap(self, prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = prices  # Simplified, would use (H+L+C)/3 with OHLC data
        return (typical_price * volumes).rolling(window=period).sum() / volumes.rolling(window=period).sum()
    
    def generate_signals_rsi_bounce(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate signals for RSI bounce strategy."""
        prices = data['close']
        volumes = data['volume']
        
        rsi = self.calculate_rsi(prices, params['rsi_period'])
        vol_avg = volumes.rolling(window=20).mean()
        vol_ratio = volumes / vol_avg
        
        # Buy signals: RSI oversold + volume confirmation
        buy_signals = (
            (rsi < params['rsi_oversold']) & 
            (vol_ratio > params.get('min_volume_ratio', 1.0))
        )
        
        # Sell signals: RSI overbought or take profit/stop loss
        sell_signals = (rsi > params['rsi_overbought'])
        
        return buy_signals, sell_signals
    
    def generate_signals_bollinger_squeeze(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate signals for Bollinger squeeze strategy."""
        prices = data['close']
        volumes = data['volume']
        
        upper, middle, lower = self.calculate_bollinger_bands(
            prices, params['bb_period'], params['bb_std']
        )
        
        # Band width as percentage of middle
        band_width = (upper - lower) / middle
        is_squeeze = band_width < params['squeeze_threshold']
        
        # Breakout signals
        price_above_upper = prices > upper * (1 + params['breakout_threshold'])
        price_below_lower = prices < lower * (1 - params['breakout_threshold'])
        
        vol_confirmation = True
        if params.get('volume_confirmation'):
            vol_avg = volumes.rolling(window=20).mean()
            vol_confirmation = volumes > vol_avg * 1.2
        
        buy_signals = is_squeeze.shift(1) & price_above_upper & vol_confirmation
        sell_signals = is_squeeze.shift(1) & price_below_lower & vol_confirmation
        
        return buy_signals, sell_signals
    
    def generate_signals_volume_breakout(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate signals for volume breakout strategy."""
        prices = data['close']
        volumes = data['volume']
        
        vol_avg = volumes.rolling(window=20).mean()
        vol_spike = volumes > vol_avg * params['volume_spike_ratio']
        
        price_change = prices.pct_change(params['momentum_period'])
        strong_move_up = price_change > params['price_move_threshold']
        strong_move_down = price_change < -params['price_move_threshold']
        
        buy_signals = vol_spike & strong_move_up
        sell_signals = vol_spike & strong_move_down
        
        return buy_signals, sell_signals
    
    def generate_signals_momentum_scalp(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate signals for momentum scalp strategy."""
        prices = data['close']
        volumes = data['volume']
        
        ema_fast = self.calculate_ema(prices, params['ema_fast'])
        ema_slow = self.calculate_ema(prices, params['ema_slow'])
        
        momentum = (ema_fast - ema_slow) / ema_slow
        
        vol_avg = volumes.rolling(window=20).mean()
        vol_filter = volumes > vol_avg * params['volume_filter']
        
        buy_signals = (
            (momentum > params['momentum_threshold']) & 
            vol_filter &
            (ema_fast > ema_slow)
        )
        
        sell_signals = (
            (momentum < -params['momentum_threshold']) |
            (ema_fast < ema_slow)
        )
        
        return buy_signals, sell_signals
    
    def generate_signals_mean_reversion(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate signals for mean reversion strategies."""
        prices = data['close']
        volumes = data['volume']
        
        if 'vwap_period' in params:
            # VWAP mean reversion
            vwap = self.calculate_vwap(prices, volumes, params['vwap_period'])
            deviation = (prices - vwap) / vwap
            
            buy_signals = deviation < -params['deviation_threshold']
            sell_signals = deviation > params['deviation_threshold']
            
        else:
            # Support/Resistance bounce
            rolling_min = prices.rolling(window=params['lookback_periods']).min()
            rolling_max = prices.rolling(window=params['lookback_periods']).max()
            
            near_support = (prices - rolling_min) / rolling_min < params['touch_threshold']
            near_resistance = (rolling_max - prices) / rolling_max < params['touch_threshold']
            
            buy_signals = near_support
            sell_signals = near_resistance
        
        return buy_signals, sell_signals
    
    def generate_signals_for_strategy(self, data: pd.DataFrame, strategy: StrategyParams) -> Tuple[pd.Series, pd.Series]:
        """Generate buy/sell signals for a given strategy."""
        if strategy.type == StrategyType.RSI_BOUNCE:
            return self.generate_signals_rsi_bounce(data, strategy.params)
        elif strategy.type == StrategyType.BOLLINGER_SQUEEZE:
            return self.generate_signals_bollinger_squeeze(data, strategy.params)
        elif strategy.type == StrategyType.VOLUME_BREAKOUT:
            return self.generate_signals_volume_breakout(data, strategy.params)
        elif strategy.type == StrategyType.MOMENTUM_SCALP:
            return self.generate_signals_momentum_scalp(data, strategy.params)
        elif strategy.type == StrategyType.MEAN_REVERSION:
            return self.generate_signals_mean_reversion(data, strategy.params)
        else:
            # Default to simple momentum
            return self.generate_signals_momentum_scalp(data, strategy.params)
    
    def backtest_strategy(self, strategy: StrategyParams, price_data: pd.DataFrame, 
                         initial_capital: float = 10000) -> Dict[str, Any]:
        """Backtest a single strategy."""
        try:
            import vectorbt as vbt
            
            # Generate signals
            buy_signals, sell_signals = self.generate_signals_for_strategy(price_data, strategy)
            
            # Ensure we have valid signals
            if buy_signals.sum() == 0 and sell_signals.sum() == 0:
                return {
                    'strategy_name': strategy.name,
                    'error': 'No signals generated',
                    'total_trades': 0,
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            # Create portfolio
            portfolio = vbt.Portfolio.from_signals(
                price_data['close'],
                buy_signals,
                sell_signals,
                init_cash=initial_capital,
                fees=0.001,  # 0.1% fee
                freq='1h'
            )
            
            # Calculate metrics
            stats = portfolio.stats()
            
            def safe_stat(stat_name, default=0.0):
                try:
                    value = stats.get(stat_name, default)
                    return float(value) if not pd.isna(value) else default
                except:
                    return default
            
            # Calculate win rate manually if needed
            trades = portfolio.trades.records_readable
            if len(trades) > 0:
                winning_trades = len(trades[trades['PnL'] > 0])
                total_trades = len(trades)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            else:
                win_rate = 0
                total_trades = 0
            
            # Calculate average trade duration and frequency
            avg_trade_duration = 0
            trade_frequency = 0
            if len(trades) > 0:
                durations = (trades['Exit Timestamp'] - trades['Entry Timestamp']).dt.total_seconds() / 3600  # Hours
                avg_trade_duration = durations.mean()
                total_time_hours = (price_data.index[-1] - price_data.index[0]).total_seconds() / 3600
                trade_frequency = len(trades) / (total_time_hours / 24)  # Trades per day
            
            return {
                'strategy_name': strategy.name,
                'strategy_type': strategy.type.value,
                'description': strategy.description,
                'total_return': safe_stat('Total Return [%]'),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'max_drawdown': safe_stat('Max Drawdown [%]'),
                'sharpe_ratio': safe_stat('Sharpe Ratio'),
                'calmar_ratio': safe_stat('Calmar Ratio'),
                'profit_factor': safe_stat('Profit Factor', 1.0),
                'avg_trade_duration_hours': avg_trade_duration,
                'trade_frequency_per_day': trade_frequency,
                'final_value': float(portfolio.value().iloc[-1]),
                'buy_signals': int(buy_signals.sum()),
                'sell_signals': int(sell_signals.sum()),
                'params': strategy.params
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy.name,
                'error': str(e),
                'total_trades': 0,
                'total_return': 0.0
            }
    
    def compare_all_strategies(self, coin: str = 'BTC', days: int = 30) -> List[Dict[str, Any]]:
        """Compare all strategies on a given coin."""
        # Get historical data
        from core.vectorbt_backtester import VectorBTBacktester
        vbt_backtester = VectorBTBacktester(self.config)
        
        # Fetch historical data
        raw_data = vbt_backtester.fetch_historical_data(coin, days)
        if raw_data.empty:
            print(f"No data available for {coin}")
            return []
        
        # Add volume column if missing
        if 'volume' not in raw_data.columns:
            raw_data['volume'] = raw_data['close'] * 1000000  # Simplified volume
        
        results = []
        
        print(f"🧪 Testing {len(self.strategies)} strategies on {coin} ({days} days)")
        
        for strategy in self.strategies:
            print(f"  📊 Testing {strategy.name}...")
            result = self.backtest_strategy(strategy, raw_data)
            results.append(result)
            
            if 'error' not in result:
                print(f"    ✅ Return: {result['total_return']:.2f}%, "
                      f"Trades: {result['total_trades']}, "
                      f"Win Rate: {result['win_rate']:.1f}%")
            else:
                print(f"    ❌ Error: {result['error']}")
        
        # Sort by total return descending
        results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        return results
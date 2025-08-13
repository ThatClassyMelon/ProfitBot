"""
Multi-Signal Enhanced Trading Strategy with comprehensive market analysis.
Integrates price, volume, technical indicators, sentiment, and market data for institutional-grade decisions.
"""
import time
import math
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import deque
from core.enhanced_strategy import TradeAction, TradeSignal, CoinState
from core.advanced_data_fetcher import AdvancedDataFetcher, MarketData, MarketSentiment
from core.trade_explainer import TradeExplainer
from utils.math_tools import percent_change, calculate_trade_amount, round_to_precision


class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


class MultiSignalStrategy:
    """
    Advanced multi-signal trading strategy that combines:
    - Price action and volatility
    - Volume and liquidity analysis
    - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
    - Market sentiment (Fear & Greed, BTC dominance)
    - Social and developer metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-signal strategy."""
        self.config = config
        strategy_config = config['strategy']
        
        # Core parameters
        self.base_trade_percentage = strategy_config['trade_amount_percentage']
        self.min_trade_amount = strategy_config['min_trade_amount']
        
        # Enhanced parameters
        self.k_factor = strategy_config.get('adaptive_k_factor', 1.5)
        self.baseline_weight = strategy_config.get('baseline_update_weight', 0.3)
        
        # Multi-signal parameters
        self.min_signal_strength = strategy_config.get('min_signal_strength', 0.6)
        self.volume_threshold = strategy_config.get('min_volume_threshold', 1000000)  # $1M
        self.rsi_overbought = strategy_config.get('rsi_overbought', 70)
        self.rsi_oversold = strategy_config.get('rsi_oversold', 30)
        
        # Sentiment adjustments
        self.fear_greed_multipliers = strategy_config.get('fear_greed_multipliers', {
            'extreme_fear': 0.7,    # More aggressive when everyone is fearful
            'fear': 0.85,
            'neutral': 1.0,
            'greed': 1.15,
            'extreme_greed': 1.4    # More conservative when everyone is greedy
        })
        
        # DCA and risk management
        self.dca_multipliers = strategy_config.get('dca_multipliers', [0.10, 0.15, 0.20])
        self.max_dca_levels = len(self.dca_multipliers)
        self.max_hold_ticks = strategy_config.get('max_hold_ticks', 300)
        
        # Profit rebalancing
        self.rebalance_profit_threshold = strategy_config.get('rebalance_profit_threshold', 0.01)
        self.rebalance_percentage = strategy_config.get('rebalance_percentage', 0.10)
        self.trades_since_rebalance = 0
        self.last_portfolio_value = 0.0
        
        # Initialize advanced data fetcher and trade explainer
        self.data_fetcher = AdvancedDataFetcher(config)
        self.trade_explainer = TradeExplainer()
        
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
        Advanced market analysis with multi-signal approach.
        """
        signals = []
        current_time = time.time()
        
        # Fetch comprehensive market data
        market_data, market_sentiment = self.data_fetcher.fetch_all_data()
        
        # Update price histories (for fallback when live data unavailable)
        for coin, price in current_prices.items():
            if coin in self.coin_states:
                self.coin_states[coin].update_price_history(price)
        
        # Check for profit rebalancing first
        rebalance_signal = self._check_profit_rebalancing(current_prices, portfolio)
        if rebalance_signal:
            signals.append(rebalance_signal)
        
        # Analyze each coin with multi-signal approach
        for coin in self.coin_states.keys():
            current_price = current_prices.get(coin, 0.0)
            if current_price <= 0:
                continue
            
            signal = self._analyze_coin_multi_signal(
                coin, current_price, portfolio, current_time, 
                market_data.get(coin), market_sentiment
            )
            
            # Add explanation to the signal
            signal.explanation = self._get_trade_explanation(signal, market_data.get(coin), market_sentiment)
            
            signals.append(signal)
        
        return signals
    
    def _analyze_coin_multi_signal(self, coin: str, current_price: float, 
                                 portfolio, current_time: float,
                                 market_data: Optional[MarketData],
                                 market_sentiment: MarketSentiment) -> TradeSignal:
        """
        Comprehensive multi-signal analysis for a single coin.
        """
        coin_state = self.coin_states[coin]
        
        # Get sentiment-adjusted threshold
        adaptive_threshold = self._get_sentiment_adjusted_threshold(coin_state, market_sentiment)
        
        # Calculate base price change
        price_change = percent_change(coin_state.baseline_price, current_price)
        
        # Multi-signal analysis
        buy_signal_strength = 0.0
        sell_signal_strength = 0.0
        analysis_details = []
        
        # 1. Price Action Analysis
        if price_change <= -adaptive_threshold:
            buy_signal_strength += 0.3
            analysis_details.append(f"Price drop {abs(price_change)*100:.1f}%")
        elif price_change >= adaptive_threshold:
            sell_signal_strength += 0.3
            analysis_details.append(f"Price rise {price_change*100:.1f}%")
        
        # 2. Volume Analysis
        if market_data:
            volume_analysis = self.data_fetcher.get_volume_analysis(coin)
            if volume_analysis.get('sufficient_liquidity', False):
                volume_strength = volume_analysis.get('volume_strength', 'LOW')
                if volume_strength == 'HIGH':
                    buy_signal_strength += 0.2
                    sell_signal_strength += 0.2
                    analysis_details.append("High volume")
                elif volume_strength == 'MEDIUM':
                    buy_signal_strength += 0.1
                    sell_signal_strength += 0.1
                    analysis_details.append("Medium volume")
            else:
                # Penalize low liquidity
                buy_signal_strength -= 0.2
                sell_signal_strength -= 0.2
                analysis_details.append("Low liquidity")
        
        # 3. Technical Indicator Analysis
        if market_data:
            ta_analysis = self.data_fetcher.get_technical_analysis(coin)
            signals = ta_analysis.get('signals', {})
            
            # RSI Analysis
            rsi_signal = signals.get('rsi_signal', 'NEUTRAL')
            if rsi_signal == 'OVERSOLD':
                buy_signal_strength += 0.25
                analysis_details.append("RSI oversold")
            elif rsi_signal == 'OVERBOUGHT':
                sell_signal_strength += 0.25
                analysis_details.append("RSI overbought")
            
            # MACD Analysis
            macd_signal = signals.get('macd_signal', 'NEUTRAL')
            if macd_signal == 'BULLISH':
                buy_signal_strength += 0.15
                analysis_details.append("MACD bullish")
            elif macd_signal == 'BEARISH':
                sell_signal_strength += 0.15
                analysis_details.append("MACD bearish")
            
            # Bollinger Bands Analysis
            bb_signal = signals.get('bb_signal', 'NEUTRAL')
            if bb_signal == 'OVERSOLD':
                buy_signal_strength += 0.2
                analysis_details.append("BB oversold")
            elif bb_signal == 'OVERBOUGHT':
                sell_signal_strength += 0.2
                analysis_details.append("BB overbought")
            
            # Moving Average Analysis
            ma_signal = signals.get('ma_signal', 'NEUTRAL')
            if ma_signal == 'BULLISH':
                buy_signal_strength += 0.1
                analysis_details.append("MA bullish")
            elif ma_signal == 'BEARISH':
                sell_signal_strength += 0.1
                analysis_details.append("MA bearish")
        
        # 4. Market Sentiment Analysis
        sentiment_analysis = self.data_fetcher.get_sentiment_analysis()
        fear_greed = sentiment_analysis.get('fear_greed_classification', 'neutral')
        
        if fear_greed in ['extreme fear', 'fear']:
            buy_signal_strength += 0.15
            analysis_details.append(f"Market {fear_greed}")
        elif fear_greed in ['extreme greed', 'greed']:
            sell_signal_strength += 0.1
            buy_signal_strength -= 0.1
            analysis_details.append(f"Market {fear_greed}")
        
        # 5. Social/Developer Score Analysis
        if market_data and market_data.social_score:
            if market_data.social_score > 0.7:
                buy_signal_strength += 0.05
                sell_signal_strength += 0.05
                analysis_details.append("High social activity")
        
        # Determine action based on signal strengths
        analysis_summary = " | ".join(analysis_details) if analysis_details else "Insufficient signals"
        
        # Check signal strength thresholds
        if buy_signal_strength >= self.min_signal_strength and buy_signal_strength > sell_signal_strength:
            return self._generate_multi_signal_buy(
                coin, current_price, portfolio, coin_state, 
                buy_signal_strength, analysis_summary
            )
        elif sell_signal_strength >= self.min_signal_strength and sell_signal_strength > buy_signal_strength:
            return self._generate_multi_signal_sell(
                coin, current_price, portfolio, coin_state,
                sell_signal_strength, analysis_summary
            )
        
        # Check for timed sell fallback
        elif self._should_timed_sell(coin_state, current_price, current_time):
            return self._generate_timed_sell_signal(coin, current_price, portfolio, coin_state)
        
        # Hold signal
        else:
            movement_needed = adaptive_threshold - abs(price_change)
            signal_strength_needed = self.min_signal_strength - max(buy_signal_strength, sell_signal_strength)
            
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Signals: {max(buy_signal_strength, sell_signal_strength):.2f}/{self.min_signal_strength:.2f} | {analysis_summary}"
            )
    
    def _get_sentiment_adjusted_threshold(self, coin_state: CoinState, 
                                        market_sentiment: MarketSentiment) -> float:
        """Calculate threshold adjusted for market sentiment."""
        base_threshold = coin_state.get_adaptive_threshold(self.k_factor)
        
        # Adjust based on Fear & Greed Index
        sentiment_multiplier = 1.0
        if market_sentiment.fear_greed_classification:
            classification = market_sentiment.fear_greed_classification.lower().replace(' ', '_')
            sentiment_multiplier = self.fear_greed_multipliers.get(classification, 1.0)
        
        return base_threshold * sentiment_multiplier
    
    def _generate_multi_signal_buy(self, coin: str, current_price: float, 
                                 portfolio, coin_state: CoinState,
                                 signal_strength: float, analysis: str) -> TradeSignal:
        """Generate buy signal with multi-signal validation."""
        
        # Check DCA level limits
        if coin_state.dca_level >= self.max_dca_levels:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Max DCA level reached ({coin_state.dca_level}/{self.max_dca_levels})"
            )
        
        # Calculate signal-strength-adjusted trade amount
        usdt_balance = portfolio.get_usdt_balance()
        base_multiplier = self.dca_multipliers[coin_state.dca_level]
        
        # Adjust trade size based on signal strength
        strength_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5 to 1.0 range
        trade_multiplier = base_multiplier * strength_multiplier
        
        trade_amount_usdt = usdt_balance * trade_multiplier
        
        if trade_amount_usdt < self.min_trade_amount:
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Insufficient balance for signal-adjusted trade (need ${self.min_trade_amount})"
            )
        
        # Calculate quantity
        quantity = trade_amount_usdt / current_price
        
        # Update coin state
        coin_state.dca_level += 1
        coin_state.last_buy_price = current_price
        coin_state.last_buy_time = time.time()
        coin_state.total_buy_count += 1
        coin_state.update_baseline_weighted(current_price, self.baseline_weight)
        
        return TradeSignal(
            coin, TradeAction.BUY, quantity, current_price,
            f"Multi-signal BUY [{signal_strength:.2f}] T{coin_state.dca_level} | {analysis}",
            tier=coin_state.dca_level
        )
    
    def _generate_multi_signal_sell(self, coin: str, current_price: float, 
                                  portfolio, coin_state: CoinState,
                                  signal_strength: float, analysis: str) -> TradeSignal:
        """Generate sell signal with multi-signal validation."""
        holding = portfolio.get_holding(coin)
        
        if holding <= 0:
            # Update baseline when we can't sell
            coin_state.update_baseline_weighted(current_price, self.baseline_weight)
            return TradeSignal(
                coin, TradeAction.HOLD, 0.0, current_price,
                f"Updated baseline to ${current_price:.2f} | {analysis}"
            )
        
        # Calculate signal-strength-adjusted sell percentage
        base_sell_percentage = self.base_trade_percentage
        dca_bonus = coin_state.dca_level * 0.05
        strength_multiplier = 0.7 + (signal_strength * 0.3)  # 0.7 to 1.0 range
        
        sell_percentage = min((base_sell_percentage + dca_bonus) * strength_multiplier, 0.8)  # Cap at 80%
        sell_quantity = holding * sell_percentage
        sell_value = sell_quantity * current_price
        
        # Ensure minimum trade size
        if sell_value < self.min_trade_amount:
            sell_quantity = holding  # Sell all if below minimum
        
        # Update coin state
        coin_state.last_sell_time = time.time()
        coin_state.update_baseline_weighted(current_price, self.baseline_weight)
        
        # Reset DCA level if profitable exit
        if current_price > coin_state.last_buy_price:
            coin_state.reset_dca_level()
        
        return TradeSignal(
            coin, TradeAction.SELL, sell_quantity, current_price,
            f"Multi-signal SELL [{signal_strength:.2f}] {sell_percentage*100:.0f}% | {analysis}"
        )
    
    def _should_timed_sell(self, coin_state: CoinState, current_price: float, current_time: float) -> bool:
        """Check if timed sell conditions are met."""
        if coin_state.last_buy_time == 0:
            return False
        
        hold_duration = current_time - coin_state.last_buy_time
        if hold_duration < self.max_hold_ticks:
            return False
        
        price_change = percent_change(coin_state.baseline_price, current_price)
        return price_change >= 0.015  # 1.5% threshold
    
    def _generate_timed_sell_signal(self, coin: str, current_price: float, 
                                  portfolio, coin_state: CoinState) -> TradeSignal:
        """Generate timed sell signal."""
        holding = portfolio.get_holding(coin)
        
        if holding <= 0:
            return TradeSignal(coin, TradeAction.HOLD, 0.0, current_price, "No holdings for timed sell")
        
        sell_quantity = holding * 0.05  # 5% timed sell
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
        
        if self.last_portfolio_value == 0:
            self.last_portfolio_value = current_portfolio_value
            return None
        
        profit_ratio = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        
        should_rebalance = (
            profit_ratio >= self.rebalance_profit_threshold or 
            self.trades_since_rebalance >= 10
        )
        
        if should_rebalance:
            self.last_portfolio_value = current_portfolio_value
            self.trades_since_rebalance = 0
            
            return TradeSignal(
                "PORTFOLIO", TradeAction.REBALANCE, self.rebalance_percentage, 0.0,
                f"Multi-signal rebalancing - profit: {profit_ratio*100:.1f}%"
            )
        
        return None
    
    def _get_trade_explanation(self, signal: TradeSignal, market_data: Optional[MarketData],
                             market_sentiment: MarketSentiment) -> str:
        """Generate a simple explanation for the trade signal."""
        try:
            # Get additional analysis data
            volume_analysis = {}
            technical_analysis = {}
            
            if market_data:
                volume_analysis = self.data_fetcher.get_volume_analysis(signal.coin)
                technical_analysis = self.data_fetcher.get_technical_analysis(signal.coin)
            
            return self.trade_explainer.explain_trade(
                signal, market_data, market_sentiment, volume_analysis, technical_analysis
            )
        except Exception as e:
            return f"Trade explanation unavailable: {str(e)}"
    
    def get_market_explanation(self, current_prices: Dict[str, float]) -> str:
        """Get an explanation of current market conditions."""
        try:
            _, market_sentiment = self.data_fetcher.fetch_all_data()
            return self.trade_explainer.explain_market_conditions(market_sentiment, current_prices)
        except Exception as e:
            return f"Market explanation unavailable: {str(e)}"
    
    def get_strategy_explanation(self) -> str:
        """Get an explanation of how the strategy works."""
        return self.trade_explainer.get_strategy_explanation()
    
    def on_trade_executed(self, trade_signal: TradeSignal) -> None:
        """Callback when a trade is executed."""
        if trade_signal.action in [TradeAction.BUY, TradeAction.SELL]:
            self.trades_since_rebalance += 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information."""
        return {
            'strategy_type': 'multi_signal_enhanced',
            'data_sources': [
                'CoinGecko_comprehensive',
                'Fear_Greed_Index', 
                'Technical_Indicators',
                'Volume_Analysis',
                'Social_Metrics',
                'Market_Sentiment'
            ],
            'technical_indicators': ['RSI', 'MACD', 'Bollinger_Bands', 'Moving_Averages'],
            'min_signal_strength': self.min_signal_strength,
            'adaptive_thresholds': True,
            'sentiment_adjustments': True,
            'volume_filtering': True,
            'multi_timeframe_analysis': True
        }
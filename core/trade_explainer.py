"""
Trade explanation system that converts complex trading signals into simple, human-readable explanations.
"""
from typing import Dict, List, Any, Optional
from core.enhanced_strategy import TradeAction, TradeSignal
from core.advanced_data_fetcher import MarketData, MarketSentiment


class TradeExplainer:
    """
    Converts complex trading decisions into simple, easy-to-understand explanations.
    """
    
    def __init__(self):
        # Simple explanations for technical indicators
        self.rsi_explanations = {
            'OVERSOLD': "The coin has been heavily sold and might bounce back up soon",
            'OVERBOUGHT': "The coin has been heavily bought and might drop soon",
            'NEUTRAL': "The buying and selling pressure is balanced"
        }
        
        self.macd_explanations = {
            'BULLISH': "The momentum is pointing upward - price trend is getting stronger",
            'BEARISH': "The momentum is pointing downward - price trend is weakening",
            'NEUTRAL': "The momentum is unclear"
        }
        
        self.bb_explanations = {
            'OVERSOLD': "The price is unusually low compared to recent average",
            'OVERBOUGHT': "The price is unusually high compared to recent average",
            'NEUTRAL': "The price is within normal range"
        }
        
        self.volume_explanations = {
            'HIGH': "Lots of people are trading this coin right now",
            'MEDIUM': "A decent amount of people are trading this coin",
            'LOW': "Very few people are trading this coin (risky)"
        }
        
        self.sentiment_explanations = {
            'extreme_fear': "Everyone is panicking and selling (often a good time to buy)",
            'fear': "People are worried and selling more than usual",
            'neutral': "People are neither too excited nor too scared",
            'greed': "People are getting excited and buying more",
            'extreme_greed': "Everyone is euphoric and buying (often risky)"
        }
    
    def explain_trade(self, signal: TradeSignal, market_data: Optional[MarketData] = None,
                     market_sentiment: Optional[MarketSentiment] = None,
                     volume_analysis: Dict = None, technical_analysis: Dict = None) -> str:
        """
        Generate a simple explanation for why a trade was made.
        
        Args:
            signal: The trade signal that was executed
            market_data: Market data for the coin
            market_sentiment: Overall market sentiment
            volume_analysis: Volume analysis results
            technical_analysis: Technical analysis results
            
        Returns:
            Simple explanation string
        """
        if signal.action == TradeAction.HOLD:
            return self._explain_hold_decision(signal, market_data, technical_analysis)
        elif signal.action == TradeAction.BUY:
            return self._explain_buy_decision(signal, market_data, market_sentiment, volume_analysis, technical_analysis)
        elif signal.action == TradeAction.SELL:
            return self._explain_sell_decision(signal, market_data, market_sentiment, volume_analysis, technical_analysis)
        elif signal.action == TradeAction.REBALANCE:
            return self._explain_rebalance_decision(signal)
        else:
            return "Trade decision made based on strategy rules."
    
    def _explain_hold_decision(self, signal: TradeSignal, market_data: Optional[MarketData],
                             technical_analysis: Dict) -> str:
        """Explain why we're holding (not trading)."""
        explanations = []
        
        # Check if it's a signal strength issue
        if "Signals:" in signal.reason:
            parts = signal.reason.split("|")
            if len(parts) > 0 and "Signals:" in parts[0]:
                strength_part = parts[0].strip()
                explanations.append(f"🤔 **Why we're waiting**: Not enough trading signals are lining up yet ({strength_part})")
        
        # Check for insufficient balance
        if "Insufficient balance" in signal.reason:
            explanations.append("💰 **Can't trade**: Not enough money available for this trade size")
        
        # Check for max DCA level
        if "Max DCA level" in signal.reason:
            explanations.append("🛑 **Position limit reached**: Already bought this coin 3 times, waiting for better opportunity")
        
        # Check for baseline update
        if "Updated baseline" in signal.reason:
            explanations.append("📈 **Price jumped up**: No coins to sell, so updating our reference price for future trades")
        
        # Add technical context if available
        if technical_analysis and technical_analysis.get('signals'):
            tech_context = self._get_technical_context(technical_analysis['signals'])
            if tech_context:
                explanations.append(f"📊 **Market signals**: {tech_context}")
        
        # Default explanation
        if not explanations:
            explanations.append("⏳ **Waiting for better opportunity**: Current market conditions don't meet our trading criteria")
        
        return "\n".join(explanations)
    
    def _explain_buy_decision(self, signal: TradeSignal, market_data: Optional[MarketData],
                            market_sentiment: Optional[MarketSentiment], volume_analysis: Dict,
                            technical_analysis: Dict) -> str:
        """Explain why we bought."""
        explanations = []
        
        # Main buy reason
        if signal.tier > 0:
            explanations.append(f"🟢 **BUYING {signal.coin}** (Purchase #{signal.tier})")
        else:
            explanations.append(f"🟢 **BUYING {signal.coin}**")
        
        explanations.append(f"💵 **Amount**: ${signal.quantity * signal.price:.2f} ({signal.quantity:.6f} {signal.coin})")
        
        # Price drop explanation
        if "Price dropped" in signal.reason or "Price drop" in signal.reason:
            explanations.append("📉 **Why**: The price dropped significantly - good buying opportunity!")
        
        # Multi-signal explanation
        if "Multi-signal BUY" in signal.reason:
            strength = self._extract_signal_strength(signal.reason)
            explanations.append(f"🎯 **Confidence Level**: {strength:.1%} - Multiple indicators agree this is a good buy")
        
        # Technical reasons
        tech_reasons = []
        if technical_analysis and technical_analysis.get('signals'):
            signals = technical_analysis['signals']
            
            if signals.get('rsi_signal') == 'OVERSOLD':
                tech_reasons.append("RSI shows it's oversold (heavily sold recently)")
            if signals.get('bb_signal') == 'OVERSOLD':
                tech_reasons.append("Price is unusually low compared to recent average")
            if signals.get('macd_signal') == 'BULLISH':
                tech_reasons.append("Momentum is turning positive")
        
        if tech_reasons:
            explanations.append("📊 **Technical indicators**: " + " | ".join(tech_reasons))
        
        # Volume validation
        if volume_analysis:
            volume_strength = volume_analysis.get('volume_strength', 'LOW')
            if volume_strength in ['HIGH', 'MEDIUM']:
                explanations.append(f"💪 **Good liquidity**: {volume_strength.lower()} trading volume means easier to buy/sell")
        
        # Market sentiment
        if market_sentiment and market_sentiment.fear_greed_classification:
            sentiment = market_sentiment.fear_greed_classification.lower()
            if 'fear' in sentiment:
                explanations.append(f"😱 **Market sentiment**: {sentiment.title()} - {self.sentiment_explanations.get(sentiment.replace(' ', '_'), '')}")
        
        # DCA explanation
        if signal.tier > 1:
            explanations.append(f"📈 **Strategy**: This is purchase #{signal.tier} - averaging down to get a better overall price")
        
        return "\n".join(explanations)
    
    def _explain_sell_decision(self, signal: TradeSignal, market_data: Optional[MarketData],
                             market_sentiment: Optional[MarketSentiment], volume_analysis: Dict,
                             technical_analysis: Dict) -> str:
        """Explain why we sold."""
        explanations = []
        
        # Main sell reason
        explanations.append(f"🔴 **SELLING {signal.coin}**")
        explanations.append(f"💰 **Amount**: ${signal.quantity * signal.price:.2f} ({signal.quantity:.6f} {signal.coin})")
        
        # Price increase explanation
        if "Price up" in signal.reason or "Price increased" in signal.reason:
            explanations.append("📈 **Why**: The price went up significantly - taking profits!")
        
        # Multi-signal explanation
        if "Multi-signal SELL" in signal.reason:
            strength = self._extract_signal_strength(signal.reason)
            explanations.append(f"🎯 **Confidence Level**: {strength:.1%} - Multiple indicators suggest it's time to sell")
        
        # Timed sell explanation
        if "Timed sell" in signal.reason:
            explanations.append("⏰ **Why**: Held this coin for a while and it's up a bit - cutting position to avoid losses")
        
        # Technical reasons
        tech_reasons = []
        if technical_analysis and technical_analysis.get('signals'):
            signals = technical_analysis['signals']
            
            if signals.get('rsi_signal') == 'OVERBOUGHT':
                tech_reasons.append("RSI shows it's overbought (heavily bought recently)")
            if signals.get('bb_signal') == 'OVERBOUGHT':
                tech_reasons.append("Price is unusually high compared to recent average")
            if signals.get('macd_signal') == 'BEARISH':
                tech_reasons.append("Momentum is turning negative")
        
        if tech_reasons:
            explanations.append("📊 **Technical indicators**: " + " | ".join(tech_reasons))
        
        # Volume validation
        if volume_analysis:
            volume_strength = volume_analysis.get('volume_strength', 'LOW')
            if volume_strength in ['HIGH', 'MEDIUM']:
                explanations.append(f"💪 **Good liquidity**: {volume_strength.lower()} trading volume means easy to sell")
        
        # Market sentiment
        if market_sentiment and market_sentiment.fear_greed_classification:
            sentiment = market_sentiment.fear_greed_classification.lower()
            if 'greed' in sentiment:
                explanations.append(f"🤑 **Market sentiment**: {sentiment.title()} - {self.sentiment_explanations.get(sentiment.replace(' ', '_'), '')}")
        
        # Rebalancing explanation
        if "REBALANCE" in signal.action.value:
            explanations.append("⚖️ **Strategy**: Taking some profits to maintain balanced portfolio")
        
        return "\n".join(explanations)
    
    def _explain_rebalance_decision(self, signal: TradeSignal) -> str:
        """Explain portfolio rebalancing."""
        explanations = []
        explanations.append("⚖️ **REBALANCING PORTFOLIO**")
        explanations.append("💰 **Why**: Portfolio has grown in value - taking some profits to lock in gains")
        explanations.append(f"📊 **Action**: Selling {signal.quantity*100:.0f}% of all holdings to convert back to cash")
        explanations.append("🎯 **Strategy**: This keeps the portfolio balanced and secures profits")
        
        return "\n".join(explanations)
    
    def _get_technical_context(self, signals: Dict) -> str:
        """Get a simple explanation of technical signals."""
        context_parts = []
        
        rsi_signal = signals.get('rsi_signal', 'NEUTRAL')
        if rsi_signal != 'NEUTRAL':
            context_parts.append(self.rsi_explanations.get(rsi_signal, ''))
        
        macd_signal = signals.get('macd_signal', 'NEUTRAL')
        if macd_signal != 'NEUTRAL':
            context_parts.append(self.macd_explanations.get(macd_signal, ''))
        
        bb_signal = signals.get('bb_signal', 'NEUTRAL')
        if bb_signal != 'NEUTRAL':
            context_parts.append(self.bb_explanations.get(bb_signal, ''))
        
        return " | ".join(context_parts) if context_parts else "Indicators are mixed"
    
    def _extract_signal_strength(self, reason: str) -> float:
        """Extract signal strength from reason string."""
        try:
            # Look for pattern like [0.75] in the reason
            import re
            match = re.search(r'\[(\d+\.?\d*)\]', reason)
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.6  # Default
    
    def explain_market_conditions(self, market_sentiment: Optional[MarketSentiment],
                                current_prices: Dict[str, float]) -> str:
        """Provide a simple explanation of current market conditions."""
        explanations = []
        
        explanations.append("🌍 **CURRENT MARKET CONDITIONS**")
        
        # Fear & Greed Index
        if market_sentiment and market_sentiment.fear_greed_index is not None:
            fgi = market_sentiment.fear_greed_index
            classification = market_sentiment.fear_greed_classification or "neutral"
            
            explanations.append(f"😱 **Market Mood**: {classification.title()} ({fgi}/100)")
            explanations.append(f"   📝 **What this means**: {self.sentiment_explanations.get(classification.lower().replace(' ', '_'), 'Market sentiment is unclear')}")
        
        # Bitcoin Dominance
        if market_sentiment and market_sentiment.btc_dominance:
            btc_dom = market_sentiment.btc_dominance
            if btc_dom > 60:
                dom_explanation = "Bitcoin is leading the market - altcoins might struggle"
            elif btc_dom < 40:
                dom_explanation = "Altcoin season - smaller coins might outperform Bitcoin"
            else:
                dom_explanation = "Balanced market between Bitcoin and altcoins"
            
            explanations.append(f"₿ **Bitcoin Dominance**: {btc_dom:.1f}% - {dom_explanation}")
        
        # Current prices context
        explanations.append("💰 **Current Prices**:")
        for coin, price in current_prices.items():
            explanations.append(f"   {coin}: ${price:,.2f}")
        
        return "\n".join(explanations)
    
    def get_strategy_explanation(self) -> str:
        """Explain how the trading strategy works in simple terms."""
        return """
🤖 **HOW THE TRADING BOT WORKS**

📊 **Data Collection**: 
   • Gets live prices from CoinGecko every 30 seconds
   • Checks market sentiment (Fear & Greed Index)
   • Calculates technical indicators (RSI, MACD, etc.)
   • Analyzes trading volume and liquidity

🎯 **Decision Making**:
   • Combines multiple signals to get confidence score (0-100%)
   • Only trades when confidence is above 60%
   • Adjusts based on market sentiment (more careful when everyone is greedy)
   • Requires good trading volume before making moves

💰 **Trading Strategy**:
   • BUY when price drops significantly AND multiple indicators agree
   • SELL when price rises significantly AND indicators suggest it's time
   • Use dollar-cost averaging (buy more if price keeps dropping)
   • Take profits automatically when portfolio grows
   • Never risk more than 10-20% of balance on single trade

🛡️ **Risk Management**:
   • Wait for multiple confirmations before trading
   • Only trade coins with good liquidity
   • Automatically rebalance to lock in profits
   • Use time-based stops to avoid holding bad positions too long
        """.strip()
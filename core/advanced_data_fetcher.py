"""
Advanced multi-source data fetcher for institutional-grade trading.
Integrates price, volume, technical indicators, sentiment, and on-chain data.
"""
import requests
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import ta  # Technical Analysis library
from utils.math_tools import round_to_precision


@dataclass
class MarketData:
    """Comprehensive market data for a coin."""
    symbol: str
    price: float
    volume_24h: float
    volume_change_24h: float
    market_cap: float
    price_change_24h: float
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Liquidity metrics
    bid_ask_spread: Optional[float] = None
    market_depth_2pct: Optional[float] = None
    
    # Sentiment
    social_score: Optional[float] = None
    developer_score: Optional[float] = None
    community_score: Optional[float] = None


@dataclass
class MarketSentiment:
    """Overall market sentiment indicators."""
    fear_greed_index: Optional[int] = None
    fear_greed_classification: Optional[str] = None
    btc_dominance: Optional[float] = None
    total_market_cap: Optional[float] = None
    market_cap_change_24h: Optional[float] = None
    
    # Cross-market correlations
    sp500_correlation: Optional[float] = None
    gold_correlation: Optional[float] = None


class AdvancedDataFetcher:
    """
    Advanced data fetcher that aggregates multiple data sources for comprehensive market analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced data fetcher."""
        self.config = config
        self.coin_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'SOL': 'solana',
            'XRP': 'ripple'
        }
        self.coins = list(config['coins'].keys())
        
        # Data storage
        self.current_market_data: Dict[str, MarketData] = {}
        self.market_sentiment: MarketSentiment = MarketSentiment()
        self.price_history: Dict[str, deque] = {}
        
        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.api_key = config.get('api', {}).get('api_key', '')
        
        # Check for Alpaca data source
        alpaca_config = config.get('alpaca', {})
        self.use_alpaca_data = (alpaca_config.get('use_paper_trading', False) and 
                               alpaca_config.get('api_key') and 
                               alpaca_config.get('api_key') != "YOUR_ALPACA_API_KEY")
        
        if self.use_alpaca_data:
            try:
                from core.alpaca_price_fetcher import AlpacaPriceFetcher
                self.alpaca_fetcher = AlpacaPriceFetcher(config)
                print("📊 Using Alpaca for advanced market data")
            except Exception as e:
                print(f"⚠️  Alpaca data fetcher failed: {e}")
                self.use_alpaca_data = False
        
        # Rate limiting
        self.last_fetch_time = 0
        self.min_fetch_interval = 15  # 15 seconds between full data updates
        
        # Initialize price history
        for coin in self.coins:
            self.price_history[coin] = deque(maxlen=200)  # Store last 200 prices for TA
    
    def fetch_all_data(self) -> Tuple[Dict[str, MarketData], MarketSentiment]:
        """
        Fetch comprehensive market data from all sources.
        
        Returns:
            Tuple of (market_data_dict, market_sentiment)
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_fetch_time < self.min_fetch_interval:
            return self.current_market_data, self.market_sentiment
        
        try:
            # Choose data source based on configuration
            if self.use_alpaca_data:
                self._fetch_alpaca_data()
            else:
                self._fetch_coingecko_data()
            
            # Always fetch Fear & Greed from alternative.me (independent of price source)
            self._fetch_fear_greed_index()
            self._calculate_technical_indicators()
            self._calculate_market_correlations()
            
            self.last_fetch_time = current_time
            source = "Alpaca" if self.use_alpaca_data else "CoinGecko"
            print(f"📊 Advanced market data updated successfully (via {source})")
            
        except Exception as e:
            print(f"⚠️ Error fetching advanced market data: {e}")
        
        return self.current_market_data, self.market_sentiment
    
    def _fetch_coingecko_data(self) -> None:
        """Fetch comprehensive data from CoinGecko API."""
        try:
            # Get coin IDs
            coin_ids = [self.coin_mapping[coin] for coin in self.coins if coin in self.coin_mapping]
            coin_ids_str = ','.join(coin_ids)
            
            # Fetch detailed market data
            url = f"{self.coingecko_base}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': coin_ids_str,
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            headers = {}
            if self.api_key:
                headers['x-cg-demo-api-key'] = self.api_key
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process coin data
            for coin_data in data:
                coin_id = coin_data['id']
                symbol = None
                
                # Find matching symbol
                for sym, id_val in self.coin_mapping.items():
                    if id_val == coin_id:
                        symbol = sym
                        break
                
                if not symbol:
                    continue
                
                # Create MarketData object
                market_data = MarketData(
                    symbol=symbol,
                    price=coin_data.get('current_price', 0.0),
                    volume_24h=coin_data.get('total_volume', 0.0),
                    volume_change_24h=0.0,  # Not available in this endpoint
                    market_cap=coin_data.get('market_cap', 0.0),
                    price_change_24h=coin_data.get('price_change_percentage_24h', 0.0) / 100
                )
                
                # Update price history
                if market_data.price > 0:
                    self.price_history[symbol].append(market_data.price)
                
                self.current_market_data[symbol] = market_data
            
            # Fetch additional detailed data for volume and social metrics
            self._fetch_detailed_coin_data()
            
        except requests.RequestException as e:
            print(f"Error fetching CoinGecko data: {e}")
        except Exception as e:
            print(f"Unexpected error in CoinGecko fetch: {e}")
    
    def _fetch_detailed_coin_data(self) -> None:
        """Fetch detailed data including social metrics and developer activity."""
        try:
            for symbol in self.coins:
                if symbol not in self.coin_mapping:
                    continue
                
                coin_id = self.coin_mapping[symbol]
                url = f"{self.coingecko_base}/coins/{coin_id}"
                params = {
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'true',
                    'community_data': 'true',
                    'developer_data': 'true'
                }
                
                headers = {}
                if self.api_key:
                    headers['x-cg-demo-api-key'] = self.api_key
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if symbol in self.current_market_data:
                    # Update with social and developer metrics
                    market_data = self.current_market_data[symbol]
                    
                    # Social metrics
                    community_data = data.get('community_data', {})
                    market_data.social_score = self._calculate_social_score(community_data)
                    
                    # Developer metrics
                    developer_data = data.get('developer_data', {})
                    market_data.developer_score = self._calculate_developer_score(developer_data)
                    
                    # Market depth (approximate from volume)
                    market_data_section = data.get('market_data', {})
                    if market_data_section.get('total_volume'):
                        btc_volume = market_data_section['total_volume'].get('btc', 0)
                        market_data.market_depth_2pct = btc_volume * 0.1  # Rough approximation
                
                # Add small delay to respect rate limits
                time.sleep(0.2)
                
        except Exception as e:
            print(f"Error fetching detailed coin data: {e}")
    
    def _fetch_alpaca_data(self) -> None:
        """Fetch comprehensive data from Alpaca API."""
        try:
            # Get market data from Alpaca fetcher
            alpaca_market_data = self.alpaca_fetcher.get_market_data()
            
            for symbol, data in alpaca_market_data.items():
                # Create MarketData object from Alpaca data
                market_data = MarketData(
                    symbol=symbol,
                    price=data['price'],
                    volume_24h=data['volume_24h'],
                    volume_change_24h=0.0,  # Not available from Alpaca directly
                    market_cap=data['market_cap'],
                    price_change_24h=data['price_change_24h'],
                    
                    # Technical indicators from Alpaca
                    rsi=data['technical_indicators']['rsi'],
                    macd=data['technical_indicators']['macd'],
                    macd_signal=0.0,  # Could be calculated
                    sma_20=data['technical_indicators']['sma_20'],
                    sma_50=data['technical_indicators'].get('sma_50', data['technical_indicators']['sma_20']),
                    
                    # Liquidity metrics (simplified)
                    bid_ask_spread=0.001,  # Default small spread
                    market_depth_2pct=data['technical_indicators']['volume_avg'] * 0.1,
                    
                    # Social metrics from Alpaca data
                    social_score=data['social_metrics']['social_score'] / 100,
                    developer_score=data['social_metrics']['developer_score'] / 100,
                    community_score=data['social_metrics']['community_score'] / 100
                )
                
                self.current_market_data[symbol] = market_data
                
                # Update price history for technical analysis
                price_entry = {
                    'timestamp': time.time(),
                    'price': data['price'],
                    'volume': data['volume_24h']
                }
                self.price_history[symbol].append(price_entry)
                
        except Exception as e:
            print(f"Error fetching Alpaca data: {e}")
            # Fallback to CoinGecko if Alpaca fails
            self._fetch_coingecko_data()
    
    def _calculate_social_score(self, community_data: Dict) -> float:
        """Calculate normalized social score from community data."""
        try:
            twitter_followers = community_data.get('twitter_followers', 0)
            reddit_subscribers = community_data.get('reddit_subscribers', 0)
            telegram_users = community_data.get('telegram_channel_user_count', 0)
            
            # Normalize and weight different social metrics
            twitter_score = min(twitter_followers / 1000000, 1.0) * 0.4  # Cap at 1M followers
            reddit_score = min(reddit_subscribers / 500000, 1.0) * 0.3   # Cap at 500K subscribers
            telegram_score = min(telegram_users / 100000, 1.0) * 0.3     # Cap at 100K users
            
            return round_to_precision(twitter_score + reddit_score + telegram_score, 3)
        except:
            return 0.0
    
    def _calculate_developer_score(self, developer_data: Dict) -> float:
        """Calculate normalized developer activity score."""
        try:
            commits_4w = developer_data.get('commit_count_4_weeks', 0)
            forks = developer_data.get('forks', 0)
            stars = developer_data.get('stars', 0)
            
            # Normalize developer metrics
            commit_score = min(commits_4w / 100, 1.0) * 0.5    # Cap at 100 commits/4w
            fork_score = min(forks / 10000, 1.0) * 0.25       # Cap at 10K forks
            star_score = min(stars / 50000, 1.0) * 0.25       # Cap at 50K stars
            
            return round_to_precision(commit_score + fork_score + star_score, 3)
        except:
            return 0.0
    
    def _fetch_fear_greed_index(self) -> None:
        """Fetch Fear & Greed Index from Alternative.me."""
        try:
            response = requests.get(self.fear_greed_api, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get('data') and len(data['data']) > 0:
                latest = data['data'][0]
                self.market_sentiment.fear_greed_index = int(latest['value'])
                self.market_sentiment.fear_greed_classification = latest['value_classification']
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
    
    def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators for each coin."""
        for symbol in self.coins:
            if symbol not in self.current_market_data:
                continue
            
            price_history = list(self.price_history[symbol])
            if len(price_history) < 50:  # Need sufficient data for indicators
                continue
            
            try:
                # Convert to pandas Series for TA calculations
                prices = pd.Series(price_history)
                
                market_data = self.current_market_data[symbol]
                
                # RSI (14 period)
                if len(prices) >= 14:
                    market_data.rsi = round_to_precision(ta.momentum.RSIIndicator(prices, window=14).rsi().iloc[-1], 2)
                
                # MACD
                if len(prices) >= 26:
                    macd = ta.trend.MACD(prices)
                    market_data.macd = round_to_precision(macd.macd().iloc[-1], 4)
                    market_data.macd_signal = round_to_precision(macd.macd_signal().iloc[-1], 4)
                
                # Bollinger Bands
                if len(prices) >= 20:
                    bb = ta.volatility.BollingerBands(prices, window=20)
                    market_data.bb_upper = round_to_precision(bb.bollinger_hband().iloc[-1], 2)
                    market_data.bb_lower = round_to_precision(bb.bollinger_lband().iloc[-1], 2)
                
                # Moving Averages
                if len(prices) >= 20:
                    market_data.sma_20 = round_to_precision(prices.rolling(20).mean().iloc[-1], 2)
                if len(prices) >= 50:
                    market_data.sma_50 = round_to_precision(prices.rolling(50).mean().iloc[-1], 2)
                if len(prices) >= 12:
                    market_data.ema_12 = round_to_precision(prices.ewm(span=12).mean().iloc[-1], 2)
                if len(prices) >= 26:
                    market_data.ema_26 = round_to_precision(prices.ewm(span=26).mean().iloc[-1], 2)
                
            except Exception as e:
                print(f"Error calculating technical indicators for {symbol}: {e}")
    
    def _calculate_market_correlations(self) -> None:
        """Calculate market correlations and dominance metrics."""
        try:
            # Fetch global market data
            url = f"{self.coingecko_base}/global"
            headers = {}
            if self.api_key:
                headers['x-cg-demo-api-key'] = self.api_key
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            global_data = data.get('data', {})
            
            # Bitcoin dominance
            self.market_sentiment.btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0.0)
            
            # Total market cap
            total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0.0)
            self.market_sentiment.total_market_cap = total_market_cap
            
            # Market cap change
            market_cap_change = global_data.get('market_cap_change_percentage_24h_usd', 0.0)
            self.market_sentiment.market_cap_change_24h = market_cap_change / 100
            
        except Exception as e:
            print(f"Error fetching global market data: {e}")
    
    def get_volume_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyze volume patterns for a specific coin."""
        if symbol not in self.current_market_data:
            return {}
        
        market_data = self.current_market_data[symbol]
        
        # Calculate volume metrics
        volume_24h = market_data.volume_24h
        market_cap = market_data.market_cap
        
        volume_to_mcap_ratio = (volume_24h / market_cap) if market_cap > 0 else 0
        
        # Volume classification
        volume_strength = "LOW"
        if volume_to_mcap_ratio > 0.1:
            volume_strength = "HIGH"
        elif volume_to_mcap_ratio > 0.05:
            volume_strength = "MEDIUM"
        
        return {
            'volume_24h': volume_24h,
            'volume_to_mcap_ratio': round_to_precision(volume_to_mcap_ratio, 4),
            'volume_strength': volume_strength,
            'sufficient_liquidity': volume_24h > 1000000  # $1M+ daily volume
        }
    
    def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive technical analysis for a coin."""
        if symbol not in self.current_market_data:
            return {}
        
        market_data = self.current_market_data[symbol]
        
        # Technical signal analysis
        signals = {
            'rsi_signal': 'NEUTRAL',
            'macd_signal': 'NEUTRAL',
            'bb_signal': 'NEUTRAL',
            'ma_signal': 'NEUTRAL',
            'overall_signal': 'NEUTRAL'
        }
        
        # RSI analysis
        if market_data.rsi is not None:
            if market_data.rsi > 70:
                signals['rsi_signal'] = 'OVERBOUGHT'
            elif market_data.rsi < 30:
                signals['rsi_signal'] = 'OVERSOLD'
        
        # MACD analysis
        if market_data.macd is not None and market_data.macd_signal is not None:
            if market_data.macd > market_data.macd_signal:
                signals['macd_signal'] = 'BULLISH'
            else:
                signals['macd_signal'] = 'BEARISH'
        
        # Bollinger Bands analysis
        current_price = market_data.price
        if market_data.bb_upper and market_data.bb_lower:
            if current_price > market_data.bb_upper:
                signals['bb_signal'] = 'OVERBOUGHT'
            elif current_price < market_data.bb_lower:
                signals['bb_signal'] = 'OVERSOLD'
        
        # Moving Average analysis
        if market_data.sma_20 and market_data.sma_50:
            if market_data.sma_20 > market_data.sma_50:
                signals['ma_signal'] = 'BULLISH'
            else:
                signals['ma_signal'] = 'BEARISH'
        
        # Overall signal (simplified)
        bullish_signals = sum(1 for signal in signals.values() if signal in ['BULLISH', 'OVERSOLD'])
        bearish_signals = sum(1 for signal in signals.values() if signal in ['BEARISH', 'OVERBOUGHT'])
        
        if bullish_signals > bearish_signals:
            signals['overall_signal'] = 'BULLISH'
        elif bearish_signals > bullish_signals:
            signals['overall_signal'] = 'BEARISH'
        
        return {
            'technical_data': {
                'rsi': market_data.rsi,
                'macd': market_data.macd,
                'macd_signal': market_data.macd_signal,
                'bb_upper': market_data.bb_upper,
                'bb_lower': market_data.bb_lower,
                'sma_20': market_data.sma_20,
                'sma_50': market_data.sma_50,
                'current_price': current_price
            },
            'signals': signals
        }
    
    def get_sentiment_analysis(self) -> Dict[str, Any]:
        """Get comprehensive market sentiment analysis."""
        sentiment = self.market_sentiment
        
        # Sentiment scoring
        sentiment_score = 0.5  # Neutral baseline
        
        if sentiment.fear_greed_index is not None:
            # Convert fear/greed to sentiment score (0-1 scale)
            sentiment_score = sentiment.fear_greed_index / 100
        
        # BTC dominance analysis
        btc_dom_signal = 'NEUTRAL'
        if sentiment.btc_dominance:
            if sentiment.btc_dominance > 60:
                btc_dom_signal = 'BTC_DOMINANCE_HIGH'
            elif sentiment.btc_dominance < 40:
                btc_dom_signal = 'ALTCOIN_SEASON'
        
        return {
            'fear_greed_index': sentiment.fear_greed_index,
            'fear_greed_classification': sentiment.fear_greed_classification,
            'btc_dominance': sentiment.btc_dominance,
            'btc_dominance_signal': btc_dom_signal,
            'market_cap_change_24h': sentiment.market_cap_change_24h,
            'sentiment_score': round_to_precision(sentiment_score, 3),
            'total_market_cap': sentiment.total_market_cap
        }
    
    def reset_data(self) -> None:
        """Reset all stored data."""
        self.current_market_data.clear()
        self.market_sentiment = MarketSentiment()
        for coin in self.coins:
            self.price_history[coin].clear()
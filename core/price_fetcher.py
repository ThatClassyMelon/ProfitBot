"""
Real-time cryptocurrency price fetcher using CoinGecko API.
"""
import requests
import time
from typing import Dict, Any, Optional
from utils.math_tools import round_to_precision


class CoinGeckoPriceFetcher:
    """Fetches real-time cryptocurrency prices from CoinGecko API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the price fetcher.
        
        Args:
            config: Configuration dictionary
        """
        self.api_base_url = "https://api.coingecko.com/api/v3"
        self.api_key = config.get('api', {}).get('api_key', '')
        self.coin_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'SOL': 'solana',
            'XRP': 'ripple'
        }
        self.coins = list(config['coins'].keys())
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, list] = {}
        self.last_fetch_time = 0
        self.min_fetch_interval = config.get('api', {}).get('price_update_interval', 10)  # Use config interval
        
        # Initialize price history
        for coin in self.coins:
            self.price_history[coin] = []
    
    def fetch_prices(self) -> Dict[str, float]:
        """
        Fetch current prices from CoinGecko API.
        
        Returns:
            Dictionary of current prices
        """
        current_time = time.time()
        
        # Rate limiting: don't fetch more than once every 10 seconds
        if current_time - self.last_fetch_time < self.min_fetch_interval:
            return self.current_prices.copy()
        
        try:
            # Build API request for all coins
            coin_ids = [self.coin_mapping[coin] for coin in self.coins if coin in self.coin_mapping]
            coin_ids_str = ','.join(coin_ids)
            
            url = f"{self.api_base_url}/simple/price"
            params = {
                'ids': coin_ids_str,
                'vs_currencies': 'usd'
            }
            
            headers = {}
            if self.api_key:
                headers['x-cg-demo-api-key'] = self.api_key
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Update prices
            for coin in self.coins:
                if coin in self.coin_mapping:
                    coin_id = self.coin_mapping[coin]
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = data[coin_id]['usd']
                        self.current_prices[coin] = round_to_precision(price, 8)
                        self.price_history[coin].append(self.current_prices[coin])
                        
                        # Keep history manageable (last 1000 prices)
                        if len(self.price_history[coin]) > 1000:
                            self.price_history[coin] = self.price_history[coin][-1000:]
            
            self.last_fetch_time = current_time
            return self.current_prices.copy()
            
        except requests.RequestException as e:
            print(f"Error fetching prices from CoinGecko: {e}")
            return self.current_prices.copy()
        except Exception as e:
            print(f"Unexpected error fetching prices: {e}")
            return self.current_prices.copy()
    
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
    
    def initialize_prices(self) -> bool:
        """
        Initialize prices by fetching them once.
        
        Returns:
            True if successful, False otherwise
        """
        prices = self.fetch_prices()
        return len(prices) > 0 and all(price > 0 for price in prices.values())


class RealTimePriceSimulator:
    """
    Wrapper that combines real price fetching with fallback simulation.
    This maintains the same interface as the original PriceSimulator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real-time price simulator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.use_real_prices = config.get('api', {}).get('use_real_prices', True)
        
        if self.use_real_prices:
            # Check if we should use Alpaca for market data
            alpaca_config = config.get('alpaca', {})
            use_alpaca_data = (alpaca_config.get('use_paper_trading', False) and 
                             alpaca_config.get('api_key') and 
                             alpaca_config.get('api_key') != "YOUR_ALPACA_API_KEY")
            
            if use_alpaca_data:
                try:
                    from core.alpaca_price_fetcher import AlpacaPriceFetcher
                    self.price_fetcher = AlpacaPriceFetcher(config)
                    print("📈 Alpaca market data mode enabled")
                except Exception as e:
                    print(f"⚠️  Alpaca data failed, using CoinGecko: {e}")
                    self.price_fetcher = CoinGeckoPriceFetcher(config)
                    print("🌐 CoinGecko price mode enabled (Alpaca fallback)")
            else:
                self.price_fetcher = CoinGeckoPriceFetcher(config)
                print("🌐 CoinGecko price mode enabled")
            
            # Initialize simulation as fallback but don't change use_real_prices flag
            from core.simulator import PriceSimulator
            self.simulator = PriceSimulator(self.config)
        else:
            self._init_simulation_mode()
    
    def _init_simulation_mode(self):
        """Initialize simulation mode as fallback."""
        from core.simulator import PriceSimulator
        self.simulator = PriceSimulator(self.config)
        self.use_real_prices = False
    
    def update_prices(self) -> Dict[str, float]:
        """
        Update prices using either real API or simulation.
        
        Returns:
            Dictionary of updated prices
        """
        if self.use_real_prices:
            # Try to get real prices first
            real_prices = self.price_fetcher.fetch_prices()
            if real_prices and any(price > 0 for price in real_prices.values()):
                return real_prices
            else:
                # If real prices fail, use simulation as fallback but keep real price mode
                return self.simulator.update_prices()
        else:
            return self.simulator.update_prices()
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices."""
        if self.use_real_prices:
            return self.price_fetcher.get_current_prices()
        else:
            return self.simulator.get_current_prices()
    
    def get_price(self, coin: str) -> float:
        """Get price for specific coin."""
        if self.use_real_prices:
            return self.price_fetcher.get_price(coin)
        else:
            return self.simulator.get_price(coin)
    
    def get_price_history(self, coin: str, limit: int = 100) -> list:
        """Get price history for specific coin."""
        if self.use_real_prices:
            return self.price_fetcher.get_price_history(coin, limit)
        else:
            return self.simulator.get_price_history(coin, limit)
    
    def reset_prices(self) -> None:
        """Reset prices."""
        if not self.use_real_prices:
            self.simulator.reset_prices()
    
    def is_using_real_prices(self) -> bool:
        """Check if using real prices or simulation."""
        return self.use_real_prices
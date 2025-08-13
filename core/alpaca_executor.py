"""
Alpaca Markets paper trading integration for ProfitBot.
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from core.enhanced_strategy import TradeSignal, TradeAction
from core.enhanced_executor import EnhancedTrade, EnhancedTradeExecutor


class AlpacaExecutor(EnhancedTradeExecutor):
    """Trade executor using Alpaca Markets paper trading API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Alpaca executor with paper trading."""
        super().__init__(config)
        
        # Get Alpaca credentials from config
        alpaca_config = config.get('alpaca', {})
        self.api_key = alpaca_config.get('api_key')
        self.secret_key = alpaca_config.get('secret_key')
        self.base_url = alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found in config")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
            api_version='v2'
        )
        
        # Crypto symbol mapping (Alpaca uses different symbols)
        self.symbol_mapping = {
            'BTC': 'BTC/USD',
            'ETH': 'ETH/USD', 
            'SOL': 'SOL/USD',
            'XRP': 'XRP/USD',
            'ADA': 'ADA/USD',
            'DOT': 'DOT/USD',
            'MATIC': 'MATIC/USD',
            'AVAX': 'AVAX/USD',
            'MLG': 'MLG/USD'  # Custom coin - will fall back to simulation if not available
        }
        
        # Minimum order amounts for crypto (in USD)
        self.min_order_amounts = {
            'BTC/USD': 1.0,
            'ETH/USD': 1.0,
            'SOL/USD': 1.0,
            'XRP/USD': 1.0,
            'ADA/USD': 1.0,
            'DOT/USD': 1.0,
            'MATIC/USD': 1.0,
            'AVAX/USD': 1.0,
            'MLG/USD': 0.01  # Lower minimum for smaller coins
        }
        
        # Verify connection and store account info
        self.account_info = self._verify_connection()
    
    def _verify_connection(self) -> Dict[str, Any]:
        """Verify connection to Alpaca API and return account info."""
        try:
            account = self.api.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   Account Status: {account.status}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'status': account.status
            }
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")
    
    def get_account_balance(self) -> float:
        """Get current account balance (buying power) from Alpaca."""
        try:
            account = self.api.get_account()
            return float(account.buying_power)
        except Exception as e:
            print(f"Error getting account balance: {e}")
            return self.account_info.get('buying_power', 1000.0)
    
    def get_alpaca_symbol(self, coin: str) -> str:
        """Convert ProfitBot coin symbol to Alpaca symbol."""
        return self.symbol_mapping.get(coin.upper(), f"{coin.upper()}USD")
    
    def get_current_price(self, coin: str) -> float:
        """Get current price for a coin from Alpaca."""
        try:
            symbol = self.get_alpaca_symbol(coin)
            
            # Get latest crypto bar
            bars = self.api.get_crypto_bars(
                symbol,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            
            if bars and len(bars) > 0:
                latest_bar = bars[-1]
                return float(latest_bar.c)  # Close price
            else:
                # Fallback to latest quote
                quote = self.api.get_latest_crypto_quote(symbol)
                if quote:
                    return float(quote.bp)  # Bid price
                    
        except Exception as e:
            print(f"Error getting price for {coin}: {e}")
            
        return 0.0
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            # Convert positions to our format
            holdings = {}
            for position in positions:
                if position.symbol.endswith('USD') and position.symbol in self.symbol_mapping.values():
                    # Find the coin symbol
                    coin = None
                    for k, v in self.symbol_mapping.items():
                        if v == position.symbol:
                            coin = k
                            break
                    
                    if coin:
                        holdings[coin] = {
                            'quantity': float(position.qty),
                            'market_value': float(position.market_value),
                            'avg_entry_price': float(position.avg_entry_price),
                            'unrealized_pnl': float(position.unrealized_pnl),
                            'unrealized_pnl_pct': float(position.unrealized_plpc) * 100
                        }
            
            return {
                'account_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'holdings': holdings,
                'day_trade_count': int(account.daytrade_count),
                'status': account.status
            }
            
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def _execute_signal(self, signal: TradeSignal, portfolio) -> Optional[EnhancedTrade]:
        """Execute a trade signal using Alpaca API."""
        try:
            symbol = self.get_alpaca_symbol(signal.coin)
            current_price = self.get_current_price(signal.coin)
            
            if current_price <= 0:
                print(f"❌ Could not get price for {signal.coin}")
                return None
            
            # Calculate order value
            order_value = signal.quantity * current_price
            min_amount = self.min_order_amounts.get(symbol, 1.0)
            
            if order_value < min_amount:
                print(f"❌ Order value ${order_value:.2f} below minimum ${min_amount} for {symbol}")
                return None
            
            # Prepare order
            side = 'buy' if signal.action == TradeAction.BUY else 'sell'
            
            # For crypto, we use notional orders (dollar amounts)
            order_request = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'time_in_force': 'ioc',  # Immediate or cancel
                'notional': round(order_value, 2)  # Dollar amount
            }
            
            print(f"🔄 Submitting {side.upper()} order for {symbol}: ${order_value:.2f}")
            
            # Submit order
            order = self.api.submit_order(**order_request)
            
            # Wait for order to fill (up to 10 seconds)
            filled_order = self._wait_for_fill(order.id, timeout=10)
            
            if filled_order and filled_order.status == 'filled':
                # Calculate actual quantity and price from filled order
                filled_qty = float(filled_order.filled_qty)
                filled_notional = float(filled_order.filled_avg_price) * filled_qty if filled_order.filled_avg_price else order_value
                avg_price = filled_notional / filled_qty if filled_qty > 0 else current_price
                
                # Update portfolio with actual executed amounts
                if signal.action == TradeAction.BUY:
                    portfolio.execute_buy(signal.coin, filled_qty, avg_price)
                else:
                    portfolio.execute_sell(signal.coin, filled_qty, avg_price)
                
                # Create trade record
                trade = EnhancedTrade(
                    timestamp=datetime.now(),
                    coin=signal.coin,
                    action=signal.action.value,
                    quantity=filled_qty,
                    price=avg_price,
                    total_value=filled_notional,
                    reason=f"{signal.reason} [Alpaca: {order.id}]",
                    tier=signal.tier
                )
                
                print(f"✅ {side.upper()} filled: {filled_qty:.6f} {signal.coin} @ ${avg_price:.2f}")
                return trade
                
            else:
                print(f"❌ Order {order.id} not filled: {filled_order.status if filled_order else 'timeout'}")
                return None
                
        except Exception as e:
            print(f"❌ Error executing {signal.action.value} for {signal.coin}: {e}")
            return None
    
    def _wait_for_fill(self, order_id: str, timeout: int = 10) -> Optional[Any]:
        """Wait for order to fill with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = self.api.get_order(order_id)
                if order.status in ['filled', 'canceled', 'rejected']:
                    return order
                time.sleep(0.5)
            except Exception as e:
                print(f"Error checking order status: {e}")
                break
        
        return None
    
    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent order history from Alpaca."""
        try:
            orders = self.api.list_orders(
                status='all',
                limit=limit,
                nested=True
            )
            
            order_history = []
            for order in orders:
                if order.symbol.endswith('USD'):
                    # Find coin symbol
                    coin = None
                    for k, v in self.symbol_mapping.items():
                        if v == order.symbol:
                            coin = k
                            break
                    
                    if coin:
                        order_history.append({
                            'id': order.id,
                            'coin': coin,
                            'symbol': order.symbol,
                            'side': order.side,
                            'quantity': float(order.qty) if order.qty else 0.0,
                            'notional': float(order.notional) if order.notional else 0.0,
                            'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                            'status': order.status,
                            'submitted_at': order.submitted_at,
                            'filled_at': order.filled_at
                        })
            
            return order_history
            
        except Exception as e:
            print(f"Error getting order history: {e}")
            return []
    
    def sync_portfolio_with_alpaca(self, portfolio) -> None:
        """Sync local portfolio with actual Alpaca positions."""
        try:
            account_info = self.get_account_info()
            
            # Update cash balance
            # Note: Portfolio balance is managed separately, account info is for reference
            
            # Update holdings
            alpaca_holdings = account_info.get('holdings', {})
            
            # Clear local holdings and sync with Alpaca
            portfolio.holdings = {}
            for coin, position in alpaca_holdings.items():
                portfolio.holdings[coin] = position['quantity']
                
                # Update price history if needed
                current_price = self.get_current_price(coin)
                if current_price > 0:
                    portfolio.prices[coin] = current_price
            
            print(f"✅ Portfolio synced with Alpaca")
            print(f"   Cash: ${portfolio.get_usdt_balance():.2f}")
            print(f"   Holdings: {list(portfolio.holdings.keys())}")
            
        except Exception as e:
            print(f"❌ Error syncing portfolio: {e}")
    
    def get_trading_summary(self) -> str:
        """Get a summary of recent trading activity."""
        try:
            account_info = self.get_account_info()
            recent_orders = self.get_order_history(limit=10)
            
            summary = []
            summary.append("🏦 Alpaca Paper Trading Summary")
            summary.append("=" * 35)
            summary.append(f"Account Value: ${account_info.get('account_value', 0):,.2f}")
            summary.append(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            summary.append(f"Cash: ${account_info.get('cash', 0):,.2f}")
            summary.append("")
            
            holdings = account_info.get('holdings', {})
            if holdings:
                summary.append("📈 Current Positions:")
                for coin, position in holdings.items():
                    pnl_str = f"{position['unrealized_pnl']:+.2f}"
                    pnl_pct_str = f"({position['unrealized_pnl_pct']:+.1f}%)"
                    summary.append(f"  {coin}: {position['quantity']:.6f} @ ${position['avg_entry_price']:.2f} "
                                 f"= ${position['market_value']:.2f} [PnL: ${pnl_str} {pnl_pct_str}]")
            else:
                summary.append("📈 No open positions")
            
            summary.append("")
            summary.append(f"📊 Recent Orders ({len(recent_orders)}):")
            for order in recent_orders[-5:]:  # Last 5 orders
                status_emoji = "✅" if order['status'] == 'filled' else "❌"
                summary.append(f"  {status_emoji} {order['side'].upper()} {order['coin']} "
                             f"${order['notional']:.2f} - {order['status']}")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"❌ Error generating trading summary: {e}"
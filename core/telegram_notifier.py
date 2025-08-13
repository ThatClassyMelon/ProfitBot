"""
Telegram notification system for ProfitBot.
Sends hourly trading updates and important alerts.
"""
import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading


class TelegramNotifier:
    """Telegram bot for ProfitBot notifications."""
    
    def __init__(self, bot_token: str, chat_id: Optional[str] = None):
        """Initialize Telegram notifier."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Tracking
        self.last_hourly_update = 0
        self.hourly_interval = 3600  # 1 hour
        self.startup_time = time.time()
        
        # Rate limiting
        self.last_message_time = 0
        self.min_message_interval = 30  # 30 seconds between messages
        
        # Verify bot token
        self._verify_bot()
    
    def _verify_bot(self):
        """Verify bot token is valid."""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    bot_name = bot_info['result']['username']
                    print(f"📱 Telegram bot connected: @{bot_name}")
                else:
                    print(f"❌ Invalid Telegram bot token")
            else:
                print(f"❌ Failed to verify Telegram bot: {response.status_code}")
        except Exception as e:
            print(f"❌ Telegram bot verification error: {e}")
    
    def get_chat_id(self) -> Optional[str]:
        """Get chat ID from recent messages (for setup)."""
        try:
            response = requests.get(f"{self.base_url}/getUpdates", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    updates = data['result']
                    if updates:
                        # Get the most recent chat ID
                        latest_update = updates[-1]
                        chat_id = str(latest_update['message']['chat']['id'])
                        print(f"📱 Found chat ID: {chat_id}")
                        return chat_id
            return None
        except Exception as e:
            print(f"❌ Error getting chat ID: {e}")
            return None
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram."""
        if not self.chat_id:
            # Try to get chat ID automatically
            self.chat_id = self.get_chat_id()
            if not self.chat_id:
                print("❌ No Telegram chat ID available. Send a message to the bot first.")
                return False
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_message_time < self.min_message_interval:
            print("⏱️ Rate limiting Telegram message")
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(
                f"{self.base_url}/sendMessage", 
                data=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    self.last_message_time = current_time
                    return True
                else:
                    print(f"❌ Telegram API error: {result.get('description')}")
            else:
                print(f"❌ Telegram HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending Telegram message: {e}")
        
        return False
    
    def send_startup_notification(self):
        """Send bot startup notification."""
        message = (
            "🤖 *ProfitBot Started*\n"
            f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "🚀 Strategy: Optimized Momentum Scalp\n"
            "💰 Ready to hunt for profits!\n\n"
            "📊 Will send hourly updates with trading performance."
        )
        self.send_message(message)
    
    def send_trade_notification(self, trade_info: Dict[str, Any]):
        """Send immediate trade notification."""
        action_emoji = "📈" if trade_info.get('action') == 'BUY' else "📉"
        coin = trade_info.get('coin', 'Unknown')
        action = trade_info.get('action', 'Unknown')
        quantity = trade_info.get('quantity', 0)
        price = trade_info.get('price', 0)
        total_value = trade_info.get('total_value', 0)
        reason = trade_info.get('reason', 'No reason')
        
        message = (
            f"{action_emoji} *Trade Executed*\n"
            f"🪙 Coin: {coin}\n"
            f"⚡ Action: {action}\n"
            f"📊 Quantity: {quantity:.6f}\n"
            f"💰 Price: ${price:,.2f}\n"
            f"💵 Value: ${total_value:,.2f}\n"
            f"🎯 Reason: {reason}"
        )
        self.send_message(message)
    
    def get_hourly_stats(self) -> Dict[str, Any]:
        """Get trading statistics for the last hour."""
        try:
            # Read trade history
            trade_file = Path("data/trade_history.csv")
            if not trade_file.exists():
                return {"error": "No trade history found"}
            
            df = pd.read_csv(trade_file)
            if df.empty:
                return {"trades": 0, "profit_loss": 0, "volume": 0}
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_trades = df[df['timestamp'] > one_hour_ago]
            
            if recent_trades.empty:
                return {"trades": 0, "profit_loss": 0, "volume": 0}
            
            # Calculate stats
            total_trades = len(recent_trades)
            total_volume = recent_trades['total_value'].sum()
            
            # Calculate P&L (simplified)
            buys = recent_trades[recent_trades['action'] == 'BUY']
            sells = recent_trades[recent_trades['action'].str.contains('SELL')]
            
            total_buys = buys['total_value'].sum()
            total_sells = sells['total_value'].sum()
            profit_loss = total_sells - total_buys
            
            # Get coin breakdown
            coin_trades = recent_trades['coin'].value_counts().to_dict()
            
            return {
                "trades": total_trades,
                "profit_loss": profit_loss,
                "volume": total_volume,
                "coin_breakdown": coin_trades,
                "buy_volume": total_buys,
                "sell_volume": total_sells
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        try:
            # Try to read portfolio info from logs
            log_file = Path("data/trading.log")
            if not log_file.exists():
                return {"error": "No trading log found"}
            
            # Read last few lines to get current portfolio value
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for portfolio value in recent lines
            portfolio_value = 1000.0  # Default
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if "Portfolio:" in line:
                    try:
                        # Extract portfolio value
                        import re
                        match = re.search(r'Portfolio: \$([0-9,]+\.?\d*)', line)
                        if match:
                            portfolio_value = float(match.group(1).replace(',', ''))
                            break
                    except:
                        continue
            
            return {
                "portfolio_value": portfolio_value,
                "initial_value": 1000.0,
                "profit_loss": portfolio_value - 1000.0,
                "profit_loss_pct": ((portfolio_value - 1000.0) / 1000.0) * 100
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def send_hourly_update(self):
        """Send hourly trading update."""
        try:
            # Get stats
            hourly_stats = self.get_hourly_stats()
            portfolio_stats = self.get_portfolio_summary()
            
            # Calculate uptime
            uptime_hours = (time.time() - self.startup_time) / 3600
            
            # Build message
            message_parts = [
                "📊 *Hourly Trading Update*",
                f"⏰ Uptime: {uptime_hours:.1f} hours",
                ""
            ]
            
            # Portfolio summary
            if "error" not in portfolio_stats:
                portfolio_value = portfolio_stats.get('portfolio_value', 1000)
                profit_loss = portfolio_stats.get('profit_loss', 0)
                profit_pct = portfolio_stats.get('profit_loss_pct', 0)
                
                message_parts.extend([
                    "💰 *Portfolio Status*",
                    f"📈 Value: ${portfolio_value:,.2f}",
                    f"💵 P&L: ${profit_loss:+,.2f} ({profit_pct:+.2f}%)",
                    ""
                ])
            
            # Hourly trading activity
            if "error" not in hourly_stats:
                trades = hourly_stats.get('trades', 0)
                volume = hourly_stats.get('volume', 0)
                hourly_pl = hourly_stats.get('profit_loss', 0)
                
                message_parts.extend([
                    "⚡ *Last Hour Activity*",
                    f"🔄 Trades: {trades}",
                    f"📊 Volume: ${volume:,.2f}",
                    f"💰 P&L: ${hourly_pl:+,.2f}",
                ])
                
                # Coin breakdown
                coin_breakdown = hourly_stats.get('coin_breakdown', {})
                if coin_breakdown:
                    message_parts.append("🪙 Coins traded:")
                    for coin, count in coin_breakdown.items():
                        message_parts.append(f"   • {coin}: {count} trades")
            else:
                message_parts.append("❌ No trading data available")
            
            message = "\n".join(message_parts)
            self.send_message(message)
            
        except Exception as e:
            error_msg = f"❌ Error generating hourly update: {e}"
            self.send_message(error_msg)
    
    def check_hourly_update(self):
        """Check if it's time for hourly update."""
        current_time = time.time()
        if current_time - self.last_hourly_update >= self.hourly_interval:
            self.send_hourly_update()
            self.last_hourly_update = current_time
    
    def send_error_alert(self, error_message: str):
        """Send error alert."""
        message = (
            "🚨 *ProfitBot Alert*\n"
            f"❌ Error: {error_message}\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_message(message)
    
    def send_shutdown_notification(self):
        """Send bot shutdown notification."""
        uptime_hours = (time.time() - self.startup_time) / 3600
        
        message = (
            "🛑 *ProfitBot Stopped*\n"
            f"⏰ Uptime: {uptime_hours:.1f} hours\n"
            f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "💤 Bot is now offline."
        )
        self.send_message(message)
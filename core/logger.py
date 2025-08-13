"""
Comprehensive logging system for the trading bot.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from utils.math_tools import format_currency, format_percentage


class TradingLogger:
    """Handles all logging for the trading bot."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading logger.
        
        Args:
            config: Configuration dictionary
        """
        self.log_file = config['logging']['log_file']
        self.log_level = config['logging']['log_level']
        self._ensure_log_directory()
        self._setup_logger()
    
    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _setup_logger(self) -> None:
        """Setup the logger with appropriate formatting."""
        # Create logger
        self.logger = logging.getLogger('ProfitBot')
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, self.log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set formatter for handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_bot_start(self, config: Dict[str, Any], strategy=None) -> None:
        """
        Log bot startup information.
        
        Args:
            config: Configuration dictionary
            strategy: Trading strategy instance
        """
        self.logger.info("=" * 60)
        self.logger.info("🤖 ProfitBot Trading Simulator Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Initial Balance: {format_currency(config['initial_balance'])}")
        self.logger.info(f"Tracked Coins: {', '.join(config['coins'].keys())}")
        self.logger.info(f"Movement Threshold: {format_percentage(config['strategy']['movement_threshold'])}")
        self.logger.info(f"Tick Interval: {config['simulation']['tick_interval']}s")
        self.logger.info("-" * 60)
        
        # Add strategy explanation on startup
        if strategy and hasattr(strategy, 'get_strategy_explanation'):
            try:
                strategy_explanation = strategy.get_strategy_explanation()
                self.logger.info("\n📚 HOW THIS BOT WORKS:")
                self.logger.info("=" * 60)
                self.logger.info(strategy_explanation)
                self.logger.info("=" * 60)
            except Exception as e:
                self.logger.info(f"Strategy explanation error: {e}")
    
    def log_bot_stop(self, final_portfolio_value: float, initial_balance: float, 
                    runtime_seconds: float) -> None:
        """
        Log bot shutdown information.
        
        Args:
            final_portfolio_value: Final portfolio value
            initial_balance: Initial balance
            runtime_seconds: Total runtime in seconds
        """
        profit_loss = final_portfolio_value - initial_balance
        profit_loss_percent = (profit_loss / initial_balance) * 100 if initial_balance > 0 else 0
        
        self.logger.info("-" * 60)
        self.logger.info("🛑 ProfitBot Trading Simulator Stopped")
        self.logger.info(f"Final Portfolio Value: {format_currency(final_portfolio_value)}")
        self.logger.info(f"Profit/Loss: {format_currency(profit_loss)} ({profit_loss_percent:+.2f}%)")
        self.logger.info(f"Runtime: {runtime_seconds:.1f} seconds")
        self.logger.info("=" * 60)
    
    def log_trading_cycle(self, current_prices: Dict[str, float], portfolio_summary: Dict[str, Any], 
                         signals: List, executed_trades: List, strategy=None) -> None:
        """
        Log a complete trading cycle.
        
        Args:
            current_prices: Current market prices
            portfolio_summary: Portfolio summary
            signals: Trading signals
            executed_trades: Executed trades
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log portfolio status
        total_value = portfolio_summary['total_value']
        self.logger.info(f"[{timestamp}] | Portfolio: {format_currency(total_value)}")
        
        # Log coin prices and movement requirements
        for coin, price in current_prices.items():
            signal = next((s for s in signals if s.coin == coin), None)
            if signal and hasattr(signal, 'reason'):
                self.logger.info(f"{coin}: ${price:.2f} ({signal.reason})")
            else:
                self.logger.info(f"{coin}: ${price:.2f}")
        
        # Log executed trades if any
        if executed_trades:
            self.logger.info("Executed Trades:")
            for trade in executed_trades:
                trade_value = format_currency(trade.total_value)
                self.logger.info(
                    f"  {trade.action} {trade.quantity:.6f} {trade.coin} "
                    f"@ ${trade.price:.2f} = {trade_value} ({trade.reason})"
                )
        
        # Log trade explanations for executed trades
        if executed_trades and strategy:
            self.logger.info("\n📋 TRADE EXPLANATIONS:")
            try:
                for i, signal in enumerate(signals):
                    if (signal.action.value in ['BUY', 'SELL', 'REBALANCE'] and 
                        hasattr(signal, 'explanation') and signal.explanation):
                        
                        self.logger.info("=" * 50)
                        self.logger.info(signal.explanation)
                        self.logger.info("=" * 50)
            except Exception as e:
                self.logger.info(f"Trade explanation error: {e}")
        
        # Log market conditions explanation (once per few cycles)
        if strategy and hasattr(strategy, 'get_market_explanation'):
            import time
            if not hasattr(self, '_last_market_explanation_time'):
                self._last_market_explanation_time = 0
            
            # Show market explanation every 5 minutes
            if time.time() - self._last_market_explanation_time > 300:
                try:
                    market_explanation = strategy.get_market_explanation(current_prices)
                    self.logger.info("\n🌍 MARKET CONDITIONS:")
                    self.logger.info("=" * 50)
                    self.logger.info(market_explanation)
                    self.logger.info("=" * 50)
                    self._last_market_explanation_time = time.time()
                except Exception as e:
                    self.logger.info(f"Market explanation error: {e}")
        
        self.logger.info("")  # Empty line for readability
    
    def log_price_update(self, current_prices: Dict[str, float]) -> None:
        """
        Log price updates (debug level).
        
        Args:
            current_prices: Current market prices
        """
        price_info = " | ".join([f"{coin}: ${price:.2f}" for coin, price in current_prices.items()])
        self.logger.debug(f"Price Update - {price_info}")
    
    def log_trade_execution(self, trade) -> None:
        """
        Log individual trade execution.
        
        Args:
            trade: Executed trade object
        """
        trade_value = format_currency(trade.total_value)
        self.logger.info(
            f"✅ Trade Executed: {trade.action} {trade.quantity:.6f} {trade.coin} "
            f"@ ${trade.price:.2f} = {trade_value}"
        )
    
    def log_strategy_signal(self, signal) -> None:
        """
        Log strategy signals (debug level).
        
        Args:
            signal: Trading signal
        """
        self.logger.debug(f"Strategy Signal: {signal}")
    
    def log_portfolio_update(self, portfolio_summary: Dict[str, Any]) -> None:
        """
        Log portfolio updates (debug level).
        
        Args:
            portfolio_summary: Portfolio summary
        """
        total_value = format_currency(portfolio_summary['total_value'])
        usdt_balance = format_currency(portfolio_summary['usdt_balance'])
        profit_loss = portfolio_summary['profit_loss_percent']
        
        self.logger.debug(f"Portfolio Update - Total: {total_value}, USDT: {usdt_balance}, P/L: {profit_loss:+.2f}%")
    
    def log_error(self, error_message: str, exception: Exception = None) -> None:
        """
        Log errors.
        
        Args:
            error_message: Error description
            exception: Exception object if available
        """
        if exception:
            self.logger.error(f"❌ {error_message}: {str(exception)}")
        else:
            self.logger.error(f"❌ {error_message}")
    
    def log_warning(self, warning_message: str) -> None:
        """
        Log warnings.
        
        Args:
            warning_message: Warning message
        """
        self.logger.warning(f"⚠️  {warning_message}")
    
    def log_info(self, info_message: str) -> None:
        """
        Log general information.
        
        Args:
            info_message: Information message
        """
        self.logger.info(info_message)
    
    def log_debug(self, debug_message: str) -> None:
        """
        Log debug information.
        
        Args:
            debug_message: Debug message
        """
        self.logger.debug(debug_message)
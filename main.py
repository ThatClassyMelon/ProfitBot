#!/usr/bin/env python3
"""
ProfitBot - A modular crypto trading bot simulator.

This is the main entry point for the trading bot simulation.
Run with: python main.py
"""
import yaml
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from core.price_fetcher import RealTimePriceSimulator
from core.portfolio import Portfolio
from core.multi_signal_strategy import MultiSignalStrategy
from core.optimized_strategy import OptimizedMomentumStrategy
from core.enhanced_executor import EnhancedTradeExecutor
from core.alpaca_executor import AlpacaExecutor
from core.vectorbt_backtester import VectorBTBacktester
from core.logger import TradingLogger
from core.telegram_notifier import TelegramNotifier


class ProfitBot:
    """Main trading bot class that orchestrates all components."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the ProfitBot.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.running = False
        self.start_time = None
        
        # Initialize components
        self.logger = TradingLogger(self.config)
        self.simulator = RealTimePriceSimulator(self.config)
        self.portfolio = Portfolio(self.config['initial_balance'])
        
        # Use optimized strategy by default
        strategy_mode = self.config.get('strategy_mode', 'optimized')
        if strategy_mode == 'optimized':
            self.strategy = OptimizedMomentumStrategy(self.config)
            print("🚀 Using Optimized Momentum Scalp Strategy")
        else:
            self.strategy = MultiSignalStrategy(self.config)
            print("📊 Using Multi-Signal Strategy")
        
        # Choose executor based on configuration
        alpaca_config = self.config.get('alpaca', {})
        if (alpaca_config.get('use_paper_trading', False) and 
            alpaca_config.get('api_key') and 
            alpaca_config.get('api_key') != "YOUR_ALPACA_API_KEY"):
            print("🏦 Using Alpaca Paper Trading")
            self.executor = AlpacaExecutor(self.config)
            self.using_alpaca = True
        else:
            print("🎲 Using Mock Trading (Simulation)")
            self.executor = EnhancedTradeExecutor(self.config)
            self.using_alpaca = False
        
        # Initialize Telegram notifications
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('enable_notifications', False):
            try:
                self.telegram = TelegramNotifier(
                    bot_token=telegram_config.get('bot_token'),
                    chat_id=telegram_config.get('chat_id') or None
                )
                print("📱 Telegram notifications enabled")
            except Exception as e:
                print(f"⚠️ Telegram setup failed: {e}")
                self.telegram = None
        else:
            self.telegram = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"❌ Configuration file '{config_file}' not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.log_info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self) -> None:
        """Start the trading bot simulation."""
        self.running = True
        self.start_time = datetime.now()
        
        # Log startup information
        self.logger.log_bot_start(self.config, self.strategy)
        
        # Send startup notification
        if self.telegram:
            self.telegram.send_startup_notification()
        
        try:
            self._run_simulation()
        except KeyboardInterrupt:
            self.logger.log_info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.log_error("Unexpected error occurred", e)
            if self.telegram:
                self.telegram.send_error_alert(str(e))
            raise
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading bot simulation."""
        if not self.running:
            return
        
        self.running = False
        
        # Calculate runtime
        runtime_seconds = 0
        if self.start_time:
            runtime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Get final portfolio value
        current_prices = self.simulator.get_current_prices()
        final_value = self.portfolio.calculate_portfolio_value(current_prices)
        
        # Log shutdown information
        self.logger.log_bot_stop(final_value, self.config['initial_balance'], runtime_seconds)
        
        # Send shutdown notification
        if self.telegram:
            self.telegram.send_shutdown_notification()
        
        # Display final statistics
        self._display_final_statistics()
    
    def _run_simulation(self) -> None:
        """Run the main simulation loop."""
        # Use real price update interval if using real prices, otherwise simulation interval
        if self.simulator.is_using_real_prices():
            tick_interval = self.config['api']['price_update_interval']
            self.logger.log_info(f"📊 Using real-time prices with {tick_interval}s update interval")
        else:
            tick_interval = self.config['simulation']['tick_interval']
            self.logger.log_info(f"🎲 Using simulated prices with {tick_interval}s update interval")
        
        while self.running:
            try:
                # Update prices
                current_prices = self.simulator.update_prices()
                
                # Get portfolio summary
                portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
                
                # Analyze market and generate signals
                signals = self.strategy.analyze_market(
                    current_prices, 
                    self.portfolio, 
                    self.portfolio.last_trade_prices
                )
                
                # Execute trades
                executed_trades = self.executor.execute_signals(signals, self.portfolio, self.strategy)
                
                # Send trade notifications
                if self.telegram and executed_trades:
                    telegram_config = self.config.get('telegram', {})
                    if telegram_config.get('trade_alerts', True):
                        for trade in executed_trades:
                            trade_info = {
                                'coin': trade.coin,
                                'action': trade.action,
                                'quantity': trade.quantity,
                                'price': trade.price,
                                'total_value': trade.total_value,
                                'reason': trade.reason
                            }
                            self.telegram.send_trade_notification(trade_info)
                
                # Check for hourly updates
                if self.telegram:
                    telegram_config = self.config.get('telegram', {})
                    if telegram_config.get('hourly_updates', True):
                        self.telegram.check_hourly_update()
                
                # Log the trading cycle
                self.logger.log_trading_cycle(
                    current_prices, portfolio_summary, signals, executed_trades, self.strategy
                )
                
                # Wait for next tick
                time.sleep(tick_interval)
                
            except Exception as e:
                self.logger.log_error("Error in simulation loop", e)
                time.sleep(tick_interval)  # Continue after error
    
    def _display_final_statistics(self) -> None:
        """Display final trading statistics."""
        try:
            # Portfolio statistics
            current_prices = self.simulator.get_current_prices()
            portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
            
            # Enhanced trade statistics
            trade_stats = self.executor.get_enhanced_statistics()
            dca_stats = self.executor.get_dca_statistics()
            
            print("\n" + "=" * 60)
            print("📊 FINAL STATISTICS")
            print("=" * 60)
            
            # Portfolio stats
            print(f"Initial Balance: ${self.config['initial_balance']:.2f}")
            print(f"Final Portfolio Value: ${portfolio_summary['total_value']:.2f}")
            print(f"USDT Balance: ${portfolio_summary['usdt_balance']:.2f}")
            print(f"Profit/Loss: ${portfolio_summary['profit_loss']:+.2f} ({portfolio_summary['profit_loss_percent']:+.2f}%)")
            
            # Holdings
            if portfolio_summary['holdings']:
                print("\nCurrent Holdings:")
                for coin, holding_info in portfolio_summary['holdings'].items():
                    print(f"  {coin}: {holding_info['quantity']:.6f} @ ${holding_info['price']:.2f} = ${holding_info['value']:.2f}")
            
            # Enhanced trade stats
            print(f"\nTotal Trades: {trade_stats['total_trades']}")
            print(f"Buy Trades: {trade_stats['buy_trades']}")
            print(f"Sell Trades: {trade_stats['sell_trades']}")
            print(f"Rebalance Trades: {trade_stats['rebalance_trades']}")
            print(f"Total Volume: ${trade_stats['total_volume']:.2f}")
            print(f"Realized P&L: ${trade_stats['realized_pnl']:+.2f}")
            
            if trade_stats['total_trades'] > 0:
                print(f"Average Trade Value: ${trade_stats['avg_trade_value']:.2f}")
            
            # DCA Statistics
            if dca_stats['total_dca_trades'] > 0:
                print(f"\nDCA Statistics:")
                print(f"Total DCA Trades: {dca_stats['total_dca_trades']}")
                print(f"Total DCA Value: ${dca_stats['total_dca_value']:.2f}")
                print(f"Average DCA Size: ${dca_stats['avg_dca_size']:.2f}")
                
                for tier, stats in dca_stats['dca_by_tier'].items():
                    print(f"  Tier {tier}: {stats['count']} trades, ${stats['total_value']:.2f}")
            
            print("=" * 60)
            
        except Exception as e:
            self.logger.log_error("Error displaying final statistics", e)
    
    def run_backtest(self, days: int = 30) -> None:
        """Run a comprehensive backtest using VectorBT."""
        print(f"🧪 Starting Backtest Mode ({days} days)")
        print("=" * 50)
        
        try:
            backtester = VectorBTBacktester(self.config)
            results = backtester.backtest_strategy(days=days)
            
            # Display results
            report = backtester.generate_report(results)
            print(report)
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"data/backtest_report_{timestamp}.txt"
            
            # Ensure data directory exists
            import os
            os.makedirs("data", exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📋 Report saved to: {report_file}")
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_optimization(self, coin: str = "BTC", days: int = 30) -> None:
        """Run parameter optimization for a specific coin."""
        print(f"🔧 Starting Parameter Optimization for {coin} ({days} days)")
        print("=" * 60)
        
        try:
            backtester = VectorBTBacktester(self.config)
            
            # Get optimization ranges from config
            optimization_ranges = self.config.get('backtesting', {}).get('optimization_ranges', {
                'movement_threshold': [0.02, 0.03, 0.04, 0.05],
                'min_signal_strength': [0.3, 0.4, 0.5, 0.6]
            })
            
            results = backtester.optimize_parameters(coin, optimization_ranges, days)
            
            print(f"✅ Optimization completed for {coin}")
            print(f"Best Parameters: {results['best_parameters']}")
            print(f"Best Return: {results['best_return']:.2f}%")
            print(f"Combinations Tested: {results['total_combinations_tested']}")
            
            # Save optimization results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            opt_file = f"data/optimization_{coin}_{timestamp}.txt"
            
            import os
            os.makedirs("data", exist_ok=True)
            
            with open(opt_file, 'w') as f:
                f.write(f"Parameter Optimization Results for {coin}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Best Parameters: {results['best_parameters']}\n")
                f.write(f"Best Return: {results['best_return']:.2f}%\n")
                f.write(f"Combinations Tested: {results['total_combinations_tested']}\n\n")
                
                f.write("All Results:\n")
                for i, result in enumerate(results['all_results']):
                    f.write(f"{i+1}. {result}\n")
            
            print(f"📋 Optimization results saved to: {opt_file}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ProfitBot - Crypto Trading Bot')
    parser.add_argument('--mode', choices=['trade', 'backtest', 'optimize'], 
                       default='trade', help='Operation mode')
    parser.add_argument('--days', type=int, default=30, 
                       help='Days of historical data for backtest/optimization')
    parser.add_argument('--coin', type=str, default='BTC',
                       help='Coin for optimization mode')
    
    args = parser.parse_args()
    
    try:
        bot = ProfitBot()
        
        if args.mode == 'trade':
            print("🤖 Starting ProfitBot Trading Mode...")
            print("Press Ctrl+C to stop the bot gracefully.\n")
            bot.start()
        elif args.mode == 'backtest':
            bot.run_backtest(args.days)
        elif args.mode == 'optimize':
            bot.run_optimization(args.coin, args.days)
            
    except Exception as e:
        print(f"❌ Failed to start ProfitBot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
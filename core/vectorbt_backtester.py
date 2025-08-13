"""
VectorBT backtesting integration for automated strategy testing.
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import requests
import time

from core.multi_signal_strategy import MultiSignalStrategy
from core.advanced_data_fetcher import AdvancedDataFetcher
from core.enhanced_strategy import TradeSignal, TradeAction


class VectorBTBacktester:
    """Automated backtesting engine using VectorBT."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the backtester."""
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        
        # Backtesting parameters
        self.initial_balance = self.backtest_config.get('initial_balance', 10000.0)
        self.commission = self.backtest_config.get('commission', 0.001)  # 0.1%
        self.slippage = self.backtest_config.get('slippage', 0.001)  # 0.1%
        
        # Data fetcher for historical data
        self.data_fetcher = AdvancedDataFetcher(config)
        
        # Strategy instance for testing
        self.strategy = MultiSignalStrategy(config)
        
        # Results storage
        self.results = {}
        
    def fetch_historical_data(self, coin: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical price data for backtesting.
        
        Args:
            coin: Cryptocurrency symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use CoinGecko API for historical data
            coin_id_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'SOL': 'solana',
                'XRP': 'ripple',
                'ADA': 'cardano',
                'DOT': 'polkadot',
                'MATIC': 'polygon',
                'AVAX': 'avalanche-2'
            }
            
            coin_id = coin_id_map.get(coin.upper(), coin.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': str(days)
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume (approximate based on price movement)
            df['volume'] = abs(df['close'].pct_change()) * 1000000  # Simplified volume
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {coin}: {e}")
            
            # Generate synthetic data as fallback
            return self._generate_synthetic_data(coin, days)
    
    def _generate_synthetic_data(self, coin: str, days: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='1h'
        )
        
        # Get volatility from config
        volatility = self.config['coins'].get(coin, {}).get('volatility', 0.1)
        initial_price = self.config['coins'].get(coin, {}).get('initial_price', 100.0)
        
        # Generate random walk
        returns = np.random.normal(0, volatility/24, len(dates))  # Hourly volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(df)))
        df['volume'] = np.random.uniform(100000, 2000000, len(df))
        
        return df
    
    def simulate_strategy_signals(self, price_data: pd.DataFrame, coin: str) -> pd.DataFrame:
        """
        Simulate strategy signals over historical data.
        
        Args:
            price_data: Historical OHLCV data
            coin: Cryptocurrency symbol
            
        Returns:
            DataFrame with buy/sell signals
        """
        signals = []
        
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            # Create mock market data
            market_data = {
                coin: {
                    'price': row['close'],
                    'price_change_24h': row['close'] / row['open'] - 1 if i > 0 else 0,
                    'volume_24h': row['volume'],
                    'market_cap': row['close'] * 1000000,  # Simplified
                    'technical_indicators': {
                        'rsi': 50 + np.random.uniform(-20, 20),  # Simplified RSI
                        'macd': np.random.uniform(-1, 1),
                        'bollinger_position': np.random.uniform(0, 1)
                    }
                }
            }
            
            # Get strategy signals
            try:
                strategy_signals = self.strategy.analyze_market(market_data)
                
                for signal in strategy_signals:
                    if signal.coin == coin:
                        signals.append({
                            'timestamp': timestamp,
                            'action': signal.action.value,
                            'quantity': signal.quantity,
                            'price': signal.price,
                            'strength': signal.strength,
                            'reason': signal.reason
                        })
            except Exception as e:
                # Continue on errors during simulation
                continue
        
        return pd.DataFrame(signals)
    
    def run_vectorbt_backtest(self, price_data: pd.DataFrame, signals: pd.DataFrame, coin: str) -> Dict[str, Any]:
        """
        Run backtest using VectorBT.
        
        Args:
            price_data: Historical price data
            signals: Trading signals
            coin: Cryptocurrency symbol
            
        Returns:
            Backtest results dictionary
        """
        try:
            # Prepare price series
            price_series = price_data['close']
            
            # Create signal arrays
            buy_signals = np.zeros(len(price_series), dtype=bool)
            sell_signals = np.zeros(len(price_series), dtype=bool)
            
            # Map signals to price data timestamps
            for _, signal in signals.iterrows():
                # Find closest timestamp in price data
                closest_idx = price_series.index.get_indexer([signal['timestamp']], method='nearest')[0]
                
                if signal['action'] == 'BUY':
                    buy_signals[closest_idx] = True
                elif signal['action'] == 'SELL':
                    sell_signals[closest_idx] = True
            
            # Run portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                price_series,
                buy_signals,
                sell_signals,
                init_cash=self.initial_balance,
                fees=self.commission,
                slippage=self.slippage,
                freq='1h'
            )
            
            # Calculate metrics - handle missing statistics gracefully
            stats = portfolio.stats()
            
            def safe_get_stat(stat_name, default=0.0):
                try:
                    return float(stats.get(stat_name, default))
                except (KeyError, TypeError, ValueError):
                    return default
            
            results = {
                'coin': coin,
                'start_date': price_series.index[0],
                'end_date': price_series.index[-1],
                'initial_balance': self.initial_balance,
                'final_value': float(portfolio.value().iloc[-1]),
                'total_return': safe_get_stat('Total Return [%]'),
                'total_trades': int(safe_get_stat('Total Trades', 0)),
                'win_rate': safe_get_stat('Win Rate [%]'),
                'profit_factor': safe_get_stat('Profit Factor', 1.0),
                'max_drawdown': safe_get_stat('Max Drawdown [%]'),
                'sharpe_ratio': safe_get_stat('Sharpe Ratio'),
                'calmar_ratio': safe_get_stat('Calmar Ratio'),
                'avg_trade_return': safe_get_stat('Avg Trade [%]'),
                'best_trade': safe_get_stat('Best Trade [%]'),
                'worst_trade': safe_get_stat('Worst Trade [%]'),
                'buy_and_hold_return': float((price_series.iloc[-1] / price_series.iloc[0] - 1) * 100),
                'signals_generated': len(signals),
                'buy_signals': int(buy_signals.sum()),
                'sell_signals': int(sell_signals.sum())
            }
            
            # Store portfolio object for detailed analysis
            results['portfolio'] = portfolio
            results['price_data'] = price_data
            results['signals_data'] = signals
            
            return results
            
        except Exception as e:
            print(f"Error running VectorBT backtest: {e}")
            return {
                'coin': coin,
                'error': str(e),
                'signals_generated': len(signals)
            }
    
    def backtest_strategy(self, coins: List[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive backtest for specified coins.
        
        Args:
            coins: List of coins to test (default: all coins from config)
            days: Number of days of historical data
            
        Returns:
            Complete backtest results
        """
        if coins is None:
            coins = list(self.config['coins'].keys())
        
        print(f"🧪 Starting VectorBT backtest for {len(coins)} coins over {days} days")
        
        all_results = {}
        
        for coin in coins:
            print(f"📊 Testing {coin}...")
            
            try:
                # Fetch historical data
                price_data = self.fetch_historical_data(coin, days)
                
                if price_data.empty:
                    print(f"❌ No data available for {coin}")
                    continue
                
                # Generate signals
                signals = self.simulate_strategy_signals(price_data, coin)
                
                # Run backtest
                results = self.run_vectorbt_backtest(price_data, signals, coin)
                all_results[coin] = results
                
                # Brief results summary
                if 'error' not in results:
                    print(f"✅ {coin}: {results['total_return']:.2f}% return, "
                          f"{results['total_trades']} trades, "
                          f"{results['win_rate']:.1f}% win rate")
                else:
                    print(f"❌ {coin}: {results['error']}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error testing {coin}: {e}")
                all_results[coin] = {'error': str(e)}
        
        # Generate summary
        summary = self._generate_backtest_summary(all_results)
        all_results['summary'] = summary
        
        self.results = all_results
        return all_results
    
    def _generate_backtest_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all tested coins."""
        successful_tests = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_tests:
            return {'error': 'No successful backtests'}
        
        # Aggregate metrics
        total_returns = [r['total_return'] for r in successful_tests.values()]
        win_rates = [r['win_rate'] for r in successful_tests.values()]
        sharpe_ratios = [r['sharpe_ratio'] for r in successful_tests.values() if not np.isnan(r['sharpe_ratio'])]
        max_drawdowns = [r['max_drawdown'] for r in successful_tests.values()]
        
        # Calculate portfolio-level metrics (equal weight)
        portfolio_return = np.mean(total_returns)
        portfolio_volatility = np.std(total_returns)
        portfolio_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        
        # Best and worst performers
        best_coin = max(successful_tests.keys(), key=lambda k: successful_tests[k]['total_return'])
        worst_coin = min(successful_tests.keys(), key=lambda k: successful_tests[k]['total_return'])
        
        summary = {
            'tested_coins': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(results) - len(successful_tests),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe': portfolio_sharpe,
            'avg_win_rate': np.mean(win_rates),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'best_performer': {
                'coin': best_coin,
                'return': successful_tests[best_coin]['total_return']
            },
            'worst_performer': {
                'coin': worst_coin,
                'return': successful_tests[worst_coin]['total_return']
            },
            'total_trades': sum(r['total_trades'] for r in successful_tests.values()),
            'total_signals': sum(r['signals_generated'] for r in successful_tests.values())
        }
        
        return summary
    
    def optimize_parameters(self, coin: str, parameter_ranges: Dict[str, List], days: int = 30) -> Dict[str, Any]:
        """
        Run parameter optimization for a specific coin.
        
        Args:
            coin: Cryptocurrency to optimize
            parameter_ranges: Dictionary of parameter names and their test ranges
            days: Number of days of historical data
            
        Returns:
            Optimization results
        """
        print(f"🔧 Optimizing parameters for {coin}...")
        
        # Fetch data once
        price_data = self.fetch_historical_data(coin, days)
        
        best_params = None
        best_return = float('-inf')
        optimization_results = []
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            
            # Update strategy config temporarily
            original_config = self.config['strategy'].copy()
            self.config['strategy'].update(param_dict)
            
            try:
                # Re-initialize strategy with new parameters
                test_strategy = MultiSignalStrategy(self.config)
                
                # Generate signals with new parameters
                signals = self.simulate_strategy_signals(price_data, coin)
                
                # Run backtest
                results = self.run_vectorbt_backtest(price_data, signals, coin)
                
                if 'error' not in results:
                    total_return = results['total_return']
                    optimization_results.append({
                        'parameters': param_dict.copy(),
                        'return': total_return,
                        'sharpe': results['sharpe_ratio'],
                        'max_drawdown': results['max_drawdown'],
                        'win_rate': results['win_rate'],
                        'total_trades': results['total_trades']
                    })
                    
                    if total_return > best_return:
                        best_return = total_return
                        best_params = param_dict.copy()
                
            except Exception as e:
                print(f"Error testing parameters {param_dict}: {e}")
            
            finally:
                # Restore original config
                self.config['strategy'] = original_config
        
        return {
            'coin': coin,
            'best_parameters': best_params,
            'best_return': best_return,
            'total_combinations_tested': len(optimization_results),
            'all_results': optimization_results
        }
    
    def generate_report(self, results: Dict[str, Any] = None) -> str:
        """Generate a comprehensive backtest report."""
        if results is None:
            results = self.results
        
        if not results:
            return "No backtest results available."
        
        report = []
        report.append("📈 VectorBT Backtest Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'summary' in results:
            summary = results['summary']
            report.append("📊 Portfolio Summary:")
            report.append(f"  Tested Coins: {summary['tested_coins']}")
            report.append(f"  Successful Tests: {summary['successful_tests']}")
            report.append(f"  Portfolio Return: {summary['portfolio_return']:.2f}%")
            report.append(f"  Portfolio Volatility: {summary['portfolio_volatility']:.2f}%")
            report.append(f"  Portfolio Sharpe: {summary['portfolio_sharpe']:.2f}")
            report.append(f"  Average Win Rate: {summary['avg_win_rate']:.1f}%")
            report.append(f"  Average Max Drawdown: {summary['avg_max_drawdown']:.2f}%")
            report.append(f"  Best Performer: {summary['best_performer']['coin']} ({summary['best_performer']['return']:.2f}%)")
            report.append(f"  Worst Performer: {summary['worst_performer']['coin']} ({summary['worst_performer']['return']:.2f}%)")
            report.append(f"  Total Trades: {summary['total_trades']}")
            report.append("")
        
        report.append("🪙 Individual Coin Results:")
        for coin, result in results.items():
            if coin == 'summary':
                continue
            
            if 'error' in result:
                report.append(f"  {coin}: ❌ {result['error']}")
            else:
                report.append(f"  {coin}:")
                report.append(f"    Return: {result['total_return']:.2f}% (vs B&H: {result['buy_and_hold_return']:.2f}%)")
                report.append(f"    Trades: {result['total_trades']} ({result['win_rate']:.1f}% win rate)")
                report.append(f"    Sharpe: {result['sharpe_ratio']:.2f}, Max DD: {result['max_drawdown']:.2f}%")
                report.append(f"    Signals: {result['signals_generated']} ({result['buy_signals']} buy, {result['sell_signals']} sell)")
        
        return "\n".join(report)
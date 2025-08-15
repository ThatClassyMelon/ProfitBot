#!/usr/bin/env python3
"""
VectorBT backtesting module for ProfitBot strategies.
"""
import vectorbt as vbt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
from typing import Dict, Tuple, List
import yaml

from core.config_loader import load_config_with_env
from core.optimized_strategy import OptimizedMomentumStrategy


def fetch_crypto_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical cryptocurrency data.
    
    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, XRP)
        days: Number of days of historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Map symbols to yfinance tickers
        symbol_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'SOL': 'SOL-USD',
            'XRP': 'XRP-USD'
        }
        
        ticker = symbol_map.get(symbol, f"{symbol}-USD")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data using yfinance
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',  # Hourly data for better granularity
            progress=False,
            auto_adjust=False  # Prevent column issues
        )
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Handle column names properly
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure we have standard column names
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if 'Adj Close' in data.columns:
            data = data.drop('Adj Close', axis=1)
        
        # Forward fill any missing values
        data = data.ffill()
        
        print(f"✅ Fetched {len(data)} hours of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        # Return mock data as fallback
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='1h'
        )
        
        # Create realistic mock data with random walk
        np.random.seed(42)  # Reproducible results
        base_price = {'BTC': 60000, 'ETH': 3000, 'SOL': 150, 'XRP': 0.6}.get(symbol, 100)
        
        returns = np.random.normal(0, 0.02, len(dates))  # 2% hourly volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        print(f"⚠️ Using mock data for {symbol}")
        return data


def simulate_optimized_strategy(data: pd.DataFrame, config: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate the optimized momentum strategy using vectorized operations.
    
    Args:
        data: OHLCV data
        config: Strategy configuration
        
    Returns:
        Tuple of (entries, exits) boolean series
    """
    prices = data['Close']
    volume = data['Volume']
    
    # Calculate EMAs
    ema_fast = prices.ewm(span=5).mean()
    ema_slow = prices.ewm(span=13).mean()
    
    # Calculate RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate momentum signals
    momentum = (ema_fast - ema_slow) / ema_slow
    
    # EMA crossover signals
    crossover = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    
    # RSI signals
    rsi_oversold = rsi < 35
    rsi_mild_oversold = (rsi >= 35) & (rsi < 45)
    rsi_overbought = rsi > 65
    rsi_mild_overbought = (rsi <= 65) & (rsi > 55)
    
    # Volume filter
    volume_avg = volume.rolling(window=10).mean()
    volume_filter = volume >= (volume_avg * 1.2)
    
    # Combined entry signals
    momentum_threshold = 0.001  # 0.1%
    
    # Buy signals: momentum + crossover + RSI + volume
    entries = (
        (momentum > momentum_threshold) &  # Basic momentum
        (crossover | rsi_oversold | rsi_mild_oversold) &  # Technical signals
        volume_filter  # Volume confirmation
    )
    
    # Exit signals: take profit (1%) or stop loss (0.5%) or RSI overbought
    entry_prices = prices.where(entries).ffill()
    
    # Calculate returns since entry
    returns_since_entry = (prices / entry_prices - 1).fillna(0)
    
    # Exit conditions
    take_profit = returns_since_entry >= 0.01  # 1% profit
    stop_loss = returns_since_entry <= -0.005  # 0.5% loss
    rsi_exit = rsi_overbought | rsi_mild_overbought
    
    # Time-based exit (15 periods = 15 hours max hold)
    periods_since_entry = entries.cumsum() - entries.cumsum().where(~entries).ffill()
    time_exit = periods_since_entry >= 15
    
    exits = take_profit | stop_loss | rsi_exit | time_exit
    
    return entries, exits


def run_vectorbt_backtest(coins: List[str], days: int = 30, initial_balance: float = 1000.0) -> Dict:
    """
    Run comprehensive VectorBT backtest for multiple coins.
    
    Args:
        coins: List of cryptocurrency symbols
        days: Number of days to backtest
        initial_balance: Starting balance in USD
        
    Returns:
        Dictionary with backtest results
    """
    print(f"🚀 Starting VectorBT backtest for {len(coins)} coins over {days} days")
    print(f"💰 Initial balance: ${initial_balance:,.2f}")
    print("=" * 60)
    
    # Load configuration
    config = load_config_with_env('config.yaml')
    
    all_results = {}
    
    for coin in coins:
        print(f"\n📊 Backtesting {coin}...")
        
        try:
            # Fetch historical data
            data = fetch_crypto_data(coin, days)
            
            if len(data) < 50:  # Need minimum data for indicators
                print(f"❌ Insufficient data for {coin}")
                continue
            
            # Simulate strategy
            entries, exits = simulate_optimized_strategy(data, config)
            
            # Run VectorBT portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries,
                exits,
                init_cash=initial_balance,
                fees=0.001,  # 0.1% trading fees
                freq='1H'
            )
            
            # Calculate performance metrics
            stats = portfolio.stats()
            
            # Additional metrics
            trades = portfolio.trades.records_readable
            
            # Safely get trade duration
            avg_duration = 0
            if trades is not None and len(trades) > 0:
                if hasattr(trades, 'columns') and 'Duration' in trades.columns:
                    avg_duration = trades['Duration'].mean()
                elif hasattr(trades, 'columns') and 'duration' in trades.columns:
                    avg_duration = trades['duration'].mean()
            
            results = {
                'total_return': stats['Total Return [%]'],
                'total_return_dollars': portfolio.value().iloc[-1] - initial_balance,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max Drawdown [%]'],
                'win_rate': stats['Win Rate [%]'] if 'Win Rate [%]' in stats else 0,
                'total_trades': len(trades) if trades is not None else 0,
                'avg_trade_duration': avg_duration,
                'profit_factor': stats['Profit Factor'] if 'Profit Factor' in stats else 0,
                'final_value': portfolio.value().iloc[-1],
                'data_points': len(data),
                'strategy_stats': stats
            }
            
            all_results[coin] = results
            
            # Print results
            print(f"✅ {coin} Results:")
            print(f"   Total Return: {results['total_return']:.2f}%")
            print(f"   Dollar Return: ${results['total_return_dollars']:+.2f}")
            print(f"   Final Value: ${results['final_value']:,.2f}")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"   Win Rate: {results['win_rate']:.1f}%")
            print(f"   Total Trades: {results['total_trades']}")
            
        except Exception as e:
            print(f"❌ Error backtesting {coin}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary
    if all_results:
        print("\n" + "=" * 60)
        print("📈 BACKTEST SUMMARY")
        print("=" * 60)
        
        total_return_sum = sum(r['total_return'] for r in all_results.values())
        total_dollar_return = sum(r['total_return_dollars'] for r in all_results.values())
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values() if not np.isnan(r['sharpe_ratio'])])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
        total_trades = sum(r['total_trades'] for r in all_results.values())
        
        print(f"Coins Tested: {len(all_results)}")
        print(f"Average Return: {total_return_sum / len(all_results):.2f}%")
        print(f"Total Dollar Return: ${total_dollar_return:+.2f}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Win Rate: {avg_win_rate:.1f}%")
        print(f"Total Trades: {total_trades}")
        
        # Best performing coin
        best_coin = max(all_results.keys(), key=lambda k: all_results[k]['total_return'])
        print(f"Best Performer: {best_coin} ({all_results[best_coin]['total_return']:.2f}%)")
        
        # Worst performing coin
        worst_coin = min(all_results.keys(), key=lambda k: all_results[k]['total_return'])
        print(f"Worst Performer: {worst_coin} ({all_results[worst_coin]['total_return']:.2f}%)")
        
    else:
        print("❌ No successful backtests completed")
    
    return all_results


def create_detailed_report(results: Dict, days: int) -> str:
    """Create detailed backtest report."""
    if not results:
        return "No backtest results to report."
    
    report = f"""
🤖 PROFITBOT VECTORBT BACKTEST REPORT
=====================================
Test Period: {days} days
Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INDIVIDUAL COIN PERFORMANCE:
"""
    
    for coin, data in results.items():
        report += f"""
{coin}:
  - Total Return: {data['total_return']:.2f}%
  - Dollar P&L: ${data['total_return_dollars']:+.2f}
  - Final Value: ${data['final_value']:,.2f}
  - Sharpe Ratio: {data['sharpe_ratio']:.2f}
  - Max Drawdown: {data['max_drawdown']:.2f}%
  - Win Rate: {data['win_rate']:.1f}%
  - Total Trades: {data['total_trades']}
  - Avg Trade Duration: {data['avg_trade_duration']:.1f} hours
"""
    
    # Portfolio-wide stats
    total_return = sum(r['total_return'] for r in results.values())
    total_dollar = sum(r['total_return_dollars'] for r in results.values())
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values() if not np.isnan(r['sharpe_ratio'])])
    
    report += f"""
PORTFOLIO SUMMARY:
  - Combined Return: {total_return:.2f}%
  - Combined Dollar P&L: ${total_dollar:+.2f}
  - Average Sharpe Ratio: {avg_sharpe:.2f}
  - Coins Tested: {len(results)}
  - Total Trades: {sum(r['total_trades'] for r in results.values())}

STRATEGY ANALYSIS:
  - Momentum Threshold: 0.1% (very sensitive)
  - Take Profit: 1.0%
  - Stop Loss: 0.5% 
  - Max Hold Time: 15 hours
  - Volume Filter: 1.2x average
  - RSI Oversold/Overbought signals included
"""
    
    return report


def main():
    """Main backtesting function."""
    print("🤖 ProfitBot VectorBT Backtesting Tool")
    print("=" * 50)
    
    # Configuration
    coins = ['BTC', 'ETH', 'SOL', 'XRP']
    days = 30
    initial_balance = 1000.0
    
    # Run backtest
    results = run_vectorbt_backtest(coins, days, initial_balance)
    
    # Generate and display report
    report = create_detailed_report(results, days)
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"data/backtest_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📊 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")


if __name__ == "__main__":
    main()
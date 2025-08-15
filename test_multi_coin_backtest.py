#!/usr/bin/env python3
"""
Multi-Coin Strategy Backtesting - Test individual coin strategies.
"""
import vectorbt as vbt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Tuple

from core.config_loader import load_config_with_env
from core.multi_coin_strategy import BTCStrategy, ETHStrategy, SOLStrategy, XRPStrategy


def fetch_crypto_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch historical cryptocurrency data."""
    try:
        symbol_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'SOL': 'SOL-USD',
            'XRP': 'XRP-USD'
        }
        
        ticker = symbol_map.get(symbol, f"{symbol}-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            progress=False,
            auto_adjust=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        if 'Adj Close' in data.columns:
            data = data.drop('Adj Close', axis=1)
        
        data = data.ffill()
        
        print(f"✅ Fetched {len(data)} hours of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def simulate_coin_strategy(data: pd.DataFrame, strategy_class, coin: str, config: Dict) -> Tuple[pd.Series, pd.Series]:
    """Simulate a specific coin strategy."""
    if data.empty:
        return pd.Series(), pd.Series()
    
    prices = data['Close']
    volume = data['Volume']
    
    # Initialize strategy
    strategy = strategy_class(config)
    
    entries = pd.Series(False, index=prices.index)
    exits = pd.Series(False, index=prices.index)
    
    in_position = False
    entry_price = 0
    periods_held = 0
    
    for i, (timestamp, price) in enumerate(prices.items()):
        vol = volume.iloc[i] if not pd.isna(volume.iloc[i]) else 1000000
        
        # Update strategy indicators
        strategy.update_indicators(price, vol)
        
        if in_position:
            periods_held += 1
            strategy.position['periods_held'] = periods_held
            
            # Check exit conditions
            exit_reason = strategy.check_exit_conditions(price)
            if exit_reason:
                exits.iloc[i] = True
                in_position = False
                periods_held = 0
                print(f"  Exit at ${price:.2f}: {exit_reason}")
        else:
            # Check entry conditions
            if strategy.check_volume_filter():
                signal_strength = strategy.get_signal_strength(price)
                
                if signal_strength > strategy.momentum_threshold:
                    entries.iloc[i] = True
                    in_position = True
                    entry_price = price
                    strategy.position = {
                        'entry_price': price,
                        'quantity': 1.0,  # Normalized
                        'periods_held': 0,
                        'position_type': 'LONG'
                    }
                    print(f"  Entry at ${price:.2f}: strength={signal_strength:.4f}")
    
    return entries, exits


def run_multi_coin_backtest():
    """Run backtest for all coin-specific strategies."""
    print("🚀 MULTI-COIN STRATEGY BACKTEST")
    print("=" * 60)
    
    config = load_config_with_env('config.yaml')
    
    strategies = {
        'BTC': BTCStrategy,
        'ETH': ETHStrategy,
        'SOL': SOLStrategy,
        'XRP': XRPStrategy
    }
    
    results = {}
    
    for coin, strategy_class in strategies.items():
        print(f"\n📊 Testing {coin} with {strategy_class.__name__}...")
        
        try:
            # Fetch data
            data = fetch_crypto_data(coin, days=30)
            if data.empty:
                continue
            
            # Simulate strategy
            entries, exits = simulate_coin_strategy(data, strategy_class, coin, config)
            
            if entries.sum() == 0:
                print(f"  No trades generated for {coin}")
                continue
            
            # Run VectorBT portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries,
                exits,
                init_cash=1000.0,
                fees=0.001,
                freq='1h'
            )
            
            # Calculate metrics
            stats = portfolio.stats()
            trades = portfolio.trades.records_readable
            
            results[coin] = {
                'total_return': stats['Total Return [%]'],
                'dollar_return': portfolio.value().iloc[-1] - 1000.0,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max Drawdown [%]'],
                'win_rate': stats['Win Rate [%]'] if 'Win Rate [%]' in stats else 0,
                'total_trades': len(trades) if trades is not None else 0,
                'final_value': portfolio.value().iloc[-1],
                'strategy_name': strategy_class.__name__
            }
            
            print(f"  ✅ Results:")
            print(f"     Return: {results[coin]['total_return']:.2f}%")
            print(f"     Dollar P&L: ${results[coin]['dollar_return']:+.2f}")
            print(f"     Sharpe: {results[coin]['sharpe_ratio']:.2f}")
            print(f"     Max DD: {results[coin]['max_drawdown']:.2f}%")
            print(f"     Win Rate: {results[coin]['win_rate']:.1f}%")
            print(f"     Trades: {results[coin]['total_trades']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("📈 MULTI-COIN STRATEGY SUMMARY")
        print("=" * 60)
        
        total_return = sum(r['total_return'] for r in results.values())
        total_dollar = sum(r['dollar_return'] for r in results.values())
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values() if not np.isnan(r['sharpe_ratio'])])
        total_trades = sum(r['total_trades'] for r in results.values())
        
        print(f"Combined Return: {total_return:.2f}%")
        print(f"Combined Dollar P&L: ${total_dollar:+.2f}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Strategies Tested: {len(results)}")
        
        # Best and worst performers
        if len(results) > 1:
            best_coin = max(results.keys(), key=lambda k: results[k]['total_return'])
            worst_coin = min(results.keys(), key=lambda k: results[k]['total_return'])
            
            print(f"\n🏆 Best: {best_coin} ({results[best_coin]['total_return']:+.2f}%)")
            print(f"🔻 Worst: {worst_coin} ({results[worst_coin]['total_return']:+.2f}%)")
        
        # Strategy insights
        print(f"\n💡 STRATEGY INSIGHTS:")
        for coin, data in results.items():
            strategy_name = data['strategy_name']
            return_pct = data['total_return']
            trades = data['total_trades']
            
            if return_pct > 2:
                performance = "🟢 Excellent"
            elif return_pct > 0:
                performance = "🟡 Profitable"
            else:
                performance = "🔴 Needs work"
            
            print(f"  {coin} ({strategy_name}): {performance} - {return_pct:+.2f}% ({trades} trades)")
    
    return results


def compare_with_single_strategy():
    """Compare multi-coin vs single strategy performance."""
    print("\n🔄 COMPARISON: Multi-Coin vs Single Strategy")
    print("=" * 60)
    
    # Import and test single strategy for comparison
    from vectorbt_backtest import run_vectorbt_backtest
    
    print("Testing single strategy on all coins...")
    single_results = run_vectorbt_backtest(['BTC', 'ETH', 'SOL', 'XRP'], days=30, initial_balance=1000.0)
    
    print("\nTesting multi-coin strategies...")
    multi_results = run_multi_coin_backtest()
    
    if single_results and multi_results:
        print("\n📊 PERFORMANCE COMPARISON:")
        print("-" * 40)
        
        single_total = sum(r['total_return'] for r in single_results.values())
        multi_total = sum(r['total_return'] for r in multi_results.values())
        
        single_trades = sum(r['total_trades'] for r in single_results.values())
        multi_trades = sum(r['total_trades'] for r in multi_results.values())
        
        print(f"Single Strategy Total Return: {single_total:+.2f}%")
        print(f"Multi-Coin Strategy Return:   {multi_total:+.2f}%")
        print(f"Improvement: {multi_total - single_total:+.2f}%")
        print(f"")
        print(f"Single Strategy Trades: {single_trades}")
        print(f"Multi-Coin Strategy Trades: {multi_trades}")
        print(f"Trade Difference: {multi_trades - single_trades:+d}")
        
        if multi_total > single_total:
            print(f"\n🎉 Multi-coin strategies outperformed by {multi_total - single_total:.2f}%!")
        else:
            print(f"\n📝 Single strategy performed better by {single_total - multi_total:.2f}%")


if __name__ == "__main__":
    print("🤖 Multi-Coin Strategy Backtesting Tool")
    print("=" * 50)
    
    # Run multi-coin strategy backtest
    multi_results = run_multi_coin_backtest()
    
    # Optional: Compare with single strategy
    # compare_with_single_strategy()
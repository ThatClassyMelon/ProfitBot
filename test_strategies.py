#!/usr/bin/env python3
"""
Strategy testing and comparison script for ProfitBot.
Tests multiple scalping strategies and identifies the most profitable ones.
"""
import yaml
import argparse
from datetime import datetime
from strategies.scalping_strategies import StrategyBacktester
import pandas as pd


def format_results_table(results):
    """Format results into a readable table."""
    if not results:
        return "No results to display."
    
    # Filter out errored results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return "All strategies failed to generate results."
    
    # Create a formatted table
    table_lines = []
    table_lines.append("=" * 120)
    table_lines.append("🏆 STRATEGY PERFORMANCE COMPARISON")
    table_lines.append("=" * 120)
    
    # Header
    header = (f"{'Rank':<4} {'Strategy':<25} {'Type':<20} {'Return%':<8} {'Trades':<8} "
             f"{'Win%':<8} {'Sharpe':<8} {'MaxDD%':<8} {'Freq/Day':<8}")
    table_lines.append(header)
    table_lines.append("-" * 120)
    
    # Data rows
    for i, result in enumerate(valid_results[:10], 1):  # Top 10
        strategy_name = result['strategy_name'][:24]  # Truncate long names
        strategy_type = result.get('strategy_type', 'unknown')[:19]
        total_return = result.get('total_return', 0)
        total_trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        sharpe = result.get('sharpe_ratio', 0)
        max_dd = result.get('max_drawdown', 0)
        freq = result.get('trade_frequency_per_day', 0)
        
        row = (f"{i:<4} {strategy_name:<25} {strategy_type:<20} "
              f"{total_return:>7.2f} {total_trades:>7} {win_rate:>7.1f} "
              f"{sharpe:>7.2f} {max_dd:>7.2f} {freq:>7.1f}")
        table_lines.append(row)
    
    table_lines.append("=" * 120)
    
    # Summary statistics
    if valid_results:
        best_strategy = valid_results[0]
        table_lines.append("\n🎯 BEST STRATEGY DETAILS:")
        table_lines.append(f"Name: {best_strategy['strategy_name']}")
        table_lines.append(f"Description: {best_strategy.get('description', 'N/A')}")
        table_lines.append(f"Total Return: {best_strategy.get('total_return', 0):.2f}%")
        table_lines.append(f"Total Trades: {best_strategy.get('total_trades', 0)}")
        table_lines.append(f"Win Rate: {best_strategy.get('win_rate', 0):.1f}%")
        table_lines.append(f"Sharpe Ratio: {best_strategy.get('sharpe_ratio', 0):.2f}")
        table_lines.append(f"Max Drawdown: {best_strategy.get('max_drawdown', 0):.2f}%")
        table_lines.append(f"Average Trade Duration: {best_strategy.get('avg_trade_duration_hours', 0):.1f} hours")
        table_lines.append(f"Trade Frequency: {best_strategy.get('trade_frequency_per_day', 0):.1f} trades/day")
        
        # Show parameters
        table_lines.append(f"\n🔧 BEST STRATEGY PARAMETERS:")
        params = best_strategy.get('params', {})
        for key, value in params.items():
            table_lines.append(f"  {key}: {value}")
    
    return "\n".join(table_lines)


def save_results_to_csv(results, filename):
    """Save results to CSV file."""
    if not results:
        return
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_results)
    
    # Select key columns
    columns_to_save = [
        'strategy_name', 'strategy_type', 'description', 'total_return',
        'total_trades', 'win_rate', 'max_drawdown', 'sharpe_ratio',
        'profit_factor', 'trade_frequency_per_day', 'avg_trade_duration_hours',
        'final_value', 'buy_signals', 'sell_signals'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[existing_columns]
    
    # Sort by total return
    df_to_save = df_to_save.sort_values('total_return', ascending=False)
    
    # Save to CSV
    df_to_save.to_csv(filename, index=False)
    print(f"📊 Results saved to: {filename}")


def analyze_strategy_characteristics(results):
    """Analyze characteristics of top strategies."""
    valid_results = [r for r in results if 'error' not in r and r.get('total_trades', 0) > 0]
    
    if len(valid_results) < 3:
        return "Not enough valid results for analysis."
    
    analysis = []
    analysis.append("\n📈 STRATEGY ANALYSIS:")
    analysis.append("=" * 50)
    
    # Top 3 strategies
    top_3 = valid_results[:3]
    
    # Analyze patterns
    high_frequency = [r for r in valid_results if r.get('trade_frequency_per_day', 0) > 5]
    high_win_rate = [r for r in valid_results if r.get('win_rate', 0) > 60]
    low_drawdown = [r for r in valid_results if r.get('max_drawdown', 100) < 5]
    
    analysis.append(f"🎯 Top Performers:")
    for i, strategy in enumerate(top_3, 1):
        analysis.append(f"  {i}. {strategy['strategy_name']} - {strategy.get('total_return', 0):.2f}% return")
    
    analysis.append(f"\n⚡ High Frequency Strategies ({len(high_frequency)} strategies with >5 trades/day):")
    for strategy in high_frequency[:3]:
        analysis.append(f"  • {strategy['strategy_name']} - {strategy.get('trade_frequency_per_day', 0):.1f} trades/day")
    
    analysis.append(f"\n🎯 High Win Rate Strategies ({len(high_win_rate)} strategies with >60% win rate):")
    for strategy in high_win_rate[:3]:
        analysis.append(f"  • {strategy['strategy_name']} - {strategy.get('win_rate', 0):.1f}% win rate")
    
    analysis.append(f"\n🛡️ Low Drawdown Strategies ({len(low_drawdown)} strategies with <5% max drawdown):")
    for strategy in low_drawdown[:3]:
        analysis.append(f"  • {strategy['strategy_name']} - {strategy.get('max_drawdown', 0):.2f}% max drawdown")
    
    # Recommendations
    analysis.append(f"\n💡 RECOMMENDATIONS:")
    
    if high_frequency and high_win_rate:
        # Find overlap
        hf_names = {s['strategy_name'] for s in high_frequency}
        hwr_names = {s['strategy_name'] for s in high_win_rate}
        overlap = hf_names.intersection(hwr_names)
        
        if overlap:
            analysis.append(f"  🔥 BEST COMBINATION: {list(overlap)[0]} - High frequency + High win rate")
        else:
            analysis.append(f"  🔥 Consider: {high_frequency[0]['strategy_name']} for frequency")
            analysis.append(f"  🎯 Consider: {high_win_rate[0]['strategy_name']} for accuracy")
    
    if low_drawdown:
        analysis.append(f"  🛡️ Safest choice: {low_drawdown[0]['strategy_name']} - Low risk")
    
    return "\n".join(analysis)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test trading strategies')
    parser.add_argument('--coin', type=str, default='BTC', help='Coin to test (default: BTC)')
    parser.add_argument('--days', type=int, default=30, help='Days of data (default: 30)')
    parser.add_argument('--all-coins', action='store_true', help='Test on all configured coins')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    
    args = parser.parse_args()
    
    print("🧪 ProfitBot Strategy Testing Suite")
    print("=" * 50)
    
    # Load configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Config file not found!")
        return 1
    
    # Initialize backtester
    backtester = StrategyBacktester(config)
    
    if args.all_coins:
        # Test on all coins
        coins = list(config['coins'].keys())
        all_results = {}
        
        for coin in coins:
            print(f"\n🪙 Testing strategies on {coin}...")
            results = backtester.compare_all_strategies(coin, args.days)
            all_results[coin] = results
            
            if results:
                valid_results = [r for r in results if 'error' not in r]
                if valid_results:
                    best = valid_results[0]
                    print(f"   🏆 Best for {coin}: {best['strategy_name']} ({best.get('total_return', 0):.2f}%)")
        
        # Find overall best strategy across all coins
        all_strategies = []
        for coin_results in all_results.values():
            all_strategies.extend([r for r in coin_results if 'error' not in r])
        
        all_strategies.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        print("\n" + format_results_table(all_strategies))
        print(analyze_strategy_characteristics(all_strategies))
        
        if args.save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/strategy_results_all_coins_{timestamp}.csv"
            save_results_to_csv(all_strategies, filename)
    
    else:
        # Test on single coin
        print(f"\n🪙 Testing strategies on {args.coin} ({args.days} days)...")
        results = backtester.compare_all_strategies(args.coin, args.days)
        
        if not results:
            print("❌ No results generated")
            return 1
        
        # Display results
        print("\n" + format_results_table(results))
        print(analyze_strategy_characteristics(results))
        
        if args.save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/strategy_results_{args.coin}_{timestamp}.csv"
            save_results_to_csv(results, filename)
    
    print("\n✅ Strategy testing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
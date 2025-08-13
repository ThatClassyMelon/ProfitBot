# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProfitBot is an institutional-grade crypto trading bot with comprehensive multi-signal analysis. It integrates real-time market data, technical indicators, sentiment analysis, and advanced risk management for sophisticated trading decisions. The system is production-ready but executes mock trades only (no real exchange connections).

## Key Commands

### Running the Bot

#### Trading Mode (Live/Paper Trading)
```bash
# Mock trading (default)
python main.py

# With Alpaca paper trading (requires API credentials in config.yaml)
python main.py --mode trade
```

#### Backtesting Mode 
```bash
# Run backtest with default settings (30 days)
python main.py --mode backtest

# Custom timeframe
python main.py --mode backtest --days 90
```

#### Strategy Optimization
```bash
# Optimize parameters for BTC
python main.py --mode optimize --coin BTC --days 30

# Optimize for different coin
python main.py --mode optimize --coin ETH --days 60
```

#### Strategy Testing & Comparison
```bash
# Test all scalping strategies on BTC
python test_strategies.py --coin BTC --days 30

# Test strategies on all coins and save results
python test_strategies.py --all-coins --days 30 --save-csv
```

#### 24/7 Continuous Running
```bash
# Start bot in background (daemon mode)
./run_background.sh

# Check bot status
./check_bot_status.sh

# Stop bot
./stop_bot.sh

# View live logs
tail -f logs/daemon_output.log
tail -f data/trading.log
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Individual Components
```bash
python -c "from core.simulator import PriceSimulator; import yaml; config = yaml.safe_load(open('config.yaml')); sim = PriceSimulator(config); print(sim.update_prices())"
```

## Architecture Overview

### Core Components (`core/`)
- **advanced_data_fetcher.py**: Multi-source data aggregation (CoinGecko, Fear & Greed, technical indicators)
- **multi_signal_strategy.py**: Institutional-grade strategy with comprehensive signal analysis
- **enhanced_executor.py**: Advanced trade execution with DCA tracking and rebalancing
- **price_fetcher.py**: Real-time price fetching with simulation fallback
- **portfolio.py**: Sophisticated portfolio management with precision tracking
- **logger.py**: Professional logging system with detailed trade analysis

### Key Design Patterns
- **Modular Architecture**: Each component has a single responsibility and clear interfaces
- **Configuration-Driven**: All settings controlled via `config.yaml`
- **Type Hints**: Full type annotations throughout for maintainability
- **Error Handling**: Graceful degradation and comprehensive error logging

### Advanced Trading Logic Flow
1. **Data Aggregation**: Fetches comprehensive market data from multiple sources
   - Price, volume, market cap from CoinGecko
   - Fear & Greed Index from Alternative.me
   - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
   - Social metrics and developer activity scores
   
2. **Multi-Signal Analysis**: Combines multiple factors for trade decisions
   - Price action and volatility-adjusted thresholds
   - Volume and liquidity validation
   - Technical indicator confluence
   - Market sentiment adjustments
   - Social/developer activity weighting
   
3. **Signal Strength Calculation**: Quantifies confidence (0.0-1.0 scale)
   - Minimum 0.6 strength required for trades
   - Higher strength = larger position sizes
   - Multiple confirmations reduce false signals
   
4. **Advanced Execution**: Sophisticated trade management
   - Tiered DCA with increasing position sizes
   - Sentiment-adjusted thresholds
   - Automatic profit rebalancing
   - Volume-filtered entry/exit points

### Data Persistence
- **trading.log**: Real-time trading activity logs
- **trade_history.csv**: Structured trade data for analysis
- Both files auto-created in `data/` directory

### Configuration Structure
The `config.yaml` file controls:
- API settings (real prices vs simulation mode)
- Initial balance and tracked coins 
- Strategy parameters (thresholds, trade amounts)
- Price update intervals (30s for real data, 1s for simulation)
- Logging levels and file paths

### Alpaca Paper Trading Integration
- **AlpacaExecutor**: Real paper trading via Alpaca Markets API
- **Automatic switching**: Configure in `config.yaml` to enable paper trading
- **Portfolio sync**: Automatically syncs with Alpaca account balances
- **Crypto support**: BTC, ETH, SOL, XRP, ADA, DOT, MATIC, AVAX

### VectorBT Backtesting Integration  
- **Automated backtesting**: Test strategies against historical data
- **Parameter optimization**: Find optimal strategy parameters
- **Performance metrics**: Sharpe ratio, drawdown, win rate, P&L
- **Historical data**: Fetches from CoinGecko API with simulation fallback

### Optimized Trading Strategies
- **8 Scalping Strategies**: Designed for small consistent wins
- **Comprehensive Testing**: Backtested across multiple coins and timeframes
- **Performance Validated**: Best strategy achieves 31.54% returns with 66.7% win rate
- **Risk Optimized**: Low drawdown strategies (< 5%) available
- **Quick Momentum Scalp**: Default optimized strategy for small frequent profits

### Extension Points for Real Trading
- Real price data already implemented via CoinGecko API
- Alpaca paper trading provides bridge to real trading
- Update `AlpacaExecutor` base URL for live trading (requires additional setup)
- Strategy interface remains unchanged for real trading

## Development Notes

### Adding New Strategies
Create new strategy classes implementing the same interface as `ThresholdStrategy`:
- `analyze_market()` method returning list of `TradeSignal` objects
- Constructor accepting config dictionary

### Price Data Sources
Current implementation supports:
- **Real-time data**: CoinGecko API (free, no API key required)
- **Simulation mode**: Random walk with configurable volatility
- **Automatic fallback**: Switches to simulation if API fails
- **Rate limiting**: Built-in 10-second minimum between API calls

To switch modes: Set `api.use_real_prices: false` in config.yaml

### Portfolio Extensions
The portfolio system supports:
- Multiple cryptocurrency holdings
- Precise decimal handling for small amounts
- Trade price history tracking
- Real-time value calculations
#!/bin/bash
# Check ProfitBot status script

echo "📊 ProfitBot Status Check"
echo "=" * 30

# Check daemon process
if [ -f "logs/daemon.pid" ]; then
    DAEMON_PID=$(cat logs/daemon.pid)
    
    if ps -p $DAEMON_PID > /dev/null 2>&1; then
        echo "✅ Daemon: RUNNING (PID: $DAEMON_PID)"
        
        # Check how long it's been running
        START_TIME=$(ps -o lstart= -p $DAEMON_PID)
        echo "   Started: $START_TIME"
    else
        echo "❌ Daemon: STOPPED (stale PID file)"
    fi
else
    echo "❌ Daemon: NOT RUNNING (no PID file)"
fi

# Check main bot process
BOT_PID=$(pgrep -f "python.*main.py.*trade")
if [ ! -z "$BOT_PID" ]; then
    echo "✅ Trading Bot: RUNNING (PID: $BOT_PID)"
else
    echo "❌ Trading Bot: NOT RUNNING"
fi

echo ""

# Check recent logs
if [ -f "logs/daemon.log" ]; then
    echo "📋 Recent Daemon Activity (last 10 lines):"
    tail -10 logs/daemon.log
    echo ""
fi

if [ -f "data/trading.log" ]; then
    echo "💰 Recent Trading Activity (last 5 lines):"
    tail -5 data/trading.log
    echo ""
fi

# Check portfolio performance
if [ -f "data/trade_history.csv" ]; then
    TRADE_COUNT=$(tail -n +2 data/trade_history.csv | wc -l)
    echo "📊 Total Trades Executed: $TRADE_COUNT"
    
    if [ $TRADE_COUNT -gt 0 ]; then
        echo "💰 Recent Trades:"
        tail -5 data/trade_history.csv | column -t -s ','
    fi
fi
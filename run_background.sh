#!/bin/bash
# Background runner for ProfitBot daemon
# Usage: ./run_background.sh

echo "🤖 Starting ProfitBot in background..."

# Create logs directory
mkdir -p logs

# Start the daemon in background
nohup python3 run_bot_daemon.py > logs/daemon_output.log 2>&1 &

# Get the process ID
DAEMON_PID=$!

echo "✅ ProfitBot Daemon started in background"
echo "   Process ID: $DAEMON_PID"
echo "   Logs: logs/daemon_output.log"
echo "   Trading logs: data/trading.log"
echo ""
echo "📋 Management commands:"
echo "   Check status: ps aux | grep run_bot_daemon"
echo "   View logs: tail -f logs/daemon_output.log"
echo "   Stop daemon: kill $DAEMON_PID"
echo ""
echo "💡 The bot will now run 24/7 and automatically restart if it crashes."

# Save PID to file for easy management
echo $DAEMON_PID > logs/daemon.pid
echo "🔖 Process ID saved to logs/daemon.pid"
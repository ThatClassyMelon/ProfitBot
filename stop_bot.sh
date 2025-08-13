#!/bin/bash
# Stop ProfitBot daemon script

echo "🛑 Stopping ProfitBot Daemon..."

# Check if PID file exists
if [ -f "logs/daemon.pid" ]; then
    DAEMON_PID=$(cat logs/daemon.pid)
    
    # Check if process is running
    if ps -p $DAEMON_PID > /dev/null 2>&1; then
        echo "   Found daemon process: $DAEMON_PID"
        
        # Send SIGTERM for graceful shutdown
        kill $DAEMON_PID
        
        # Wait for process to stop
        echo "   Waiting for graceful shutdown..."
        sleep 5
        
        # Check if still running
        if ps -p $DAEMON_PID > /dev/null 2>&1; then
            echo "   Force killing daemon..."
            kill -9 $DAEMON_PID
        fi
        
        echo "✅ ProfitBot Daemon stopped"
        rm -f logs/daemon.pid
    else
        echo "   Daemon not running (PID $DAEMON_PID not found)"
        rm -f logs/daemon.pid
    fi
else
    echo "   No daemon.pid file found"
fi

# Also kill any remaining bot processes
echo "🧹 Cleaning up any remaining ProfitBot processes..."
pkill -f "python.*main.py.*trade" 2>/dev/null || true
pkill -f "run_bot_daemon.py" 2>/dev/null || true

echo "✅ All ProfitBot processes stopped"
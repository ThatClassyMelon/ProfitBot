#!/usr/bin/env python3
"""
Railway-optimized main script for ProfitBot.
Simplified version without daemon wrapper for better Railway compatibility.
"""
import os
import sys
import time
import signal

# Add error handling for Railway environment
def handle_railway_exit():
    """Handle Railway container shutdown gracefully."""
    print("🛑 Railway container shutdown detected")
    sys.exit(0)

# Set up signal handlers for Railway
signal.signal(signal.SIGTERM, lambda s, f: handle_railway_exit())
signal.signal(signal.SIGINT, lambda s, f: handle_railway_exit())

if __name__ == "__main__":
    try:
        print("🚂 Starting ProfitBot on Railway...")
        print("🔑 Checking environment variables...")
        
        # Check critical environment variables
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing environment variables: {missing_vars}")
            print("💡 Set these in Railway Variables tab")
            sys.exit(1)
        
        print("✅ Environment variables found")
        print("🤖 Importing ProfitBot...")
        
        from main import ProfitBot
        
        print("🚀 Starting trading bot...")
        bot = ProfitBot()
        bot.start()
        
    except KeyboardInterrupt:
        print("⌨️  Keyboard interrupt received")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
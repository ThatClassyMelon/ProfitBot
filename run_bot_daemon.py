#!/usr/bin/env python3
"""
Daemon runner for ProfitBot with automatic restart capabilities.
Keeps the bot running 24/7 with error recovery and logging.
"""
import os
import sys
import time
import signal
import subprocess
import logging
from datetime import datetime
from pathlib import Path


class BotDaemon:
    """Daemon process manager for ProfitBot."""
    
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.max_restarts_per_hour = 10
        self.restart_times = []
        self.running = True
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🤖 ProfitBot Daemon Started")
    
    def setup_logging(self):
        """Setup logging for the daemon."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'daemon.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('BotDaemon')
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down daemon...")
        self.running = False
        if self.process:
            self.stop_bot()
    
    def can_restart(self):
        """Check if bot can be restarted based on rate limiting."""
        now = time.time()
        # Remove restart times older than 1 hour
        self.restart_times = [t for t in self.restart_times if now - t < 3600]
        
        if len(self.restart_times) >= self.max_restarts_per_hour:
            self.logger.error(f"❌ Too many restarts ({len(self.restart_times)}) in the last hour. Waiting...")
            return False
        
        return True
    
    def start_bot(self):
        """Start the ProfitBot process."""
        try:
            self.logger.info("🚀 Starting ProfitBot...")
            
            # Start the bot process
            self.process = subprocess.Popen(
                [sys.executable, "main.py", "--mode", "trade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            self.restart_count += 1
            self.restart_times.append(time.time())
            
            self.logger.info(f"✅ ProfitBot started (PID: {self.process.pid}, Restart #{self.restart_count})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start ProfitBot: {e}")
            return False
    
    def stop_bot(self):
        """Stop the ProfitBot process gracefully."""
        if self.process:
            self.logger.info("🛑 Stopping ProfitBot...")
            
            try:
                # Send SIGINT for graceful shutdown
                self.process.send_signal(signal.SIGINT)
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                    self.logger.info("✅ ProfitBot stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.logger.warning("⚠️ Force killing ProfitBot...")
                    self.process.kill()
                    self.process.wait()
                    self.logger.info("✅ ProfitBot force stopped")
                    
            except Exception as e:
                self.logger.error(f"❌ Error stopping ProfitBot: {e}")
            
            finally:
                self.process = None
    
    def monitor_bot(self):
        """Monitor the bot process and restart if needed."""
        if not self.process:
            return False
        
        # Check if process is still running
        poll_result = self.process.poll()
        
        if poll_result is not None:
            # Process has terminated
            if poll_result == 0:
                self.logger.info("✅ ProfitBot exited normally")
            else:
                self.logger.error(f"❌ ProfitBot crashed with exit code: {poll_result}")
                
                # Log stderr if available
                try:
                    stderr_output = self.process.stderr.read()
                    if stderr_output:
                        self.logger.error(f"Error output: {stderr_output}")
                except:
                    pass
            
            self.process = None
            return False
        
        return True
    
    def run(self):
        """Main daemon loop."""
        self.logger.info("🔄 Starting daemon monitoring loop...")
        
        while self.running:
            try:
                # Start bot if not running
                if not self.process:
                    if self.can_restart():
                        if not self.start_bot():
                            self.logger.error("❌ Failed to start bot, waiting 60 seconds...")
                            time.sleep(60)
                            continue
                    else:
                        self.logger.info("💤 Waiting 10 minutes before next restart attempt...")
                        time.sleep(600)  # Wait 10 minutes
                        continue
                
                # Monitor bot
                if not self.monitor_bot():
                    if self.running:  # Only restart if we're not shutting down
                        self.logger.info("🔄 Bot stopped, will restart in 10 seconds...")
                        time.sleep(10)
                
                # Sleep briefly before next check
                time.sleep(5)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Daemon interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in daemon: {e}")
                time.sleep(30)  # Wait before retrying
        
        # Cleanup
        self.stop_bot()
        self.logger.info("🏁 ProfitBot Daemon Stopped")


def main():
    """Main entry point."""
    print("🤖 ProfitBot Daemon - 24/7 Trading Bot Manager")
    print("=" * 50)
    print("This will keep your trading bot running continuously.")
    print("Press Ctrl+C to stop the daemon and bot.")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Run this from the ProfitBot directory.")
        sys.exit(1)
    
    # Create and run daemon
    daemon = BotDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
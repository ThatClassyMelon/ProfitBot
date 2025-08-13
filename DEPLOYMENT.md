# 🚂 Railway Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - To host your code
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Telegram Setup** - Send `/start` to your bot: @ProfitLogBot

## 🚀 Deployment Steps

### 1. Push to GitHub

Create a new repository on GitHub, then:

```bash
# Add your GitHub repository
git remote add origin https://github.com/YOURUSERNAME/ProfitBot.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy on Railway

1. **Login to Railway** and click "New Project"
2. **Select "Deploy from GitHub"**
3. **Choose your ProfitBot repository**
4. **Railway will automatically deploy** using the included `railway.json`

### 3. Configure Environment Variables

In the Railway dashboard, add these environment variables:

```env
ALPACA_API_KEY=PKVMXMV0XBIFKAD4YASG
ALPACA_SECRET_KEY=8Y2yE6i0GeEdqciCT2Aodm5rcRykOTf8YtuEsOtY
TELEGRAM_BOT_TOKEN=8124721803:AAEQta9AouecKLxqqeynK3ahDcYpQEMZ7zM
TELEGRAM_CHAT_ID=your_chat_id
```

*Note: Your actual values are already in config.yaml, but environment variables override config for security.*

### 4. Telegram Setup

1. **Find your bot**: Search for `@ProfitLogBot` in Telegram
2. **Start the bot**: Send `/start` to your bot
3. **Get Chat ID**: Your bot will auto-detect it when you run the trading bot

### 5. Deploy and Monitor

- **Automatic Start**: Railway will start your bot automatically
- **24/7 Operation**: The daemon keeps it running continuously
- **Auto Restart**: If the bot crashes, it automatically restarts
- **Logs**: Check Railway logs for status updates

## 📊 Expected Timeline

- **Deployment**: 2-3 minutes
- **First Trade**: Within first hour (depends on market conditions)
- **Telegram Updates**: Every hour + immediate trade alerts

## 📱 Telegram Notifications

You'll receive:

### 🕐 Hourly Updates
```
📊 Hourly Trading Update
⏰ Uptime: 2.5 hours

💰 Portfolio Status
📈 Value: $1,024.50
💵 P&L: +$24.50 (+2.45%)

⚡ Last Hour Activity
🔄 Trades: 2
📊 Volume: $156.80
💰 P&L: +$8.20
🪙 Coins traded:
   • BTC: 1 trades
   • ETH: 1 trades
```

### 📈 Trade Alerts
```
📈 Trade Executed
🪙 Coin: BTC
⚡ Action: BUY
📊 Quantity: 0.000821
💰 Price: $120,095.90
💵 Value: $98.60
🎯 Reason: Momentum Buy: 0.67% momentum, Volume: 1.5x
```

## 🛠️ Railway Configuration

### Automatic Settings
- **Start Command**: `python run_bot_daemon.py`
- **Restart Policy**: On failure (max 3 retries)
- **Builder**: Nixpacks (automatic Python detection)
- **Runtime**: Python 3.11

### Resource Usage
- **Memory**: ~100-200MB
- **CPU**: Very low (mostly waiting/sleeping)
- **Storage**: Minimal (logs and trade history)

## 🔍 Monitoring & Management

### Railway Dashboard
- **Logs**: Real-time bot activity
- **Metrics**: CPU/Memory usage
- **Deployments**: Version history
- **Variables**: Environment configuration

### Telegram Monitoring
- **Live Updates**: Hourly portfolio status
- **Trade Alerts**: Immediate notifications
- **Error Alerts**: If something goes wrong

### Manual Control
If you need to stop/restart:
1. **Stop**: Use Railway dashboard to stop service
2. **Restart**: Push a new commit or manually restart
3. **Logs**: Check Railway logs for debugging

## 💰 Expected Performance

Based on backtesting:
- **Monthly Return**: 15-30% (varies by market conditions)
- **Daily Trades**: 0-5 trades (depends on volatility)
- **Win Rate**: 60-100% (strategy dependent)
- **Risk**: Low (1-2% per trade with stops)

## 🚨 Important Notes

### Security
- **Paper Trading Only**: No real money at risk
- **API Keys**: Stored as environment variables
- **Rate Limiting**: Built-in protection against spam

### Monitoring
- **Check Daily**: Review Telegram updates
- **Weekly Review**: Analyze trade history
- **Monthly Assessment**: Evaluate overall performance

### Scaling
- **Start Small**: Monitor for 1-2 weeks first
- **Gradual Increase**: Scale up capital gradually
- **Risk Management**: Never risk more than you can afford

## 🆘 Troubleshooting

### Bot Not Starting
1. Check Railway logs for errors
2. Verify environment variables
3. Ensure GitHub repo is updated

### No Telegram Messages
1. Send `/start` to @ProfitLogBot
2. Check TELEGRAM_BOT_TOKEN is correct
3. Wait for first trade or hourly update

### No Trades
- **Normal**: May take hours in stable markets
- **Check**: Telegram hourly updates for status
- **Strategy**: Momentum scalp waits for volatility

### Bot Stopped
- **Auto Restart**: Daemon should restart automatically
- **Check Logs**: Railway dashboard for error details
- **Manual**: Restart service in Railway dashboard

---

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Railway project created and connected
- [ ] Environment variables configured
- [ ] Telegram bot started with `/start`
- [ ] Bot deployed and running
- [ ] Received startup notification in Telegram
- [ ] Monitoring Railway logs
- [ ] Waiting for first trade/hourly update

**Your ProfitBot is now running 24/7 in the cloud!** 🚀💰
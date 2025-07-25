# 🚀 Professional Crypto Futures Trading Bot

**8+ Years Experience Scalping System** with advanced multi-factor analysis and professional risk management.

## ⚡ Key Features

- **ATR-Based Professional Position Sizing**
- **Multi-Timeframe Momentum Confirmation** (1m→5m→15m)
- **Volatility Regime Detection & Adaptation**
- **Volume Profile & Institutional Flow Analysis** 
- **Support/Resistance Confluence Trading**
- **Professional Risk Management** (2% max portfolio risk)
- **Market Microstructure Considerations**
- **Session-Weighted Confidence Scoring**

## 🔧 Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install python-binance pandas numpy python-dotenv
```

### 2. API Configuration
Create a `.env` file in the project root:
```bash
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Trading Configuration (optional)
LEVERAGE=125
TEST_MODE=False
```

### 3. Binance Account Setup
- ✅ Enable **Futures Trading** on your Binance account
- ✅ Create API keys with **Futures** permissions only
- ✅ Add your IP to the **whitelist** in API settings
- ✅ Ensure sufficient **USDT balance** for trading

### 4. Run the Bot
```bash
source venv/bin/activate
python main.py
```

## 🛡️ Security Features

- **Environment Variables** - API keys stored securely in `.env`
- **IP Whitelisting** - Restrict API access to your IP only
- **Permission Limits** - API keys limited to futures trading only
- **Git Protection** - `.env` file excluded from version control

## 📊 Professional Trading Logic

### Confidence Thresholds:
- **55%** - Overlap session (best liquidity)
- **70%** - Standard conditions
- **85%** - High volatility markets

### Risk Management:
- **2% Max Portfolio Risk** at any time
- **ATR-Based Position Sizing** for volatility adaptation
- **Dynamic TP/SL** based on market conditions
- **Correlation Filtering** to avoid overexposure

### Technical Analysis:
- **9 Advanced Indicators** with professional weighting
- **Multi-Timeframe Confirmation** across 3 timeframes
- **Volatility Regime Detection** (HIGH/MEDIUM/LOW)
- **Volume Profile Analysis** for institutional confirmation

## ⚠️ Risk Disclaimer

This is a professional trading system for experienced users. 
- **High leverage trading** involves significant risk
- **Past performance** does not guarantee future results
- **Only trade with capital** you can afford to lose
- **Test thoroughly** before live trading

## 🔗 Support

For technical issues or questions about the professional system, please check the code comments for detailed explanations of each component. 
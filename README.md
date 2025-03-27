# Trading-Hackathon
# AI-Powered Self-Learning Trading System  

## Introduction  

Welcome to our AI-powered self-learning trading system! This project leverages machine learning and deep learning techniques to analyze market trends, detect hidden patterns, and generate smart trading signals. Our goal is to enhance automated trading strategies by improving prediction accuracy, optimizing risk management, and ensuring adaptive decision-making in different market conditions.  

## Project Structure  

Our repository follows a structured approach with **five branches**:  

- **Main Branch:** The final, production-ready version of our codebase.  
- **User-Specific Branches:** Each team member has a dedicated branch for development, testing, and experimentation before merging improvements into the main branch.  

## Trading Strategy & Problem Statement  

Financial markets are unpredictable and chaotic due to external influences like economic events, regulations, and investor sentiment. Traditional trading strategies often struggle to identify hidden patterns, leading to missed opportunities and misinterpretation of market signals. Our system tackles these challenges by utilizing:  

- **Market Regime Detection** (using Hidden Markov Models) to identify bullish, bearish, and neutral phases.  
- **Deep Learning for Price Prediction** (using 1D CNN) to forecast future price movements based on historical data.  
- **Automated Trade Signal Generation** to execute trades based on intelligent predictions.  

## Process Flow  

1️⃣ **Data Processing:** We collect OHLCV data from Binance and other sources.  
2️⃣ **Market Regime Detection:** HMMs help detect hidden market states.  
3️⃣ **Feature Extraction:** We extract critical price movement patterns.  
4️⃣ **Price Prediction:** CNN models predict the next closing price.  
5️⃣ **Trade Signal Generation:** The model generates Buy/Sell signals.  

## Performance & Limitations  

To ensure an effective trading system, we set the following success criteria:  

- **Sharpe Ratio ≥ 1.8** (Ensuring risk-adjusted returns)  
- **Max Drawdown ≥ -40%** (Limiting downside risk)  
- **Trade Frequency ≥ 3%** (Ensuring active trading)  

However, our current model has a **Sharpe Ratio of only 0.53** and a **Maximum Drawdown of -0.71%**. The likely reasons are:  

- **HMM Model not fine-tuned**  
- **Insufficient data for model training**  

We aim to refine our approach by optimizing the HMM model parameters and improving our dataset.  

## Business & Future Growth  

We envision a **subscription-based model** where traders can access AI-driven trading signals with different pricing tiers. Additionally, our system can be expanded to support NFT trading and cross-chain strategies for **Ethereum, Solana, and BSC** networks.  

## Conclusion  

This project represents a step toward **intelligent, self-learning AI in trading**. With further enhancements and optimizations, we aim to improve its performance, minimize risks, and maximize returns for traders.  

🚀 **Stay tuned for updates as we continue to refine our system!** 🚀  

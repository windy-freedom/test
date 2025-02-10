from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

class TradingAgent:
    def __init__(self):
        self.cash = 10000  # Initial capital
        self.holdings = 0  # Current crypto holdings
        self.history = []  # Trading history
        
    def analyze_market(self, prices):
        # Simple moving average strategy
        short_window = 10
        long_window = 20
        
        if len(prices) < long_window:
            return 'HOLD'
            
        short_ma = prices[-short_window:].mean()
        long_ma = prices[-long_window:].mean()
        
        if short_ma > long_ma:
            return 'BUY'
        elif short_ma < long_ma:
            return 'SELL'
        return 'HOLD'
        
    def execute_trade(self, action, price, timestamp):
        if action == 'BUY' and self.cash > 0:
            amount = self.cash / price
            self.holdings += amount
            self.cash = 0
            self.history.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': price,
                'amount': amount,
                'cash': self.cash,
                'holdings_value': self.holdings * price
            })
        elif action == 'SELL' and self.holdings > 0:
            self.cash = self.holdings * price
            self.holdings = 0
            self.history.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'amount': self.holdings,
                'cash': self.cash,
                'holdings_value': 0
            })

def get_crypto_data():
    # Get Bitcoin-USD data
    btc = yf.download('BTC-USD', start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    return btc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate')
def simulate():
    # Get historical data
    data = get_crypto_data()
    
    # Initialize agent
    agent = TradingAgent()
    
    # Simulate trading
    prices = data['Close'].values
    dates = data.index.strftime('%Y-%m-%d').tolist()
    
    for i, price in enumerate(prices):
        action = agent.analyze_market(prices[:i+1])
        agent.execute_trade(action, price, dates[i])
    
    # Prepare simulation results
    portfolio_value = agent.cash + (agent.holdings * prices[-1])
    roi = ((portfolio_value - 10000) / 10000) * 100
    
    return jsonify({
        'dates': dates,
        'prices': prices.tolist(),
        'history': agent.history,
        'final_value': round(portfolio_value, 2),
        'roi': round(roi, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
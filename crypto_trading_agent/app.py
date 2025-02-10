from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

class TradingAgent:
    def __init__(self):
        self.cash = 100000  # 增加初始资金
        self.holdings = 0  # 当前持仓数量
        self.history = []  # 交易历史
        self.trade_count = 0  # 交易次数
        self.profitable_trades = 0  # 盈利交易次数
        
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
            amount = (self.cash * 0.95) / price  # 使用95%的资金进行交易
            self.holdings += amount
            prev_cash = self.cash
            self.cash = self.cash * 0.05  # 保留5%现金
            self.trade_count += 1
            
            trade_info = {
                'timestamp': timestamp,
                'action': 'BUY',
                'price': float(price),  # 确保价格是Python float类型
                'amount': float(amount),
                'cash': float(self.cash),
                'holdings_value': float(self.holdings * price),
                'total_value': float(self.cash + (self.holdings * price)),
                'profit': 0
            }
            self.history.append(trade_info)
            
        elif action == 'SELL' and self.holdings > 0:
            prev_value = self.holdings * price
            self.cash += prev_value
            profit = prev_value - (self.history[-1]['price'] * self.holdings if self.history else 0)
            self.holdings = 0
            self.trade_count += 1
            
            if profit > 0:
                self.profitable_trades += 1
                
            trade_info = {
                'timestamp': timestamp,
                'action': 'SELL',
                'price': float(price),
                'amount': float(self.holdings),
                'cash': float(self.cash),
                'holdings_value': 0,
                'total_value': float(self.cash),
                'profit': float(profit)
            }
            self.history.append(trade_info)

def get_crypto_data():
    # Get Bitcoin-USD data
    btc = yf.download('BTC-USD', start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    return btc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate')
def simulate():
    # 获取历史数据
    data = get_crypto_data()
    
    # 初始化交易代理
    agent = TradingAgent()
    initial_capital = agent.cash
    
    # 模拟交易
    prices = data['Close'].values
    dates = data.index.strftime('%Y-%m-%d').tolist()
    
    for i, price in enumerate(prices):
        action = agent.analyze_market(prices[:i+1])
        agent.execute_trade(action, float(price), dates[i])
    
    # 准备模拟结果
    current_price = float(prices[-1])
    portfolio_value = float(agent.cash + (agent.holdings * current_price))
    roi = ((portfolio_value - initial_capital) / initial_capital) * 100
    
    # 计算额外的统计数据
    win_rate = (agent.profitable_trades / agent.trade_count * 100) if agent.trade_count > 0 else 0
    max_profit = max([trade.get('profit', 0) for trade in agent.history]) if agent.history else 0
    max_drawdown = min([trade.get('total_value', initial_capital) - initial_capital for trade in agent.history]) if agent.history else 0
    
    return jsonify({
        'dates': dates,
        'prices': [float(p) for p in prices],
        'history': agent.history,
        'final_value': round(portfolio_value, 2),
        'roi': round(roi, 2),
        'stats': {
            'initial_capital': initial_capital,
            'total_trades': agent.trade_count,
            'profitable_trades': agent.profitable_trades,
            'win_rate': round(win_rate, 2),
            'max_profit': round(max_profit, 2),
            'max_drawdown': round(max_drawdown, 2),
            'current_holdings': round(agent.holdings, 6),
            'current_price': round(current_price, 2)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
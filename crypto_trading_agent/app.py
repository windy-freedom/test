from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import talib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

class TradingAgent:
    def __init__(self, symbol, initial_capital=100000):
        self.symbol = symbol
        self.cash = initial_capital
        self.holdings = 0
        self.history = []
        self.trade_count = 0
        self.profitable_trades = 0
        self.learning_rate = 0.1
        self.strategy_weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.3
        }
        self.market_insights = []
        
    def calculate_indicators(self, prices):
        prices_series = pd.Series(prices)
        
        # 趋势指标
        sma_short = talib.SMA(prices_series, timeperiod=10)
        sma_long = talib.SMA(prices_series, timeperiod=20)
        macd, macd_signal, _ = talib.MACD(prices_series)
        
        # 动量指标
        rsi = talib.RSI(prices_series, timeperiod=14)
        
        # 波动性指标
        upper, middle, lower = talib.BBANDS(prices_series, timeperiod=20)
        
        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'macd': macd,
            'macd_signal': macd_signal,
            'rsi': rsi,
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }
    
    def analyze_pattern(self, prices):
        prices_series = pd.Series(prices)
        patterns = {}
        
        # 识别常见K线模式
        patterns['doji'] = talib.CDLDOJI(prices_series, prices_series, prices_series, prices_series)
        patterns['engulfing'] = talib.CDLENGULFING(prices_series, prices_series, prices_series, prices_series)
        patterns['hammer'] = talib.CDLHAMMER(prices_series, prices_series, prices_series, prices_series)
        
        return patterns
        
    def analyze_market(self, prices):
        if len(prices) < 20:  # 需要足够的数据来计算指标
            return 'HOLD', "等待足够的历史数据"
            
        indicators = self.calculate_indicators(prices)
        patterns = self.analyze_pattern(prices)
        
        # 计算各个策略的信号
        trend_signal = self.analyze_trend(indicators)
        momentum_signal = self.analyze_momentum(indicators)
        volatility_signal = self.analyze_volatility(indicators)
        
        # 综合分析
        final_signal = (
            trend_signal * self.strategy_weights['trend'] +
            momentum_signal * self.strategy_weights['momentum'] +
            volatility_signal * self.strategy_weights['volatility']
        )
        
        # 生成市场洞察
        insight = self.generate_market_insight(indicators, patterns, final_signal)
        self.market_insights.append(insight)
        
        # 根据综合信号确定行动
        if final_signal > 0.5:
            return 'BUY', insight
        elif final_signal < -0.5:
            return 'SELL', insight
        return 'HOLD', insight
    
    def analyze_trend(self, indicators):
        trend_signal = 0
        
        # 移动平均线趋势分析
        if indicators['sma_short'].iloc[-1] > indicators['sma_long'].iloc[-1]:
            trend_signal += 0.5
        elif indicators['sma_short'].iloc[-1] < indicators['sma_long'].iloc[-1]:
            trend_signal -= 0.5
            
        # MACD分析
        if indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1]:
            trend_signal += 0.5
        elif indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1]:
            trend_signal -= 0.5
            
        return trend_signal
    
    def analyze_momentum(self, indicators):
        momentum_signal = 0
        
        # RSI分析
        rsi = indicators['rsi'].iloc[-1]
        if rsi > 70:
            momentum_signal -= 1
        elif rsi < 30:
            momentum_signal += 1
        
        return momentum_signal
    
    def analyze_volatility(self, indicators):
        volatility_signal = 0
        
        # 布林带分析
        current_price = indicators['bb_middle'].iloc[-1]
        if current_price < indicators['bb_lower'].iloc[-1]:
            volatility_signal += 1
        elif current_price > indicators['bb_upper'].iloc[-1]:
            volatility_signal -= 1
            
        return volatility_signal
    
    def generate_market_insight(self, indicators, patterns, final_signal):
        insight = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': [],
            'recommendation': '',
            'confidence': abs(final_signal)
        }
        
        # 趋势分析
        if indicators['sma_short'].iloc[-1] > indicators['sma_long'].iloc[-1]:
            insight['analysis'].append("上升趋势:短期均线位于长期均线上方")
        else:
            insight['analysis'].append("下降趋势:短期均线位于长期均线下方")
            
        # RSI分析
        rsi = indicators['rsi'].iloc[-1]
        if rsi > 70:
            insight['analysis'].append(f"RSI超买({rsi:.2f}):市场可能出现回调")
        elif rsi < 30:
            insight['analysis'].append(f"RSI超卖({rsi:.2f}):市场可能反弹")
            
        # 布林带分析
        price = indicators['bb_middle'].iloc[-1]
        if price > indicators['bb_upper'].iloc[-1]:
            insight['analysis'].append("价格突破布林带上轨:可能出现超买")
        elif price < indicators['bb_lower'].iloc[-1]:
            insight['analysis'].append("价格突破布林带下轨:可能出现超卖")
            
        # 设置建议
        if final_signal > 0.5:
            insight['recommendation'] = "建议买入"
        elif final_signal < -0.5:
            insight['recommendation'] = "建议卖出"
        else:
            insight['recommendation'] = "建议观望"
            
        return insight
        
    def execute_trade(self, action, price, timestamp):
        if action == 'BUY' and self.cash > 0:
            amount = (self.cash * 0.95) / price
            self.holdings += amount
            prev_cash = self.cash
            self.cash = self.cash * 0.05
            self.trade_count += 1
            
            trade_info = {
                'timestamp': timestamp,
                'action': 'BUY',
                'price': float(price),
                'amount': float(amount),
                'cash': float(self.cash),
                'holdings_value': float(self.holdings * price),
                'total_value': float(self.cash + (self.holdings * price)),
                'profit': 0,
                'insight': self.market_insights[-1] if self.market_insights else None
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
                self.adjust_strategy(True)
            else:
                self.adjust_strategy(False)
                
            trade_info = {
                'timestamp': timestamp,
                'action': 'SELL',
                'price': float(price),
                'amount': float(self.holdings),
                'cash': float(self.cash),
                'holdings_value': 0,
                'total_value': float(self.cash),
                'profit': float(profit),
                'insight': self.market_insights[-1] if self.market_insights else None
            }
            self.history.append(trade_info)
            
    def adjust_strategy(self, was_profitable):
        """根据交易结果调整策略权重"""
        adjustment = self.learning_rate if was_profitable else -self.learning_rate
        
        # 更新权重
        if was_profitable:
            # 增加表现好的策略的权重
            max_weight = max(self.strategy_weights.values())
            for k, v in self.strategy_weights.items():
                if v == max_weight:
                    self.strategy_weights[k] = min(1.0, v + adjustment)
        else:
            # 减少表现差的策略的权重
            min_weight = min(self.strategy_weights.values())
            for k, v in self.strategy_weights.items():
                if v == min_weight:
                    self.strategy_weights[k] = max(0.1, v - adjustment)
        
        # 归一化权重
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v/total for k, v in self.strategy_weights.items()}

def get_crypto_data(symbol):
    data = yf.download(symbol, start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    return data

def run_simulation(symbol):
    data = get_crypto_data(symbol)
    agent = TradingAgent(symbol)
    initial_capital = agent.cash
    
    prices = data['Close'].values
    dates = data.index.strftime('%Y-%m-%d').tolist()
    
    for i, price in enumerate(prices):
        action, insight = agent.analyze_market(prices[:i+1])
        agent.execute_trade(action, float(price), dates[i])
    
    current_price = float(prices[-1])
    portfolio_value = float(agent.cash + (agent.holdings * current_price))
    roi = ((portfolio_value - initial_capital) / initial_capital) * 100
    
    win_rate = (agent.profitable_trades / agent.trade_count * 100) if agent.trade_count > 0 else 0
    max_profit = max([trade.get('profit', 0) for trade in agent.history]) if agent.history else 0
    max_drawdown = min([trade.get('total_value', initial_capital) - initial_capital for trade in agent.history]) if agent.history else 0
    
    return {
        'symbol': symbol,
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
            'current_price': round(current_price, 2),
            'strategy_weights': agent.strategy_weights
        },
        'market_insights': agent.market_insights
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate')
def simulate():
    btc_results = run_simulation('BTC-USD')
    eth_results = run_simulation('ETH-USD')
    
    return jsonify({
        'btc': btc_results,
        'eth': eth_results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
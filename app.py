import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from flask import Flask, render_template, jsonify, request, current_app
from datetime import datetime, timedelta
import logging
import json
from niftystocks import ns
import time
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from flask_cors import CORS

#######################################################################################################################
#Description: This is a simple web application that uses a LSTM model to predict stock prices for the next 7, 15, or 30 days.
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################


#######################################################################################################################
#Dictionary: INDICES
#Keys: NIFTY50, SENSEX, BANKNIFTY
#Values: Dictionary containing the Yahoo Finance and NSE symbols for the index
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################

INDICES = {
    'NIFTY50': {
        'yf': '^NSEI',
        'ns': 'NIFTY 50',
        'nse': 'NIFTY 50'
    },
    'SENSEX': {
        'yf': '^BSESN',
        'ns': 'SENSEX',
        'nse': 'SENSEX'
    },
    'BANKNIFTY': {
        'yf': '^NSEBANK',
        'ns': 'NIFTY BANK',
        'nse': 'NIFTY BANK'
    },
    # 'NIFTYAUTO': {
    #     'yf': 'NIFTY-AUTO.NS',
    #     'ns': 'NIFTY AUTO',
    #     'nse': 'NIFTY AUTO'
    # },
    # 'NIFTYFINSERV': {
    #     'yf': 'NIFTY-FIN-SERVICE.NS',
    #     'ns': 'NIFTY FINANCIAL SERVICES',
    #     'nse': 'NIFTY FINANCIAL SERVICES'
    # },
    # 'NIFTYFMCG': {
    #     'yf': 'NIFTY-FMCG.NS',
    #     'ns': 'NIFTY FMCG',
    #     'nse': 'NIFTY FMCG'
    # },
    # 'NIFTYIT': {
    #     'yf': 'NIFTY-IT.NS',
    #     'ns': 'NIFTY IT',
    #     'nse': 'NIFTY IT'
    # },
    # 'NIFTYMETAL': {
    #     'yf': 'NIFTY-METAL.NS',
    #     'ns': 'NIFTY METAL',
    #     'nse': 'NIFTY METAL'
    # },
    # 'NIFTYPHARMA': {
    #     'yf': 'NIFTY-PHARMA.NS',
    #     'ns': 'NIFTY PHARMA',
    #     'nse': 'NIFTY PHARMA'
    # }
}

#######################################################################################################################
#Function: initialize_app
#Input: app
#Output: None
#Description: This function initializes the application state by checking the data sources
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

#######################################################################################################################
#Class: AttentionLayer
#Input: nn.Module
#Output: None
#Description: Enable the CORS for all the routes
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
CORS(app) 


#######################################################################################################################
#Class: AttentionLayer
#Input: nn.Module
#Output: None

#Description: Function to initialize the application state by checking the data sources
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def initialize_app(app):
    """Initialize application state"""
    try:
        with app.app_context():
            status = check_data_sources()
            logging.info(f"Initial data source status: {status.get_json()}")
    except Exception as e:
        logging.error(f"Application initialization failed: {str(e)}")
        raise

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
#######################################################################################################################
#Class: AttentionLayer
#Input: nn.Module
#Output: None
#Description: Function to calculate the attention weights and apply them to the hidden states
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended

#######################################################################################################################
#Class: EnhancedStockBiLSTM
#Input: nn.Module
#Output: None
#Description: Function to initialize the Enhanced Stock BiLSTM model
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################

class EnhancedStockBiLSTM(nn.Module):
#######################################################################################################################
#Function: __init__
#Input: input_size, hidden_size, num_layers, output_size
#Output: None
#Description: Function to initialize the Enhanced Stock BiLSTM model
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, output_size)

#######################################################################################################################
#Function: forward
#Input: x
#Output: out
#Description: Function to define the forward pass of the model
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
    def forward(self, x):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.norm1(out)
        
        attended = self.attention(out)
        attended = self.dropout(attended)
        
        out = self.fc1(attended)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        return out

#######################################################################################################################
#Class: StockDataset
#Input: Dataset
#Output: None
#Description: Custom PyTorch dataset for stock data
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#######################################################################################################################
#Class: DataFetchError
#Input: Exception
#Output: None
#Description: Custom exception for data fetching errors
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
class DataFetchError(Exception):
    """Custom exception for data fetching errors"""
    pass

#######################################################################################################################
#Function: calculate_rsi
#Input: prices, period
#Output: 100 - (100 / (1 + rs))
#Description: Function to calculate the Relative Strength Index (RSI) of a stock
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#######################################################################################################################
#Function: calculate_macd
#Input: prices, fast, slow
#Output: exp1 - exp2
#Description: Function to calculate the Moving Average Convergence Divergence (MACD) of a stock
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

#######################################################################################################################
#Function: prepare_data
#Input: data, sequence_length, prediction_days
#Output: X_train, X_val, y_train, y_val, scaler
#Description: Function to prepare the data for training the model
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def prepare_data(data, sequence_length=30, prediction_days=7):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length - prediction_days + 1):
            seq = scaled_data[i:i + sequence_length]
            target = scaled_data[i + sequence_length:i + sequence_length + prediction_days, 0]
            sequences.append(seq)
            targets.append(target)
        
        if not sequences or not targets:
            raise ValueError("No valid sequences could be created from the data")
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        split = int(len(sequences) * 0.8)
        X_train, X_val = sequences[:split], sequences[split:]
        y_train, y_val = targets[:split], targets[split:]
        
        return X_train, X_val, y_train, y_val, scaler
        
    except Exception as e:
        logging.error(f"Data preparation error: {str(e)}")
        raise

#######################################################################################################################
#Function: fetch_nse_direct
#Input: symbol, days
#Output: df
#Description: Function to fetch data directly from the NSE website
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def fetch_nse_direct(symbol, days=365):
    """Fetch data directly from NSE website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        symbol_clean = symbol.replace('^NSEI', 'NIFTY').replace('^NSEBANK', 'BANKNIFTY').replace('.NS', '')
        
        # NSE API endpoint
        base_url = "https://www.nseindia.com/api/historical/indices"
        params = {
            'symbol': symbol_clean,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d')
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        time.sleep(1)
        
        response = session.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['data'])
        
        df = df.rename(columns={
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        return df
        
    except Exception as e:
        logging.error(f"Direct NSE fetch failed: {str(e)}")
        raise DataFetchError(f"Direct NSE fetch failed: {str(e)}")

#######################################################################################################################
#Function: get_nse_data
#Input: symbol, period, max_retries
#Output: features, df['Close'].values
#Description: Function to fetch NSE data with retries
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def get_nse_data(symbol, period='1y', max_retries=3):
    """Fetch NSE data with retries"""
    for attempt in range(max_retries):
        try:
            days = 365 if period == '1y' else 730
            
            # Get the correct index name from the mapping
            index_name = INDICES[symbol]['ns']
            
            if attempt > 0:
                time.sleep(2)
            
            data = ns.get_historical_index(index_name, days=days)
            if not data or len(data) < 50:
                raise ValueError(f"Insufficient data from niftystocks for {index_name}")
            
            df = pd.DataFrame(data)
            df = df.rename(columns={
                'Open Price': 'Open',
                'High Price': 'High',
                'Low Price': 'Low',
                'Close Price': 'Close',
                'Total Traded Quantity': 'Volume'
            })
            
            features = prepare_features(df)
            return features, df['Close'].values
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"All attempts failed for {symbol}: {str(e)}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")

#######################################################################################################################
#Function: get_stock_data_yf
#Input: symbol, period, max_retries
#Output: features, data['Close'].values
#Description: Function to fetch stock data from Yahoo Finance with retries
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################

def get_stock_data_yf(symbol, period='1y', max_retries=3):
    """Fetch data from yfinance with retries"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)
            
            # Get the correct Yahoo Finance symbol
            yf_symbol = INDICES[symbol]['yf']
            stock = yf.Ticker(yf_symbol)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data available for {yf_symbol}")
            
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            
            features = prepare_features(data)
            return features, data['Close'].values
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"All yfinance attempts failed for {symbol}: {str(e)}")
                raise
            logging.warning(f"Yfinance attempt {attempt + 1} failed: {str(e)}")

#######################################################################################################################
#Function: fetch_nse_direct
#Input: symbol, days
#Output: df
#Description: Function to fetch data directly from the NSE website
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def fetch_nse_direct(symbol, days=365):
    """Fetch data directly from NSE website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Get the correct NSE symbol
        symbol_clean = INDICES[symbol]['nse']
        
        base_url = "https://www.nseindia.com/api/historical/indices"
        params = {
            'symbol': symbol_clean,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d')
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        time.sleep(1)
        
        response = session.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['data'])
        
        df = df.rename(columns={
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        return df
        
    except Exception as e:
        logging.error(f"Direct NSE fetch failed: {str(e)}")
        raise DataFetchError(f"Direct NSE fetch failed: {str(e)}")

#######################################################################################################################
#Function: prepare_features
#Input: df
#Output: df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values
#Description: Function to prepare features from DataFrame
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def prepare_features(df):
    """Prepare features from DataFrame"""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError("Insufficient data points after processing")
    
    return df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values

#######################################################################################################################
#Function: get_data_with_fallbacks
#Input: symbol, period, max_retries
#Output: get_nse_data(symbol, period, max_retries)
#Description: Function to fetch data with fallbacks
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def get_data_with_fallbacks(symbol, period='1y', max_retries=3):
    errors = []
    
    # Validate symbol first
    if symbol not in INDICES:
        raise ValueError(f"Invalid symbol: {symbol}")
        
    # Try niftystocks first
    try:
        return get_nse_data(symbol, period, max_retries)
    except Exception as e:
        errors.append(f"NiftyStocks error: {str(e)}")
        logging.warning(f"NiftyStocks failed for {symbol}: {str(e)}")
    
    time.sleep(1)
    
    # Try direct NSE fetch
    try:
        df = fetch_nse_direct(symbol)
        if not df.empty:
            features = prepare_features(df)
            return features, df['Close'].values
    except Exception as e:
        errors.append(f"Direct NSE error: {str(e)}")
        logging.warning(f"Direct NSE failed for {symbol}: {str(e)}")
    
    time.sleep(1)
    
    # Try yfinance as last resort
    try:
        return get_stock_data_yf(symbol, period, max_retries)
    except Exception as e:
        errors.append(f"YFinance error: {str(e)}")
        logging.warning(f"YFinance failed for {symbol}: {str(e)}")
    
    error_msg = f"All data sources failed for {symbol}. Errors: {'; '.join(errors)}"
    logging.error(error_msg)
    raise DataFetchError(error_msg)

# INDICES = {
#     'NIFTY50': '^NSEI',
#     'NSE': '^NSEI',
#     'SENSEX': '^BSESN',
#     'BANKNIFTY': '^NSEBANK',
#     'NIFTYAUTO': 'NIFTY-AUTO.NS',
#     'NIFTYFINSERV': 'NIFTY-FIN-SERVICE.NS',
#     'NIFTYFMCG': 'NIFTY-FMCG.NS',
#     'NIFTYIT': 'NIFTY-IT.NS',
#     'NIFTYMETAL': 'NIFTY-METAL.NS',
#     'NIFTYPHARMA': 'NIFTY-PHARMA.NS'
# }

#######################################################################################################################
#Function: index
#Input: None
#Output: render_template('index.html')
#Description: Function to render the index.html template
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: get_indices
#Input: None
#Output: jsonify(indices)
#Description: Function to return the list of available indices
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/indices')
def get_indices():
    try:
        indices = list(INDICES.keys())
        print(f"API call to /api/indices - Returning: {indices}")  # Debug print
        logging.info(f"API call to /api/indices - Returning: {indices}")  # Debug log
        return jsonify(indices)
    except Exception as e:
        print(f"Error in /api/indices: {str(e)}")  # Debug print
        logging.error(f"Error in /api/indices: {str(e)}")  # Debug log
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: predict
#Input: symbol
#Output: jsonify(response)
#Description: Function to predict stock prices for the next 7, 15, or 30 days
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/predict/<symbol>')
def predict(symbol):
    try:
        if symbol not in INDICES:
            return jsonify({'error': f'Invalid index symbol: {symbol}'}), 400
            
        prediction_days = int(request.args.get('days', 7))
        if prediction_days not in [7, 15, 30]:
            prediction_days = 7
        
        logging.info(f"Fetching data for {symbol} with {prediction_days} days prediction")
        
        try:
            features, close_prices = get_data_with_fallbacks(symbol)
        except DataFetchError as e:
            logging.error(f"Data fetch error for {symbol}: {str(e)}")
            return jsonify({'error': str(e)}), 503
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
            return jsonify({'error': f'Error fetching data: {str(e)}'}), 500
            
        if len(features) < 30 + prediction_days:
            return jsonify({'error': 'Insufficient data points'}), 400
        
        X_train, X_val, y_train, y_val, scaler = prepare_data(features, prediction_days=prediction_days)
        
        model = EnhancedStockBiLSTM(input_size=features.shape[1], output_size=prediction_days)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_dataset = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        metrics = calculate_metrics(model, X_val, y_val, scaler)
        
        last_sequence = features[-30:]
        X = torch.FloatTensor(scaler.transform(last_sequence)).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            scaled_predictions = model(X)[0].numpy()
        
        pred_reshaped = np.zeros((len(scaled_predictions), scaler.n_features_in_))
        pred_reshaped[:, 0] = scaled_predictions
        predictions = scaler.inverse_transform(pred_reshaped)[:, 0]
        
        historical = close_prices[-30:].tolist()
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(-29, prediction_days + 1)]
        
        return jsonify({
                'dates': dates,
                'values': historical + predictions.tolist(),
                'predictions': predictions.tolist(),
                'lastClose': historical[-1],
                'metrics': metrics
            })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

#######################################################################################################################
#Function: calculate_metrics
#Input: model, X_val, y_val, scaler
#Output: {'mape': float(mape), 'r2': float(r2), 'accuracy': float(100 - mape)}
#Description: Function to calculate the model metrics
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
def calculate_metrics(model, X_val, y_val, scaler):
    try:
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            predictions = model(X_val_tensor)
            
            pred_reshaped = np.zeros((len(predictions), scaler.n_features_in_))
            pred_reshaped[:, 0] = predictions.numpy()[:, 0]
            y_reshaped = np.zeros((len(y_val), scaler.n_features_in_))
            y_reshaped[:, 0] = y_val[:, 0]
            
            pred_actual = scaler.inverse_transform(pred_reshaped)[:, 0]
            y_actual = scaler.inverse_transform(y_reshaped)[:, 0]
            
            mape = mean_absolute_percentage_error(y_actual, pred_actual)
            r2 = r2_score(y_actual, pred_actual)
            
            return {
                'mape': float(mape),
                'r2': float(r2),
                'accuracy': float(100 - mape)
            }
    except Exception as e:
        logging.error(f"Metrics calculation error: {str(e)}")
        raise

#######################################################################################################################
#Function: check_data_sources
#Input: None
#Output: jsonify({'status': 'operational' if any(v == 'available' for v in results.values()) else 'down', 'sources': results})
#Description: Function to check the availability of data sources
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/data/status')
def check_data_sources():
    try:
        results = {}
        
        def check_source(name, symbol='^NSEI'):
            try:
                if name == 'niftystocks':
                    ns.get_historical_index('NIFTY 50', days=1)
                elif name == 'yfinance':
                    yf.Ticker(symbol).history(period='1d')
                elif name == 'nse_direct':
                    fetch_nse_direct(symbol, days=1)
                return 'available'
            except:
                return 'unavailable'
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(check_source, source): source
                for source in ['niftystocks', 'yfinance', 'nse_direct']
            }
            
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                results[source] = future.result()
        
        return jsonify({
            'status': 'operational' if any(v == 'available' for v in results.values()) else 'down',
            'sources': results
        })
    except Exception as e:
        logging.error(f"Error checking data sources: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: health_check
#Input: None
#Output: jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0'})
#Description: Function to check the health of the application
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: not_found_error
#Input: error
#Output: jsonify({'error': 'Not found'}), 404
#Description: Function to handle 404 errors
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

#######################################################################################################################
#Function: internal_error
#Input: error
#Output: jsonify({'error': 'Internal server error'}), 500
#Description: Function to handle 500
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: service_unavailable_error
#Input: error
#Output: jsonify({'error': 'Service temporarily unavailable'}), 503
#Description: Function to handle 503 errors
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(503)
def service_unavailable_error(error):
    return jsonify({'error': 'Service temporarily unavailable'}), 503

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    

#######################################################################################################################
#Function: initialize_app
#Input: app
#Output: None
#Description: This function initializes the application state by checking the data sources
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################    
    initialize_app(app)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
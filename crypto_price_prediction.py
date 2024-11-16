"""
Purpose:
    Create normalized sequences from cryptocurrency price data for model training.

Inputs:
    - DataFrame containing price data for a single cryptocurrency

Process:
    1. Extracts OHLCV columns (Open, High, Low, Close, Volume)
    2. Normalizes features using MinMaxScaler to range [0,1]
    3. Creates input sequences of specified length (default 60 timesteps)
       Each sequence contains 60 consecutive OHLCV data points
    4. Target for each sequence is the next closing price after sequence

Outputs:
    - X_sequences: Array of input sequences
    - y_sequences: Array of target closing prices 
    - scaler: Fitted scaler object for inverse transformations
"""

import os
os.environ['TENSORFLOW_METAL'] = '0'  # Disable Metal/GPU support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU devices

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import json
import os
from datetime import datetime

# Load and prepare the data
def load_data(filepath):
    """
    Load cryptocurrency price data from multiple CSV files in a directory
    filepath: path to directory containing CSV files
    """
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
    
    # Initialize empty list to store dataframes
    dfs = []
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read each CSV file and append to list
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(filepath, file))
            df['Time'] = pd.to_datetime(df['Time'])
            dfs.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    # Concatenate all dataframes
    if not dfs:
        raise ValueError("No valid CSV files found in directory")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sort_values(by=['Coin', 'Time'], inplace=True)
    
    print(f"Total records: {len(combined_df)}")
    print(f"Unique coins: {combined_df['Coin'].nunique()}")
    
    return combined_df

def create_sequences(df, sequence_length=60):
    """
    Create normalized sequences from a single coin's dataframe with NO overlap
    sequence_length: number of 15-min intervals to include in each sequence
    """
    # Select only OHLCV columns
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[ohlcv_cols].values
    
    X_sequences = []
    y_sequences = []
    
    # Create non-overlapping sequences by stepping by sequence_length
    for i in range(0, len(data) - sequence_length - 1, sequence_length):
        # Extract sequence
        sequence = data[i:(i + sequence_length + 1)]  # +1 to include target
        
        # Create scaler for this sequence
        scaler = MinMaxScaler()
        normalized_sequence = scaler.fit_transform(sequence)
        
        # Split into input and target
        X_sequences.append(normalized_sequence[:-1])
        y_sequences.append(normalized_sequence[-1, 3])  # 3 is the Close column
        
    return np.array(X_sequences), np.array(y_sequences), scaler

def prepare_dataset(df, sequence_length=60, num_coins=5):
    """Prepare the complete dataset from multiple coins"""
    all_sequences_X = []
    all_sequences_y = []
    scalers = {}
    selected_coins = []
    
    # Randomly select coins instead of taking first n
    unique_coins = np.random.choice(df['Coin'].unique(), size=num_coins, replace=False)
    
    for coin in unique_coins:
        # Get data for single coin
        coin_df = df[df['Coin'] == coin].copy()
        
        if len(coin_df) < sequence_length + 1:
            print(f"Skipping {coin}: insufficient data")
            continue
            
        # Create sequences
        X_seq, y_seq, scaler = create_sequences(coin_df, sequence_length)
        
        all_sequences_X.append(X_seq)
        all_sequences_y.append(y_seq)
        scalers[coin] = scaler
        selected_coins.append(coin)
        print(f"Processed {coin}: {len(X_seq)} sequences")

    # Combine all sequences
    X = np.concatenate(all_sequences_X, axis=0)
    y = np.concatenate(all_sequences_y, axis=0)
    
    return X, y, scalers, selected_coins

def create_model(sequence_length, n_features):
    """Create the LSTM model"""
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(128, return_sequences=True, 
                   input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.3))
    
    # Second LSTM layer
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_training_history(history, save_dir=None):
    """Plot the training and validation loss"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = f'{save_dir}/training_history.png' if save_dir else f'training_history_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()
    
    # Plot learning curves separately with more detail
    plt.figure(figsize=(15, 5))
    
    # Detailed loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Detailed Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Loss distribution plot
    plt.subplot(1, 2, 2)
    plt.hist(history.history['loss'], alpha=0.5, label='Training Loss', bins=20)
    plt.hist(history.history['val_loss'], alpha=0.5, label='Validation Loss', bins=20)
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    save_path = f'{save_dir}/training_analysis.png' if save_dir else f'training_analysis_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()

def plot_sequence_prediction(model, X, y, sequence_idx, run_dir, coin_name=None):
    """
    Plot a single sequence and its prediction
    """
    # Get the sequence and true value
    sequence = X[sequence_idx]
    true_value = y[sequence_idx]
    
    # Make prediction
    prediction = model.predict(sequence.reshape(1, sequence.shape[0], sequence.shape[1]))[0][0]
    
    # Get the closing prices from the sequence
    closing_prices = sequence[:, 3]  # index 3 is Close price
    
    # Create time points
    time_points = range(len(closing_prices) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_points[:-1], closing_prices, 'b-', label='Historical Prices')
    plt.plot(time_points[-2:], [closing_prices[-1], true_value], 'g-', label='True Next Price')
    plt.plot(time_points[-2:], [closing_prices[-1], prediction], 'r--', label='Predicted Next Price')
    
    plt.title(f'Price Prediction{" for " + coin_name if coin_name else ""}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    
    # Add prediction details
    error = abs(true_value - prediction)
    error_pct = (error / true_value) * 100
    
    plt.text(0.02, 0.98, 
             f'True Value: {true_value:.4f}\n'
             f'Predicted: {prediction:.4f}\n'
             f'Abs Error: {error:.4f}\n'
             f'Error %: {error_pct:.2f}%', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Define predictions directory
    predictions_dir = os.path.join(run_dir, 'predictions')

    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(predictions_dir, f'sequence_{sequence_idx}.png')

    plt.savefig(save_path)
    plt.close()

def calculate_trading_returns(y_true, y_pred):
    """
    Calculate returns from trading based on predictions
    - Go long if predicted price is higher than current price
    - Go short if predicted price is lower than current price
    Returns array of trade returns and overall statistics
    """
    # Calculate percentage changes
    returns = []
    correct_predictions = 0
    total_trades = len(y_true)
    
    for i in range(len(y_true)):
        # If prediction is higher than current -> go long
        # If prediction is lower than current -> go short
        predicted_direction = 1 if y_pred[i] > y_true[i] else -1
        actual_direction = 1 if y_true[i] > y_pred[i] else -1
        
        # Calculate actual return (as percentage)
        trade_return = abs(y_true[i] - y_pred[i]) * predicted_direction
        returns.append(trade_return)
        
        # Track correct predictions
        if predicted_direction == actual_direction:
            correct_predictions += 1
    
    returns = np.array(returns)
    
    stats = {
        'total_return': float(np.sum(returns)),
        'avg_return_per_trade': float(np.mean(returns)),
        'win_rate': float(correct_predictions / total_trades),
        'total_trades': total_trades,
        'profitable_trades': len(returns[returns > 0]),
        'unprofitable_trades': len(returns[returns < 0])
    }
    
    return returns, stats

def evaluate_model(model, X_val, y_val, run_dir, num_examples=5):
    """
    Evaluate model performance and visualize predictions
    """
    # Calculate overall metrics
    predictions = model.predict(X_val)
    mse = np.mean((y_val - predictions.flatten()) ** 2)
    mae = np.mean(np.abs(y_val - predictions.flatten()))
    rmse = np.sqrt(mse)
    
    # Calculate percentage errors
    percentage_errors = np.abs(y_val - predictions.flatten()) / y_val * 100
    mean_percentage_error = np.mean(percentage_errors)
    
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    
    # Calculate trading returns
    print("\nCalculating trading returns...")
    returns, trading_stats = calculate_trading_returns(y_val, predictions.flatten())
    
    print("\nTrading Statistics:")
    print(f"Total Return: {trading_stats['total_return']:.2f}%")
    print(f"Average Return per Trade: {trading_stats['avg_return_per_trade']:.2f}%")
    print(f"Win Rate: {trading_stats['win_rate']*100:.2f}%")
    print(f"Total Trades: {trading_stats['total_trades']}")
    print(f"Profitable Trades: {trading_stats['profitable_trades']}")
    print(f"Unprofitable Trades: {trading_stats['unprofitable_trades']}")
    
    # Plot returns distribution
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50)
    plt.title('Distribution of Trading Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(run_dir, 'returns_distribution.png'))
    plt.close()
    
    # Add trading stats to metrics
    trading_stats

    
    # Plot random examples
    print("\nPlotting random prediction examples...")
    random_indices = np.random.choice(len(X_val), min(num_examples, len(X_val)), replace=False)
    
    for idx in random_indices:
        plot_sequence_prediction(model, X_val, y_val, idx, run_dir)

    return trading_stats


def save_model_data(model, history, config, metrics, timestamp):
    """Save all model-related data"""
    # Create directory for this run
    run_dir = f'model_runs/{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model configuration
    config_dict = {
        'SEQUENCE_LENGTH': config['SEQUENCE_LENGTH'],
        'NUM_COINS': config['NUM_COINS'],
        'TRAIN_SPLIT': config['TRAIN_SPLIT'],
        'BATCH_SIZE': config['BATCH_SIZE'],
        'EPOCHS': config['EPOCHS'],
        'SELECTED_COINS': config['SELECTED_COINS'],  # Add selected coins to config
        'architecture': model.get_config(),
        'optimizer': model.optimizer.get_config()
    }
    
    with open(f'{run_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Save training history
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mae': history.history['mae'],
        'val_mae': history.history['val_mae']
    }
    np.save(f'{run_dir}/training_history.npy', history_dict)
    
    # Save evaluation metrics
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model
    model.save(f'{run_dir}/model.keras')
    
    # Save plots
    plot_training_history(history, save_dir=run_dir)
    
    print(f"\nModel run data saved in: {run_dir}")
    return run_dir

def main():
    try:
        # Add this at the start to debug the path
        print("Current working directory:", os.getcwd())
        print("Checking if data directory exists:", os.path.exists('solana_meme_coins'))
        
        # Configuration
        config = {
            'SEQUENCE_LENGTH': 30,
            'NUM_COINS': 230,
            'TRAIN_SPLIT': 0.9,
            'BATCH_SIZE': 64,
            'EPOCHS': 200
        }
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data with explicit path
        print("Loading data...")
        data_dir = os.path.join(os.getcwd(), 'solana_meme_coins')
        print("Looking for data in:", data_dir)
        df = load_data(data_dir)
        
        # Prepare sequences
        print("Preparing sequences...")
        X, y, scalers, selected_coins = prepare_dataset(df, config['SEQUENCE_LENGTH'], config['NUM_COINS'])
        config['SELECTED_COINS'] = selected_coins # Add selected coins to config
        
        # Split into train/validation
        split_idx = int(len(X) * config['TRAIN_SPLIT'])
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Validation sequences: {X_val.shape}")
        
        # Create and train model
        print("Creating model...")
        with tf.device('/CPU:0'):
            model = create_model(sequence_length=config['SEQUENCE_LENGTH'], n_features=5)
            
            # Add early stopping
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            # Add model checkpoint
            checkpoint = ModelCheckpoint(
                f'model_runs/{timestamp}/best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            print("Training model...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config['EPOCHS'],
                batch_size=config['BATCH_SIZE'],
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
        
        # Evaluate model
        predictions = model.predict(X_val)
        pred_flat = predictions.flatten()
        
        # Calculate percentage error safely
        non_zero_mask = y_val != 0
        percentage_errors = np.zeros_like(y_val)
        percentage_errors[non_zero_mask] = np.abs(y_val[non_zero_mask] - pred_flat[non_zero_mask]) / np.abs(y_val[non_zero_mask]) * 100
        
        metrics = {
            'mse': float(np.mean((y_val - pred_flat) ** 2)),
            'mae': float(np.mean(np.abs(y_val - pred_flat))),
            'rmse': float(np.sqrt(np.mean((y_val - pred_flat) ** 2))),
            'mean_percentage_error': float(np.mean(percentage_errors))
        }
        
        
        # Save all model data

        run_dir = save_model_data(model, history, config, metrics, timestamp)
        
        # Add model evaluation
        print("\nEvaluating model and generating prediction visualizations...")
        trading_stats = evaluate_model(model, X_val, y_val, run_dir, num_examples=50)
        metrics['trading_stats'] = trading_stats
        
        # Create a README for this run
        readme_content = f"""# Model Run {timestamp}

## Configuration
- Sequence Length: {config['SEQUENCE_LENGTH']}
- Number of Coins: {config['NUM_COINS']}
- Train Split: {config['TRAIN_SPLIT']}
- Batch Size: {config['BATCH_SIZE']}
- Epochs: {config['EPOCHS']}

## Performance Metrics
- MSE: {metrics['mse']:.6f}
- MAE: {metrics['mae']:.6f}
- RMSE: {metrics['rmse']:.6f}
- Mean Percentage Error: {metrics['mean_percentage_error']:.2f}%

## Files
- `model.keras`: Trained model
- `best_model.keras`: Best model during training
- `config.json`: Model configuration
- `metrics.json`: Evaluation metrics
- `training_history.npy`: Training history
- `training_history.png`: Training curves
- `training_analysis.png`: Detailed training analysis
"""
        
        with open(f'{run_dir}/README.md', 'w') as f:
            f.write(readme_content)
        
        print("\nModel evaluation and visualization completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    import tensorflow as tf
    import random
    
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
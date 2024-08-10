import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Input
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from ta import add_all_ta_features
from sklearn.feature_selection import mutual_info_regression
import sys
import datetime

# Get current date and time for the log file name
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = f"output_log_{current_time}.txt"

# Open the file with UTF-8 encoding
log_file = open(log_file_name, 'w', encoding='utf-8')

# Redirect stdout to the file
sys.stdout = log_file

# From this point on, all print statements will be written to the file
print(f"Log started at {current_time}")

# Step 1: Prepare the data
# Load the data
df = pd.read_csv('EURUSD15.csv', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], delimiter='\t')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
df.set_index('Date', inplace=True)

# Scale volume (assuming it's in thousands)
df['Volume'] = df['Volume'] * 1000

# Add technical indicators
df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume",
    fillna=True
)

print(f"NaN values after adding indicators: {df.isna().sum().sum()}")

# Drop any rows with NaN values
rows_before = len(df)
# df = dropna(df)
rows_after = len(df)
print(f"Rows dropped due to NaN values: {rows_before - rows_after}")

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# Create sequences
def create_sequences(data, seq_length, future_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - future_steps + 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[(i + seq_length):(i + seq_length + future_steps), 3])  # Predict future Close prices
    return np.array(X), np.array(y)


seq_length = 90 # Number of time steps to look back
future_steps = 15  # Number of steps to predict into the future

X, y = create_sequences(scaled_data, seq_length, future_steps)

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the input shape
input_shape = (seq_length, X_train.shape[2])  # Make sure seq_length is defined
input_layer = Input(shape=input_shape)

model = Sequential([
    input_layer,
    LSTM(150, return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.02),
    LSTM(150, return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.02),
    LSTM(100, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.02),
    Dense(future_steps)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print model summary
model.summary()

# Step 3: Train the model
#early_stopping = EarlyStopping(monitor='val_loss', patience=75, min_delta=0.000005, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.0001)
model.fit(X_train, y_train, batch_size=32, epochs=300, validation_split=0.1, callbacks=[reduce_lr], verbose=2)


def predict_future(model, last_sequence, scaler, steps=15):
    predictions = []
    current_sequence = last_sequence.reshape(1, *last_sequence.shape)
    close_index = list(df.columns).index('Close')

    for _ in range(0, steps, future_steps):
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.extend(next_pred[0])

        # Update the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -future_steps, axis=1)
        for i in range(future_steps):
            current_sequence[0, -future_steps + i, close_index] = next_pred[0, i]

    predictions = np.array(predictions[:steps])  # Trim to exact number of steps

    # Ensure all arrays have the same length
    last_known_values = current_sequence[0, -1, :]
    repeated_values = np.tile(last_known_values, (steps, 1))

    # Replace the Close prices with our predictions
    repeated_values[:, close_index] = predictions

    # Inverse transform the entire sequence
    full_sequence = scaler.inverse_transform(repeated_values)

    return full_sequence[:, close_index]  # Return only the Close prices


def analyze_trend(prices, starting_balance, lot_size=0.01, long_entry_threshold=0.0005, short_entry_threshold=0.0005,
                  long_exit_threshold=0.0003, short_exit_threshold=0.0003):
    initial_price = prices[0]
    max_price = initial_price
    min_price = initial_price
    position = "NONE"
    entry_price = 0
    current_balance = starting_balance

    for i, price in enumerate(prices[1:], 1):
        if position == "NONE":
            if price >= initial_price + long_entry_threshold:
                position = "LONG"
                entry_price = price
            elif price <= initial_price - short_entry_threshold:
                position = "SHORT"
                entry_price = price
        elif position == "LONG":
            if price > max_price:
                max_price = price
            if price <= max_price - long_exit_threshold:
                profit = (price - entry_price) * lot_size * 100000  # Assuming EURUSD
                current_balance += profit
                return "LONG", i, entry_price, price, profit, current_balance
        elif position == "SHORT":
            if price < min_price:
                min_price = price
            if price >= min_price + short_exit_threshold:
                profit = (entry_price - price) * lot_size * 100000  # Assuming EURUSD
                current_balance += profit
                return "SHORT", i, entry_price, price, profit, current_balance

    return "HOLD", 0, 0, 0, 0, current_balance


def analyze_feature_importance(X, y, feature_names):
    # X shape is (samples, time_steps, features)
    # y shape is (samples, future_steps)

    # Use the last time step for each sample in X
    X_last_step = X[:, -1, :]

    # Use the first predicted value for each sample in y
    y_first_pred = y[:, 0]

    # Ensure X_last_step and y_first_pred have the same number of samples
    assert X_last_step.shape[0] == y_first_pred.shape[0], "Mismatch in number of samples"

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X_last_step, y_first_pred)

    # Create a series with feature names and scores
    mi_scores = pd.Series(mi_scores, index=feature_names)
    mi_scores = mi_scores.sort_values(ascending=False)

    print("Top 10 most important features:")
    print(mi_scores.head(10))

    return mi_scores


# After model training
print("\nAnalyzing feature importance...")
feature_names = df.columns.tolist()
mi_scores = analyze_feature_importance(X_train, y_train, feature_names)

# Evaluate Model Performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Test the prediction and analyze frequency
total_predictions = len(X_test)
actionable_predictions = 0
total_profit = 0
starting_balance = 10000  # Starting with $10,000
current_balance = starting_balance

print(f"Starting analysis of {total_predictions} predictions...")

for i in range(total_predictions):
    if i % 50 == 0:  # Print progress every 50 predictions
        print(f"Analyzing prediction {i + 1}/{total_predictions}...")

    future_prices = predict_future(model, X_test[i], scaler)
    action, steps, entry_price, exit_price, profit, current_balance = analyze_trend(future_prices, current_balance)

    if action != "HOLD":
        actionable_predictions += 1
        total_profit += profit
        print(f"Signal found! Predicted {action} signal. Entry at {entry_price:.5f}, "
              f"Exit at {exit_price:.5f}, after {steps} steps. "
              f"Profit: ${profit:.2f}, Current Balance: ${current_balance:.2f}")


print("\nAnalysis complete. Summary of results:")
print(f"Starting Balance: ${starting_balance:.2f}")
print(f"Final Balance: ${current_balance:.2f}")
print(f"Total profit: ${current_balance - starting_balance:.2f}")
print(f"Actionable predictions: {actionable_predictions} out of {total_predictions}")
print(f"Frequency of actionable predictions: {actionable_predictions / total_predictions:.2%}")
print(f"Average profit per trade: ${total_profit / actionable_predictions if actionable_predictions else 0:.2f}")

# Close the file
sys.stdout.close()

# Restore stdout to the console
sys.stdout = sys.__stdout__

print(f"Log file created: {log_file_name}")

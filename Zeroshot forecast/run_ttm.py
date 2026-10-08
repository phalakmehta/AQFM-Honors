import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

try:
    from tsfm_public import TinyTimeMixerForPrediction
    from transformers import Trainer, TrainingArguments
except ImportError:
    print("Please install tsfm_public and transformers: pip install tsfm_public transformers")

# Target and covariate columns based on the CSV structure
TARGET_COLS = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)']
COVARIATE_COLS = ['AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)']
ALL_COLS = TARGET_COLS + COVARIATE_COLS

CONTEXT_LENGTH = 512
FORECAST_HORIZON = 96
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"

def load_and_preprocess(data_dir, freq='1h'):
    print(f"Searching for CSVs in {data_dir}...")
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
        
    # Group files by station based on filename
    station_files = {}
    for f in csv_files:
        basename = os.path.basename(f)
        station = basename.replace('_cleaned.csv', '')
        if station not in station_files:
            station_files[station] = []
        station_files[station].append(f)
        
    print(f"Found {len(csv_files)} files across {len(station_files)} stations.")
    
    station_data = {}
    
    # Process each station iteratively to save memory
    for station, files in station_files.items():
        temp_dfs = []
        for file in files:
            # We don't need 'Station' column anymore since we group by filename
            df = pd.read_csv(file, usecols=['Timestamp'] + ALL_COLS)
            temp_dfs.append(df)
            
        group = pd.concat(temp_dfs, ignore_index=True)
        group.columns = group.columns.str.strip()
        
        group['Timestamp'] = pd.to_datetime(group['Timestamp'], errors='coerce')
        group = group.dropna(subset=['Timestamp'])
        
        group = group.sort_values('Timestamp').set_index('Timestamp')
        
        # Resample to common frequency and remove duplicates
        group = group[~group.index.duplicated(keep='first')]
        group = group.asfreq(freq)
        
        group = group[ALL_COLS]
        
        # Handle negatives
        for col in ALL_COLS:
            if 'AT' not in col:
                group.loc[group[col] < 0, col] = np.nan
                
        # Forward fill and interpolate
        group.ffill(limit=6, inplace=True)
        group.interpolate(method='time', inplace=True)
        group.dropna(inplace=True)
        
        if len(group) > CONTEXT_LENGTH + FORECAST_HORIZON:
            # Downcast to float32 to halve memory usage
            station_data[station] = group.astype(np.float32)
            
    print(f"Successfully processed {len(station_data)} stations with sufficient data length.")
    return station_data

def split_and_scale(station_data):
    print("Splitting chronologically and fitting StandardScaler (iteratively)...")
    
    train_data, val_data, test_data = {}, {}, {}
    scaler = StandardScaler()
    
    # Iteratively fit the scaler using partial_fit to save memory
    for station, df in station_data.items():
        n = len(df)
        train_end = int(n * 0.7)
        train_df = df.iloc[:train_end]
        scaler.partial_fit(train_df)
    
    # Transform iteratively
    for station, df in station_data.items():
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_scaled = scaler.transform(df.iloc[:train_end])
        val_scaled = scaler.transform(df.iloc[train_end:val_end])
        test_scaled = scaler.transform(df.iloc[val_end:])
        
        # Store scaled data as float32 tensors/arrays
        train_data[station] = pd.DataFrame(train_scaled, columns=df.columns, index=df.index[:train_end], dtype=np.float32)
        val_data[station] = pd.DataFrame(val_scaled, columns=df.columns, index=df.index[train_end:val_end], dtype=np.float32)
        test_data[station] = pd.DataFrame(test_scaled, columns=df.columns, index=df.index[val_end:], dtype=np.float32)
        
    return train_data, val_data, test_data, scaler

class TTMDataset(Dataset):
    def __init__(self, station_dict):
        self.samples = []
        
        for station, df in station_dict.items():
            # Convert to float32 to save memory
            data = df.values.astype(np.float32)
            n_samples = len(data) - CONTEXT_LENGTH - FORECAST_HORIZON + 1
            if n_samples > 0:
                self.samples.append({
                    'data': data,
                    'n_samples': n_samples
                })
                
        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for s in self.samples:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + s['n_samples'])
            
    def __len__(self):
        return self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 1 else 0
    
    def __getitem__(self, idx):
        # Find which station this index belongs to
        station_idx = next(i for i, total in enumerate(self.cumulative_lengths[1:]) if idx < total)
        
        # Local index within that station
        local_idx = idx - self.cumulative_lengths[station_idx]
        
        data = self.samples[station_idx]['data']
        
        X = data[local_idx : local_idx + CONTEXT_LENGTH]
        y = data[local_idx + CONTEXT_LENGTH : local_idx + CONTEXT_LENGTH + FORECAST_HORIZON]
        
        return {
            "past_values": torch.tensor(X, dtype=torch.float32), 
            "future_values": torch.tensor(y, dtype=torch.float32)
        }

def calculate_metrics(y_true, y_pred, cols):
    metrics = {}
    for i, col in enumerate(cols):
        yt = y_true[:, :, i].flatten()
        yp = y_pred[:, :, i].flatten()
        
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mape = mean_absolute_percentage_error(yt, yp)
        
        metrics[col] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        print(f"Metrics for {col}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
        
    return metrics

def run_evaluation(model, dataset, scaler):
    print("Running Zero-Shot Evaluation...")
    # Evaluate on a random sample to avoid iterating through millions of items during zero-shot evaluation
    # Set a reasonable max sample size
    max_eval_samples = 1000
    if len(dataset) > max_eval_samples:
        indices = torch.randperm(len(dataset))[:max_eval_samples]
        eval_dataset = torch.utils.data.Subset(dataset, indices)
    else:
        eval_dataset = dataset
        
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
    model.eval()
    
    predictions = []
    y_test_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(past_values=batch["past_values"])
            preds = outputs.prediction_outputs if hasattr(outputs, 'prediction_outputs') else outputs[0]
            predictions.append(preds.numpy())
            y_test_true.append(batch["future_values"].numpy())
            
    predictions = np.concatenate(predictions, axis=0)
    y_test_true = np.concatenate(y_test_true, axis=0)
    
    # Inverse transform
    print("Inverse transforming predictions...")
    pred_shape = predictions.shape
    true_shape = y_test_true.shape
    
    preds_flat = predictions.reshape(-1, pred_shape[-1])
    true_flat = y_test_true.reshape(-1, true_shape[-1])
    
    preds_unscaled = scaler.inverse_transform(preds_flat).reshape(pred_shape)
    true_unscaled = scaler.inverse_transform(true_flat).reshape(true_shape)
    
    # Enforce Physical Constraints (e.g., PM2.5 <= PM10)
    # Find indices for PM2.5 and PM10
    if 'PM2.5 (µg/m³)' in ALL_COLS and 'PM10 (µg/m³)' in ALL_COLS:
        pm25_idx = ALL_COLS.index('PM2.5 (µg/m³)')
        pm10_idx = ALL_COLS.index('PM10 (µg/m³)')
        # Force PM2.5 to be min(PM2.5, PM10)
        preds_unscaled[:, :, pm25_idx] = np.minimum(preds_unscaled[:, :, pm25_idx], preds_unscaled[:, :, pm10_idx])
        # Force values to be non-negative
        preds_unscaled = np.maximum(preds_unscaled, 0.0)

    # Calculate metrics for target columns only
    target_indices = [ALL_COLS.index(c) for c in TARGET_COLS]
    preds_targets = preds_unscaled[:, :, target_indices]
    true_targets = true_unscaled[:, :, target_indices]
    
    calculate_metrics(true_targets, preds_targets, TARGET_COLS)
    
    # Plotting
    print("Plotting forecasts...")
    sample_idx = 0
    fig, axes = plt.subplots(len(TARGET_COLS), 1, figsize=(12, 5 * len(TARGET_COLS)))
    if len(TARGET_COLS) == 1:
        axes = [axes]
        
    for i, (col, ax) in enumerate(zip(TARGET_COLS, axes)):
        ax.plot(true_targets[sample_idx, :, i], label='Ground Truth')
        ax.plot(preds_targets[sample_idx, :, i], label='Prediction', linestyle='--')
        ax.set_title(f"Forecast vs Truth for {col} (Sample {sample_idx})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Concentration")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('ttm_forecast_vs_truth.png')
    print("Saved plot to ttm_forecast_vs_truth.png")

def main():
    data_dir = 'yearly_data'
    
    # 1 & 2. Load and Preprocess
    station_data = load_and_preprocess(data_dir, freq='1h')
    
    # 3. Split and Scale
    train_data, val_data, test_data, scaler = split_and_scale(station_data)
    
    # 4 & 5. Create TTM Datasets
    print("Creating TTM PyTorch datasets...")
    train_dataset = TTMDataset(train_data)
    val_dataset = TTMDataset(val_data)
    test_dataset = TTMDataset(test_data)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # 6. Load Model
    print(f"Loading TTM Model ({MODEL_ID})...")
    model = TinyTimeMixerForPrediction.from_pretrained(
        MODEL_ID, 
        context_length=CONTEXT_LENGTH, 
        prediction_length=FORECAST_HORIZON,
        num_input_channels=len(ALL_COLS)
    )
    
    # 7. Run Zero-shot evaluation
    if len(test_dataset) > 0:
        run_evaluation(model, test_dataset, scaler)
        
    # 8. Few-shot Trainer Template
    print("Setting up Trainer template for Few-Shot Fine-Tuning...")
    training_args = TrainingArguments(
        output_dir="./ttm_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Pipeline ready. To start fine-tuning, uncomment `trainer.train()` in the code.")
    # trainer.train()

if __name__ == "__main__":
    main()

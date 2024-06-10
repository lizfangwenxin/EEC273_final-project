import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

col_list = ['gridID', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'internet']

class NetworkTrafficDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.linear_in = nn.Linear(input_dim, dim_model)
        self.linear_out = nn.Linear(dim_model, input_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.linear_in(src.permute(1, 0, 2))  # [sequence_length, batch_size, dim_model]
        tgt = self.linear_in(tgt.permute(1, 0, 2))  # [sequence_length, batch_size, dim_model]
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.linear_out(output)
        return output.permute(1, 0, 2)  # [batch_size, sequence_length, input_dim]


def create_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def process_file(filepath, base_output_dir, sequence_length=10, num_epochs=10, batch_size=32, learning_rate=0.001):
    print(f"Processing file: {os.path.basename(filepath)}")

    # read data
    data = pd.read_csv(filepath, sep='\t', header=None)
    data.columns = col_list

    data['timeInterval'] = pd.to_datetime(data['timeInterval'], unit='ms')

    #gridID
    grid_ids = data['gridID'].unique()

    file_output_dir = os.path.join(base_output_dir, f"{os.path.basename(filepath).split('.')[0]}_outputImage")
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    file_train_losses = []
    file_test_losses = []

    for grid_id in grid_ids:
        grid_data = data[data['gridID'] == grid_id]

        dataset = grid_data[['timeInterval', 'internet']].dropna()

        if len(dataset) == 0:
            print(f"No data for GridID: {grid_id} in file: {os.path.basename(filepath)}")
            continue

        # normalization
        mean = dataset['internet'].mean()
        std = dataset['internet'].std()
        scaled_data = (dataset['internet'] - mean) / std

        # training set and testing set split
        split_ratio = 0.8
        split_index = int(len(scaled_data) * split_ratio)
        train_data = scaled_data[:split_index].values
        test_data = scaled_data[split_index:].values

        # prepare data set and data loader
        train_dataset = NetworkTrafficDataset(train_data, sequence_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = TransformerTimeSeries(1, 64, 8, 6, 6, 0.1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for src, tgt in train_dataloader:
                tgt_input = tgt[:, :-1]  # [batch_size, sequence_length - 1]
                tgt_output = tgt[:, 1:]  # [batch_size, sequence_length - 1]

                src = src.unsqueeze(-1)  # [batch_size, sequence_length, input_dim]
                tgt_input = tgt_input.unsqueeze(-1)  # [batch_size, sequence_length - 1, input_dim]
                tgt_output = tgt_output.unsqueeze(-1)  # [batch_size, sequence_length - 1, input_dim]

                src_mask = create_mask(src.size(1)).to(src.device)
                tgt_mask = create_mask(tgt_input.size(1)).to(tgt_input.device)

                optimizer.zero_grad()
                output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(output, tgt_output)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}')

        file_train_losses.append(avg_loss)
        print(f"Training complete for GridID: {grid_id}")

        # Prepare the test dataset
        test_dataset = NetworkTrafficDataset(test_data, sequence_length)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Use batch_size=1 for sequential prediction

        # 確保有足夠的測試數據
        if len(test_dataset) == 0:
            print(f"Not enough test data for GridID: {grid_id}")
            continue

        # Make predictions
        model.eval()
        predictions = []
        test_losses = []

        with torch.no_grad():
            for src, tgt in test_dataloader:
                src = src.unsqueeze(-1)  # [batch_size, sequence_length, input_dim]
                tgt_input = src[:, :-1]  # [batch_size, sequence_length - 1]
                tgt_output = src[:, 1:]  # [batch_size, sequence_length - 1]
                src_mask = create_mask(src.size(1)).to(src.device)
                tgt_mask = create_mask(tgt_input.size(1)).to(tgt_input.device)

                predicted = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(predicted, tgt_output)
                test_losses.append(loss.item())

                predictions.append(predicted.squeeze().cpu().numpy()[-1])  # Take the last prediction

        # Inverse transform the predictions to original scale
        predictions = np.array(predictions).flatten()
        predictions = predictions * std + mean

        # Extract the corresponding actual test data for comparison
        actual = test_data[sequence_length:] * std + mean

        # Ensure predictions and actual values have the same length
        actual = actual[:len(predictions)]

        # Calculate the average test MSE
        test_mse = np.mean(test_losses)
        file_test_losses.append(test_mse)

        # Compare predictions with actual values
        plt.figure(figsize=(12, 9))
        plt.plot(dataset['timeInterval'][split_index + sequence_length:], actual, label='Actual Data')
        plt.plot(dataset['timeInterval'][split_index + sequence_length:], predictions, label='Predicted Data')
        plt.legend()
        plt.title(f"GridID: {grid_id}\nTest MSE: {test_mse:.4f}")
        plt.xlabel('Time')
        plt.ylabel('Internet Usage')
        plt.xticks(rotation=90)  # 旋轉 x 軸標籤
        plt.savefig(os.path.join(file_output_dir, f"GridID_{grid_id}.png"))
        plt.close()

        print(f"Finished processing GridID: {grid_id} in file: {os.path.basename(filepath)}")

    # Calculate the average train MSE and test MSE for the file
    file_avg_train_mse = np.mean(file_train_losses)
    file_avg_test_mse = np.mean(file_test_losses)

    print(f"File: {os.path.basename(filepath)}\nAverage Loss MSE: {file_avg_train_mse:.4f}\nAverage Test MSE: {file_avg_test_mse:.4f}")


base_data_dir = '/Users/wenxinfang/PycharmProjects/EEC273/dataset'
base_output_dir = '/Users/wenxinfang/PycharmProjects/EEC273/Transformer_outputImage'

for filename in os.listdir(base_data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(base_data_dir, filename)
        process_file(filepath, base_output_dir)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def Dataset(filepath, max_rows=10000):
    data = pd.read_csv(filepath, sep='\t', header=None, nrows=max_rows)
    data.columns = ["SquareID", "TimeInterval", "CountryCode", "SmsInActivity", "SmsOutActivity", "CallInActivity", "CallOutActivity", "InternetTrafficActivity"]
    dataset = data[['SquareID', 'TimeInterval', 'CountryCode', 'SmsInActivity', 'SmsOutActivity', 'CallInActivity', 'CallOutActivity', 'InternetTrafficActivity']].dropna()
    dataset['TimeInterval'] = pd.to_datetime(dataset['TimeInterval'], unit='ms')
    return dataset

# Define the GCN model 
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def Graph(dataset, look_back=5):
    data_list = []
    internetUsage = MinMaxScaler(feature_range=(0, 1))
    squareIDs = dataset['SquareID'].unique()
    
    for squareID in squareIDs:
        square_data = dataset[dataset['SquareID'] == squareID].sort_values(by='TimeInterval')
        scaled_internet = internetUsage.fit_transform(square_data[['InternetTrafficActivity']])
        square_data['scaled_internet'] = scaled_internet
        
        if len(square_data) <= look_back:
            continue
        
        x = []
        y = []
        edges = []
        time_intervals = []
        
        for i in range(len(square_data) - look_back):
            x.append(square_data.iloc[i:i+look_back]['scaled_internet'].values.flatten())
            y.append(square_data.iloc[i+look_back]['scaled_internet'])
            for j in range(look_back - 1):
                if (i + j + 1) < len(square_data):
                    edges.append((i + j, i + j + 1))
            time_intervals.append(square_data.iloc[i+look_back]['TimeInterval'])
        
        x = np.array(x)
        y = np.array(y)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Ensure edge_index does not exceed the bounds of x
        max_index = len(x) - 1
        edge_index = edge_index[:, edge_index[0] <= max_index]
        edge_index = edge_index[:, edge_index[1] <= max_index]
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.time_intervals = time_intervals
        data.squareID = squareID  
        data_list.append(data)
    
    return data_list, internetUsage

def dataTraining(data_list, train_ratio=0.8):
    train_size = int(len(data_list) * train_ratio)
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    return train_data, test_data

def main():
    file_path = '/Users/yqmy/Desktop/sms-call-internet-mi-2014-01-01.txt'
    output_dir = '/Users/yqmy/Desktop/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    dataset = Dataset(file_path, max_rows=10000)
    graph_data, internetUsage = Graph(dataset)
    if len(graph_data) == 0:
        print("Not enough data to create graph.")
        return
    
    train_data, test_data = dataTraining(graph_data)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    look_back = 5
    model = GCN(num_features=look_back, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    MSE = torch.nn.MSELoss()

    for epoch in range(20):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = MSE(out, data.y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for data in test_loader:
            pred = model(data)
            test_preds.append(pred.numpy())
            test_trues.append(data.y.numpy())
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)
    mse = mean_squared_error(test_trues, test_preds)
    print(f'Test MSE: {mse}')

    for data in test_data:
        with torch.no_grad():
            pred = model(data).numpy()
        true_data = data.y.numpy()
        time_intervals = data.time_intervals
        
        if len(true_data) == 0 or len(pred) == 0:
            continue
        
        plt.figure(figsize=(12, 9))
        plt.plot(time_intervals, internetUsage.inverse_transform(true_data.reshape(-1, 1)), label='True Data')
        plt.plot(time_intervals, internetUsage.inverse_transform(pred.reshape(-1, 1)), label='Predicted Data')
        plt.xlabel('Date')
        plt.ylabel('Internet Usage')
        plt.legend()
        plt.xticks(rotation=90)
        plt.title(f'SquareID: {data.squareID}')
        plt.show()
        plt.close()

    end_time = time.time()
    print(f'Total time taken: {end_time - start_time} seconds')

if __name__ == "__main__":
    main()


import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class HARDataset:
    """Human Activity Recognition dataset"""
    
    def __init__(self, num_clients=30, downsample_rate=0.2, downsample_clients=None, 
                 task_index_path='data/task_index.npy'):
        self.num_clients = num_clients
        self.downsample_rate = downsample_rate
        self.downsample_clients = downsample_clients
        self.task_index_path = task_index_path
        self.input_size = None
        self.num_classes = None
    
    def prepare_data(self):
        """Prepare HAR data for federated learning"""
        client_train_datasets = []
        client_test_datasets = []
        
        # Load dataset
        print("Loading HAR dataset...")
        X, y = fetch_openml('har', version=1, return_X_y=True, as_frame=True)
        
        # Preprocess labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Convert to PyTorch tensors
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y_encoded, dtype=torch.long)
        
        # Load task index for client assignment
        if os.path.exists(self.task_index_path):
            task_index = np.load(self.task_index_path)
        else:
            # If task_index.npy doesn't exist, create a synthetic one
            print(f"Warning: {self.task_index_path} not found. Creating synthetic client assignment.")
            task_index = np.repeat(np.arange(1, self.num_clients + 1), len(X) // self.num_clients)
            if len(task_index) < len(X):
                task_index = np.concatenate([task_index, np.ones(len(X) - len(task_index)) * self.num_clients])
            task_index = task_index[:len(X)].astype(int)
        
        self.num_clients = len(np.unique(task_index))
        self.input_size = X.shape[1]
        self.num_classes = len(np.unique(y_encoded))
        
        # Decide which clients to downsample
        if self.downsample_clients is None:
            self.downsample_clients = np.random.choice(
                self.num_clients, size=int(self.num_clients/2), replace=False
            )
        
        # Split data by client
        X_split = []
        y_split = []
        
        for i in range(self.num_clients):
            index = np.where(task_index == i+1)[0]
            if len(index) == 0:
                # Handle case where client has no data
                print(f"Warning: Client {i} has no data")
                X_client = torch.empty(0, self.input_size)
                y_client = torch.empty(0, dtype=torch.long)
            else:
                X_client = X[index]
                y_client = y[index]
            
            # Downsample if this client is in the downsample list
            if i in self.downsample_clients and len(X_client) > 0:
                downsample_idx = np.random.choice(
                    len(X_client), 
                    size=max(1, int(len(X_client) * self.downsample_rate)), 
                    replace=False
                )
                X_client = X_client[downsample_idx]
                y_client = y_client[downsample_idx]
            
            X_split.append(X_client)
            y_split.append(y_client)
        
        # Create train/test splits for each client
        for i, (X_client, y_client) in enumerate(zip(X_split, y_split)):
            if len(X_client) < 4:  # Need at least 4 samples for train/test split
                print(f"Warning: Client {i} has only {len(X_client)} samples. Using all for training.")
                train_dataset = TensorDataset(X_client, y_client)
                test_dataset = TensorDataset(X_client[:0], y_client[:0])  # Empty test set
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_client, y_client, test_size=0.25, random_state=42
                )
                train_dataset = TensorDataset(X_train, y_train)
                test_dataset = TensorDataset(X_test, y_test)
            
            client_train_datasets.append(train_dataset)
            client_test_datasets.append(test_dataset)
            
            print(f"Client {i}: Train samples = {len(train_dataset)}, "
                  f"Test samples = {len(test_dataset)}")
        
        return client_train_datasets, client_test_datasets
    
    def get_data_info(self):
        """Return dataset information"""
        return {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'num_clients': self.num_clients
        }

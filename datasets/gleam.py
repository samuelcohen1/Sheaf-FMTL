import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io
import os

class GLEAMDataset:
    """Google Glass (GLEAM) dataset for eating detection"""
    
    def __init__(self, data_path='data/gleam.mat', downsample_rate=0.2,
                 bias=False, density=1.0, standardize=False, seed=42):
        self.data_path = data_path
        self.downsample_rate = downsample_rate
        self.bias = bias
        self.density = density
        self.standardize = standardize
        self.seed = seed
        self.num_clients = None
        self.input_size = None
        self.num_classes = 2  # Binary classification (eating/not eating)
    
    def prepare_data(self):
        """Prepare GLEAM data for federated learning"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"GLEAM dataset not found at {self.data_path}")
        
        print("Loading GLEAM dataset...")
        mat = scipy.io.loadmat(self.data_path)
        raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y in {-1, 1}
        
        self.num_clients = len(raw_x)
        print(f'Google Glass (GLEAM) dataset:')
        print(f'Number of clients: {self.num_clients}')
        print(f'Number of examples per client: {[len(raw_x[i]) for i in range(len(raw_x))]}')
        print(f'Number of features: {len(raw_x[0][0])}')
        print(f'Keeping {self.density * 100:.2f}% of training data on each client')
        
        # Randomly select half of the clients for downsampling
        downsample_clients = np.random.choice(
            self.num_clients, size=int(self.num_clients/2), replace=False
        )
        
        client_train_datasets = []
        client_test_datasets = []
        
        for i in range(self.num_clients):
            # Extract features and labels
            features, label = raw_x[i], raw_y[i].flatten()
            # Convert labels from {-1, 1} to {0, 1}
            label[label == -1] = 0
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Downsampling if the client is selected
            if i in downsample_clients and len(features_tensor) > 0:
                downsample_idx = np.random.choice(
                    len(features_tensor), 
                    size=max(1, int(len(features_tensor) * self.downsample_rate)), 
                    replace=False
                )
                features_tensor = features_tensor[downsample_idx]
                label_tensor = label_tensor[downsample_idx]
            
            # Train/test split
            if len(features_tensor) < 4:
                print(f"Warning: Client {i} has only {len(features_tensor)} samples.")
                x_train, x_test = features_tensor, features_tensor[:0]
                y_train, y_test = label_tensor, label_tensor[:0]
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    features_tensor, label_tensor, test_size=0.25, random_state=self.seed
                )
            
            # Apply density reduction to training data
            if self.density != 1 and len(x_train) > 0:
                num_train_examples = max(1, int(self.density * len(x_train)))
                train_mask = np.random.permutation(len(x_train))[:num_train_examples]
                x_train = x_train[train_mask]
                y_train = y_train[train_mask]
            
            # Standardize if requested
            if self.standardize and len(x_train) > 1:
                scaler = StandardScaler()
                x_train_np = x_train.numpy()
                x_train_np = scaler.fit_transform(x_train_np)
                x_train = torch.tensor(x_train_np, dtype=torch.float32)
                
                if len(x_test) > 0:
                    x_test_np = x_test.numpy()
                    x_test_np = scaler.transform(x_test_np)
                    x_test = torch.tensor(x_test_np, dtype=torch.float32)
            
            # Add bias term if requested
            if self.bias:
                if len(x_train) > 0:
                    x_train = torch.cat([x_train, torch.ones(len(x_train), 1)], dim=1)
                if len(x_test) > 0:
                    x_test = torch.cat([x_test, torch.ones(len(x_test), 1)], dim=1)
            
            # Create datasets
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            
            client_train_datasets.append(train_dataset)
            client_test_datasets.append(test_dataset)
            
            print(f"Client {i}: Train samples = {len(train_dataset)}, "
                  f"Test samples = {len(test_dataset)}")
        
        # Set input size based on first client's data
        if len(client_train_datasets[0]) > 0:
            self.input_size = client_train_datasets[0][0][0].shape[0]
        else:
            self.input_size = 180 + (1 if self.bias else 0)
        
        return client_train_datasets, client_test_datasets
    
    def get_data_info(self):
        """Return dataset information"""
        return {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'num_clients': self.num_clients
        }

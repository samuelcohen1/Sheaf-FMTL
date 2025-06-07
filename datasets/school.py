import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io
import os

class SchoolDataset:
    """School dataset for exam score prediction"""
    
    def __init__(self, data_path='data/school.mat', bias=True, 
                 standardize=True, seed=42):
        self.data_path = data_path
        self.bias = bias
        self.standardize = standardize
        self.seed = seed
        self.num_clients = None
        self.input_size = None
        self.output_size = 1  # Regression task
    
    def prepare_data(self):
        """Prepare School data for federated learning"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"School dataset not found at {self.data_path}")
        
        print("Loading School dataset...")
        mat = scipy.io.loadmat(self.data_path)
        raw_x, raw_y = mat['X'][0], mat['Y'][0]
        
        self.num_clients = len(raw_x)
        print(f'School dataset:')
        print(f'Number of schools (clients): {self.num_clients}')
        print(f'Number of students per school: {[len(raw_x[i]) for i in range(len(raw_x))]}')
        print(f'Number of features: {raw_x[0].shape[1] if len(raw_x[0].shape) > 1 else 1}')
        
        client_train_datasets = []
        client_test_datasets = []
        
        # Hardcoded statistics from the dataset
        min_y, max_y = 1, 70
        
        for i in range(self.num_clients):
            # Extract features and labels
            features, label = raw_x[i], raw_y[i].flatten()
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            # Train/test split
            if len(features_tensor) < 4:
                print(f"Warning: School {i} has only {len(features_tensor)} students.")
                x_train, x_test = features_tensor, features_tensor[:0]
                y_train, y_test = label_tensor, label_tensor[:0]
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    features_tensor, label_tensor, test_size=0.25, random_state=self.seed
                )
            
            # Standardize features if requested
            if self.standardize and len(x_train) > 1:
                scaler = StandardScaler()
                x_train_np = x_train.numpy()
                x_train_np = scaler.fit_transform(x_train_np)
                x_train = torch.tensor(x_train_np, dtype=torch.float32)
                
                if len(x_test) > 0:
                    x_test_np = x_test.numpy()
                    x_test_np = scaler.transform(x_test_np)
                    x_test = torch.tensor(x_test_np, dtype=torch.float32)
                
                # Normalize scores to [0, 1]
                y_train = (y_train - min_y) / (max_y - min_y)
                if len(y_test) > 0:
                    y_test = (y_test - min_y) / (max_y - min_y)
            
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
            
            print(f"School {i}: Train samples = {len(train_dataset)}, "
                  f"Test samples = {len(test_dataset)}")
        
        # Set input size based on first client's data
        if len(client_train_datasets[0]) > 0:
            self.input_size = client_train_datasets[0][0][0].shape[0]
        else:
            # Default: 28 features + 1 bias
            self.input_size = 28 + (1 if self.bias else 0)
        
        return client_train_datasets, client_test_datasets
    
    def get_data_info(self):
        """Return dataset information"""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_clients': self.num_clients
        }

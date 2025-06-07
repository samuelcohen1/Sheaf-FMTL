import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import urllib.request
import gzip
import struct

class RotatedMNIST:
    """Rotated MNIST dataset with heterogeneous rotations across clients"""
    
    def __init__(self, num_clients=40, num_rotations=4, data_dir='./data/rotated_mnist'):
        self.num_clients = num_clients
        self.num_rotations = num_rotations
        self.data_dir = data_dir
        
    def prepare_data(self):
        """Download and prepare rotated MNIST data"""
        # Download MNIST if not exists
        self._download_mnist()
        
        # Load and rotate data
        train_images, train_labels, test_images, test_labels = self._load_mnist()
        
        # Apply rotations
        train_images, train_labels = self._apply_rotations(train_images, train_labels)
        test_images, test_labels = self._apply_rotations(test_images, test_labels)
        
        # Split into clients
        client_train_datasets = []
        client_test_datasets = []
        
        train_size = len(train_images) // self.num_clients
        test_size = len(test_images) // self.num_clients
        
        for i in range(self.num_clients):
            # Get client's portion
            train_start = i * train_size
            train_end = (i + 1) * train_size
            test_start = i * test_size
            test_end = (i + 1) * test_size
            
            # Create tensors
            x_train = torch.tensor(train_images[train_start:train_end], dtype=torch.float32)
            y_train = torch.tensor(train_labels[train_start:train_end], dtype=torch.long)
            x_test = torch.tensor(test_images[test_start:test_end], dtype=torch.float32)
            y_test = torch.tensor(test_labels[test_start:test_end], dtype=torch.long)
            
            # Reshape for CNN (add channel dimension)
            x_train = x_train.unsqueeze(1)
            x_test = x_test.unsqueeze(1)
            
            # Create datasets
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            
            client_train_datasets.append(train_dataset)
            client_test_datasets.append(test_dataset)
        
        return client_train_datasets, client_test_datasets
    
    def _apply_rotations(self, images, labels):
        """Apply different rotations to different portions of data"""
        n = len(images)
        chunk_size = n // self.num_rotations
        
        for k in range(self.num_rotations):
            start = k * chunk_size
            end = (k + 1) * chunk_size if k < self.num_rotations - 1 else n
            
            # Rotate by k*90 degrees
            images[start:end] = np.rot90(images[start:end], k=k, axes=(1, 2))
        
        return images, labels
    
    def _download_mnist(self):
        """Download MNIST dataset"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(base_url + filename, filepath)
    
    def _load_mnist(self):
        """Load MNIST data from files"""
        def parse_labels(filename):
            with gzip.open(filename, 'rb') as f:
                _ = struct.unpack('>II', f.read(8))
                return np.frombuffer(f.read(), dtype=np.uint8)
        
        def parse_images(filename):
            with gzip.open(filename, 'rb') as f:
                _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
        train_images = parse_images(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz'))
        train_labels = parse_labels(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz'))
        test_images = parse_images(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz'))
        test_labels = parse_labels(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz'))
        
        # Normalize
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        
        # Shuffle
        perm = np.random.permutation(len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        
        return train_images, train_labels, test_images, test_labels

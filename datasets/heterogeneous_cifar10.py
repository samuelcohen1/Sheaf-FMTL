import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import trange
import random

class HeterogeneousCIFAR10:
    """CIFAR-10 dataset with heterogeneous label distribution across clients"""
    
    def __init__(self, num_clients=30, num_labels_per_client=5, data_dir='./data'):
        self.num_clients = num_clients
        self.num_labels = num_labels_per_client
        self.num_classes = 10
        self.data_dir = data_dir
        
        # Data transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def prepare_data(self):
        """Prepare heterogeneous CIFAR-10 data for federated learning"""
        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        
        # Combine train and test data for redistribution
        cifar_data_image = np.concatenate([trainset.data, testset.data])
        cifar_data_label = np.concatenate([np.array(trainset.targets), np.array(testset.targets)])
        
        # Separate data by class
        data_by_class = [cifar_data_image[cifar_data_label == i] for i in range(self.num_classes)]
        
        print(f"Number of samples per class: {[len(v) for v in data_by_class]}")
        
        # Distribute data to clients
        X = [[] for _ in range(self.num_clients)]
        y = [[] for _ in range(self.num_clients)]
        
        # Initial distribution: give each client some samples from selected classes
        idx = np.zeros(self.num_classes, dtype=np.int64)
        
        for client in range(self.num_clients):
            # Select random classes for this client
            selected_classes = np.random.choice(self.num_classes, self.num_labels, replace=False)
            
            for class_id in selected_classes:
                # Give initial samples
                num_initial = 10
                X[client] += data_by_class[class_id][idx[class_id]:idx[class_id]+num_initial].tolist()
                y[client] += [class_id] * num_initial
                idx[class_id] += num_initial
        
        # Distribute remaining samples using power law
        props = np.random.lognormal(0, 2., (self.num_classes, self.num_clients, self.num_labels))
        props = np.array([[[len(v)-self.num_clients]] for v in data_by_class]) * \
                props / np.sum(props, (1, 2), keepdims=True)
        
        for client in trange(self.num_clients, desc="Distributing data to clients"):
            # Get classes for this client
            selected_classes = np.random.choice(self.num_classes, self.num_labels, replace=False)
            
            for j, class_id in enumerate(selected_classes):
                # Calculate number of additional samples
                num_samples = int(props[class_id, client // int(self.num_clients/10), j])
                num_samples = num_samples + random.randint(300, 600)
                
                if self.num_clients <= 20:
                    num_samples = num_samples * 2
                
                # Ensure we don't exceed available data
                if idx[class_id] + num_samples < len(data_by_class[class_id]):
                    X[client] += data_by_class[class_id][idx[class_id]:idx[class_id]+num_samples].tolist()
                    y[client] += [class_id] * num_samples
                    idx[class_id] += num_samples
        
        # Create train/test splits for each client
        client_train_datasets = []
        client_test_datasets = []
        
        for i in range(self.num_clients):
            # Ensure we have enough samples for stratified split
            if len(set(y[i])) > 1 and all(y[i].count(c) >= 2 for c in set(y[i])):
                X_train, X_test, y_train, y_test = train_test_split(
                    X[i], y[i], train_size=0.75, stratify=y[i], random_state=42
                )
            else:
                # Fallback to simple split if stratification not possible
                split_idx = int(0.75 * len(X[i]))
                X_train, X_test = X[i][:split_idx], X[i][split_idx:]
                y_train, y_test = y[i][:split_idx], y[i][split_idx:]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(np.array(X_train)).permute(0, 3, 1, 2) / 255.0
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(np.array(X_test)).permute(0, 3, 1, 2) / 255.0
            y_test_tensor = torch.LongTensor(y_test)
            
            # Apply normalization
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            X_train_tensor = torch.stack([normalize(x) for x in X_train_tensor])
            X_test_tensor = torch.stack([normalize(x) for x in X_test_tensor])
            
            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            client_train_datasets.append(train_dataset)
            client_test_datasets.append(test_dataset)
            
            print(f"Client {i}: Train samples = {len(train_dataset)}, "
                  f"Test samples = {len(test_dataset)}, "
                  f"Classes = {sorted(set(y_train))}")
        
        return client_train_datasets, client_test_datasets

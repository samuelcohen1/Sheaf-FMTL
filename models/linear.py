import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    """Linear regression model"""
    def __init__(self, input_size, output_size=1, bias=True):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, x):
        return self.linear(x)

class MultinomialLogisticRegression(nn.Module):
    """Multinomial logistic regression model"""
    def __init__(self, input_size, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    """Multi-layer perceptron for classification"""
    def __init__(self, input_size, hidden_sizes, num_classes, activation='relu'):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression"""
    def __init__(self, input_size, hidden_sizes, output_size=1, activation='relu'):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def get_model_for_dataset(dataset_name, input_size, num_classes=None, **kwargs):
    """Factory function to get appropriate model for dataset"""
    if dataset_name == 'school':
        # Regression task
        return LinearRegression(input_size, output_size=1, bias=kwargs.get('bias', True))
    
    elif dataset_name in ['har', 'vehicle_sensor', 'gleam']:
        # Classification tasks
        return MultinomialLogisticRegression(input_size, num_classes)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

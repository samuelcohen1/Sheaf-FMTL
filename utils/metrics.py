import torch
import numpy as np
from torch.utils.data import DataLoader

def evaluate_accuracy(model, dataset, batch_size=64):
    """Evaluate accuracy of a single model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data, targets in dataloader:
            # Move data to the same device as model
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total if total > 0 else 0

def evaluate_all_clients(models, test_datasets):
    """Evaluate all client models and return average accuracy"""
    accuracies = []
    for model, dataset in zip(models, test_datasets):
        acc = evaluate_accuracy(model, dataset)
        accuracies.append(acc)
    return np.mean(accuracies), accuracies

def calculate_communication_bits(graph, factor, num_params, bits_per_param=32):
    """Calculate communication bits for one round"""
    total_bits = 0
    max_neighbors = max(len(list(graph.neighbors(node))) for node in graph.nodes())
    total_bits += 2 * max_neighbors * int(factor * num_params) * bits_per_param
    return total_bits

def count_model_parameters(model):
    """Count the number of parameters in a model"""
    return sum(param.numel() for param in model.parameters())

def evaluate_ensemble_clients(client_models, client_test_datasets, graph):
    """
    For each client c, ensemble predictions by averaging softmax outputs
    from client c and its graph neighbors (equal weights).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = next(client_models[0].parameters()).device
    client_accuracies = []

    for c in range(len(client_models)):
        neighbors = list(graph.neighbors(c))
        all_nodes = [c] + neighbors

        loader = DataLoader(client_test_datasets[c], batch_size=64, shuffle=False)
        correct = 0
        total = 0
        num_classes = None

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            if num_classes is None:
                with torch.no_grad():
                    sample_logits = client_models[c](X)
                num_classes = sample_logits.shape[1]

            ensemble_probs = torch.zeros(X.size(0), num_classes, device=device)
            for v in all_nodes:
                with torch.no_grad():
                    logits = client_models[v](X)
                    probs = F.softmax(logits, dim=1)
                ensemble_probs += probs
            ensemble_probs /= len(all_nodes)

            preds = ensemble_probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        client_accuracies.append(correct / total)

    return float(np.mean(client_accuracies)), client_accuracies

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import threading

class SheafFMTL:
    """
    Sheaf-based Federated Multi-Task Learning Algorithm
    
    Args:
        models: List of client models
        graph: NetworkX graph representing client connectivity
        lambda_reg: Regularization parameter
        alpha: Learning rate for model parameters
        eta: Learning rate for restriction maps
        gamma: Factor controlling dimension of interaction space
    """
    
    def __init__(self, models: List[nn.Module], graph, 
                 lambda_reg: float = 0.1, 
                 alpha: float = 0.001,
                 eta: float = 0.001,
                 gamma: float = 0.1):
        self.models = models
        self.graph = graph
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.num_clients = len(models)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        for m in self.models:
            m.to(self.device)
        
        # One CUDA stream per client for parallel local updates
        if self.device.type == "cuda":
            self.streams = [torch.cuda.Stream() for _ in range(self.num_clients)]
        else:
            self.streams = None

        self.P = self._initialize_restriction_maps()
        
    def _initialize_restriction_maps(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Initialize restriction maps P_ij for all edges in the graph"""
        P = {}
        
        for i, j in self.graph.edges():
            num_params_i = sum(p.numel() for p in self.models[i].parameters())
            num_params_j = sum(p.numel() for p in self.models[j].parameters())
            
            d_ij = int(self.gamma * min(num_params_i, num_params_j))
            
            P[(i, j)] = (torch.randn(d_ij, num_params_i, device=self.device) * 0.01)
            P[(j, i)] = (torch.randn(d_ij, num_params_j, device=self.device) * 0.01)
            
        return P
    
    def local_update(self, client_id: int, dataloader, 
                    local_epochs: int = 1, l2_strength: float = 0.01):
        """Perform local training on client's data"""
        model = self.models[client_id]
        optimizer = torch.optim.SGD(model.parameters(), lr=self.alpha)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(local_epochs):
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def local_update_all_parallel(self, client_dataloaders: List, local_epochs: int = 1):
        """Run all clients' local updates in parallel using threads + CUDA streams"""

        def worker(client_id):
            if self.streams is not None:
                with torch.cuda.stream(self.streams[client_id]):
                    self.local_update(client_id, client_dataloaders[client_id], local_epochs)
            else:
                self.local_update(client_id, client_dataloaders[client_id], local_epochs)

        threads = [threading.Thread(target=worker, args=(cid,))
                   for cid in range(self.num_clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Ensure all GPU work is complete before sheaf updates read neighbor models
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def sheaf_update(self, client_id: int):
        """Update client model using sheaf Laplacian regularization"""
        with torch.no_grad():
            theta_i = self._get_model_params(client_id)
            
            sum_P_terms = torch.zeros_like(theta_i)
            
            for j in self.graph.neighbors(client_id):
                P_ij = self.P[(client_id, j)]
                P_ji = self.P[(j, client_id)]
                theta_j = self._get_model_params(j)
                
                sum_P_terms += P_ij.T @ (P_ij @ theta_i - P_ji @ theta_j)
            
            theta_i -= self.alpha * self.lambda_reg * sum_P_terms
            
            self._set_model_params(client_id, theta_i)
    
    def update_restriction_maps(self, client_id: int):
        """Update restriction maps P_ij"""
        with torch.no_grad():
            theta_i = self._get_model_params(client_id)
            
            for j in self.graph.neighbors(client_id):
                P_ij = self.P[(client_id, j)]
                P_ji = self.P[(j, client_id)]
                theta_j = self._get_model_params(j)
                
                diff = P_ij @ theta_i - P_ji @ theta_j
                self.P[(client_id, j)] -= self.eta * self.lambda_reg * torch.outer(diff, theta_i)
    
    def _get_model_params(self, client_id: int) -> torch.Tensor:
        """Extract model parameters as a flattened vector"""
        return torch.cat([param.view(-1) for param in self.models[client_id].parameters()])
    
    def _set_model_params(self, client_id: int, params: torch.Tensor):
        """Set model parameters from a flattened vector"""
        idx = 0
        for param in self.models[client_id].parameters():
            numel = param.numel()
            param.data.copy_(params[idx:idx+numel].reshape(param.size()))
            idx += numel
    
    def train(self, client_dataloaders: List, num_rounds: int = 100, 
              local_epochs: int = 1, evaluate_fn: Optional = None):
        """Main training loop"""
        history = {'accuracy': [], 'loss': [], 'communication_bits': []}
        cumulative_bits = 0
        
        for round_idx in range(num_rounds):
            for client_id in range(self.num_clients):
                self.local_update(client_id, client_dataloaders[client_id], 
                                local_epochs=local_epochs)
                self.sheaf_update(client_id)
                self.update_restriction_maps(client_id)
            
            round_bits = self._calculate_communication_bits()
            cumulative_bits += round_bits
            
            if evaluate_fn is not None:
                accuracy = evaluate_fn(self.models)
                history['accuracy'].append(accuracy)
                history['communication_bits'].append(cumulative_bits)
                
                if round_idx % 10 == 0:
                    print(f"Round {round_idx}: Avg Accuracy = {accuracy:.4f}")
        
        return history
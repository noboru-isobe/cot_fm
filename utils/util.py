import torch

class StandardScaler:
    
    def __init__(self, mean=None, std=None, eps=1e-7):
        self.mean = mean
        self.std = std
        self.eps = eps
        
    def fit(self, data):
        # data: [batch, ...]
        self.mean = torch.mean(data, axis=0)
        self.std = torch.std(data, axis=0)
        
    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)
    
    def inverse_transform(self, data):
        return (self.std + self.eps) * data + self.mean
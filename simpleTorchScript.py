import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t= torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
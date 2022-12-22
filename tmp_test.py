import torch
from torch import nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5,5)
 
    def forward(self,x):
        net = self.lin1(x)
        return net
 
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

x = torch.ones(5, device=device)
y = x * 2

model = Net()
model.to(device)
pred = model(x)
print(pred)
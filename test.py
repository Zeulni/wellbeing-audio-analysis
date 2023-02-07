import torch
import math

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

x = torch.rand(1, 3, 300, 300).to('mps')
print(x)
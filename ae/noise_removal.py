import torch
import torch.nn as nn
import torch.optim as optim

# data (flattened)
clean = torch.tensor([[1,1,1, 1,0,1, 1,1,1]], dtype=torch.float32)
noisy = torch.tensor([[1,0,1, 1,1,1, 1,0,1]], dtype=torch.float32)

# model (9 → 4 → 9)
model = nn.Sequential(
    nn.Linear(9, 4),
    nn.Sigmoid(),
    nn.Linear(4, 9),
    nn.Sigmoid()
)

criterion = nn.MSELoss()

def train(lr):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(10):
        output = model(noisy)
        loss = criterion(output, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"LR {lr} Epoch {epoch} Loss {loss.item()}")

# train
train(0.001)
train(0.01)

# denoise
with torch.no_grad():
    out = model(noisy)

print("\nNoisy:", noisy.numpy().reshape(3,3))
print("Denoised:", out.numpy().round().reshape(3,3))
print("Clean:", clean.numpy().reshape(3,3))
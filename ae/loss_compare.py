from noise_removal import model,clean,criterion,noisy
import torch.nn as nn
import torch.optim as optim
import torch

# loss before training
with torch.no_grad():
    out_before = model(noisy)
    loss_before = nn.MSELoss()(out_before, clean)

# retrain (fresh optimizer)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    output = model(noisy)
    loss = criterion(output, clean)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# loss after
with torch.no_grad():
    out_after = model(noisy)
    loss_after = nn.MSELoss()(out_after, clean)

print("\nLoss Before:", loss_before.item())
print("Loss After:", loss_after.item())
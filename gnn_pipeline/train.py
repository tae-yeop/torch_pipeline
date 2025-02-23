def train():
    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step() # Update parameters based on gradients
        optimizer.zero_grad() # Clear gradients
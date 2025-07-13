import torch
from topox.models import SimplicialComplexNetwork

def train_snn(X, y, simplicial_complex, epochs=100):
    model = SimplicialComplexNetwork(input_dim=X.shape[1], hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, simplicial_complex)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

    return model
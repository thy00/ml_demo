import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model = 128, nhead = 2, num_layers = 2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.out = nn.Linear(d_model, 1)
    
    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src) * math.sqrt(self.d_model)
        output = self.transformer(src)
        output = self.out(output[:, -1, :])
        return output
    
model = SimpleTransformer(input_dim = 5) # 5个特征
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

# mock data
src = torch.randn(32, 10, 5)
target = torch.randn(32 ,1)

losses = []

for epoch in range(100):
    output = model(src)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch}, loss: {loss.item():.4f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
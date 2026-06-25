from torch import nn
class DetectorHead(nn.Module):
  def __init__(self, input_dim=1024, hidden_dim=512):
    super().__init__()
    self.output_dim = 4

    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
    self.norm2 = nn.LayerNorm(hidden_dim//2)
    self.fc3 = nn.Linear(hidden_dim//2, self.output_dim)

    self.gelu = nn.GELU()
    self.sigmoid = nn.Sigmoid()
    self.drop = nn.Dropout(0.1)

  def forward(self, x):
    x = self.drop(self.gelu(self.norm1(self.fc1(x))))
    x = self.drop(self.gelu(self.norm2(self.fc2(x))))
    x = self.sigmoid(self.fc3(x))
    return x
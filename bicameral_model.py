import torch
import torch.nn as nn

class RightNet(nn.Module):
    def __init__(self, input_dim, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class LeftNet(nn.Module):
    def __init__(self, input_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, A, B):
        Q = self.query(A)
        K = self.key(B)
        V = self.value(B)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (A.size(-1)**0.5), dim=-1)
        return attn @ V

class BicameralModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.right = RightNet(input_dim)
        self.left = LeftNet(input_dim)
        self.cross = CrossAttention(64)
        self.fusion = nn.Linear(128, 64)

    def forward(self, x):
        r = self.right(x)
        l = self.left(x)
        r_cross = self.cross(r.unsqueeze(1), l.unsqueeze(1)).squeeze(1)
        l_cross = self.cross(l.unsqueeze(1), r.unsqueeze(1)).squeeze(1)
        combined = torch.cat([r_cross, l_cross], dim=-1)
        return self.fusion(combined)

if __name__ == "__main__":
    model = BicameralModel(10)
    print(model(torch.randn(1,10)))

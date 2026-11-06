import torch
import torch.nn as nn

class StandardResNet(nn.Module):
    """Standard Neural Flow Map: z_{k+1} = z_k + dt * MLP(z_k). Violates det(J) = 1."""
    def __init__(self, dt=0.15):
        super().__init__()
        self.dt = dt
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )
    def forward(self, z):
        return z + self.dt * self.net(z) - 0.0012 * z

class ExactSympNet(nn.Module):
    """Course 18 Architecture: Composed Hamiltonian Shear Maps. Guarantees det(dPhi/dz) == 1."""
    def __init__(self, dt=0.15, n_layers=3):
        super().__init__()
        self.dt = dt / n_layers
        self.n_layers = n_layers
        self.V_net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.T_net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def grad_V(self, q):
        q_req = q.detach().requires_grad_(True) if not q.requires_grad else q
        V = self.V_net(q_req).sum()
        return torch.autograd.grad(V, q_req, create_graph=True)[0]

    def grad_T(self, p):
        p_req = p.detach().requires_grad_(True) if not p.requires_grad else p
        T = self.T_net(p_req).sum()
        return torch.autograd.grad(T, p_req, create_graph=True)[0]

    def forward(self, z):
        q, p = z[:, 0:2], z[:, 2:4]
        h = self.dt
        for _ in range(self.n_layers):
            p = p - 0.5 * h * self.grad_V(q)
            q = q + h * self.grad_T(p)
            p = p - 0.5 * h * self.grad_V(q)
        return torch.cat([q, p], dim=1)
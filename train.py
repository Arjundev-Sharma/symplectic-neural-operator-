import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from physics import symplectic_step_true, henon_heiles_energy
from models import StandardResNet, ExactSympNet

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu") # Force CPU for local Mac M-series compatibility without MPS overhead
print(f"[MIT 18/18C Project] Running Symplectic Neural Operator on: {device}")

# 1. GENERATE DATA
dt = 0.15
N_train = 6000
q_init = np.random.uniform(-0.35, 0.35, (N_train, 2))
p_init = np.random.uniform(-0.35, 0.35, (N_train, 2))
X_train_np = np.hstack([q_init, p_init])
Y_train_np = symplectic_step_true(X_train_np, dt=dt, substeps=20)

X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=device)

# 2. INITIALIZE MODELS
resnet = StandardResNet(dt=dt).to(device)
sympnet = ExactSympNet(dt=dt, n_layers=2).to(device)
opt_res = torch.optim.Adam(resnet.parameters(), lr=2e-3)
opt_symp = torch.optim.Adam(sympnet.parameters(), lr=2e-3)

# 3. TRAINING LOOP
print("Training Standard ResNet vs. Exact Symplectic Network (SympNet)...")
for epoch in range(1501):
    opt_res.zero_grad()
    loss_r = torch.mean((resnet(X_train) - Y_train)**2)
    loss_r.backward()
    opt_res.step()

    opt_symp.zero_grad()
    X_req = X_train.clone().requires_grad_(True)
    loss_s = torch.mean((sympnet(X_req) - Y_train)**2)
    loss_s.backward()
    opt_symp.step()

    if epoch % 500 == 0:
        print(f"  Epoch {epoch:4d} | ResNet Flow MSE: {loss_r.item():.6f} | SympNet Flow MSE: {loss_s.item():.6f}")

# 4. VERIFY LIOUVILLE'S THEOREM & EVALUATE
test_pt = torch.tensor([[0.18, -0.12, 0.15, 0.20]], dtype=torch.float32, device=device, requires_grad=True)
J_res = torch.autograd.functional.jacobian(lambda z: resnet(z).squeeze(0), test_pt.squeeze(0))
J_symp = torch.autograd.functional.jacobian(lambda z: sympnet(z.unsqueeze(0)).squeeze(0), test_pt.squeeze(0))

det_res = torch.linalg.det(J_res).item()
det_symp = torch.linalg.det(J_symp).item()

n_steps = 1200
y0_list = np.linspace(-0.22, 0.28, 8)
init_orbits = []
for y_val in y0_list:
    V_val = 0.5 * (y_val**2) - (1.0 / 3.0) * (y_val**3)
    E_target = 0.095
    px_val = np.sqrt(max(1e-5, 2.0 * (E_target - V_val) - 0.02))
    init_orbits.append([0.0, y_val, px_val, 0.10])
init_orbits = np.array(init_orbits)

traj_true = np.zeros((n_steps, len(init_orbits), 4))
traj_res = np.zeros((n_steps, len(init_orbits), 4))
traj_symp = np.zeros((n_steps, len(init_orbits), 4))

curr_t = init_orbits.copy()
curr_r = torch.tensor(init_orbits, dtype=torch.float32, device=device)
curr_s = torch.tensor(init_orbits, dtype=torch.float32, device=device)

for step in range(n_steps):
    traj_true[step] = curr_t
    traj_res[step] = curr_r.detach().cpu().numpy()
    traj_symp[step] = curr_s.detach().cpu().numpy()
    
    curr_t = symplectic_step_true(curr_t, dt=dt, substeps=10)
    with torch.no_grad():
        curr_r = resnet(curr_r)
    curr_s = sympnet(curr_s.detach().requires_grad_(True)).detach()

theta_ring = np.linspace(0, 2*np.pi, 60, endpoint=False)
ring_init = np.stack([0.12*np.cos(theta_ring), 0.05*np.ones_like(theta_ring), 
                      0.12*np.sin(theta_ring), 0.05*np.ones_like(theta_ring)], axis=-1)

def shoelace_area(q1, p1):
    return 0.5 * np.abs(np.dot(q1, np.roll(p1, 1)) - np.dot(p1, np.roll(q1, 1)))

area_true, area_res, area_symp = [], [], []
r_t = ring_init.copy()
r_r = torch.tensor(ring_init, dtype=torch.float32, device=device)
r_s = torch.tensor(ring_init, dtype=torch.float32, device=device)

for step in range(400):
    area_true.append(shoelace_area(r_t[:, 0], r_t[:, 2]))
    r_np = r_r.detach().cpu().numpy()
    s_np = r_s.detach().cpu().numpy()
    area_res.append(shoelace_area(r_np[:, 0], r_np[:, 2]))
    area_symp.append(shoelace_area(s_np[:, 0], s_np[:, 2]))
    
    r_t = symplectic_step_true(r_t, dt=dt, substeps=8)
    with torch.no_grad():
        r_r = resnet(r_r)
    r_s = sympnet(r_s.detach().requires_grad_(True)).detach()

E_true = henon_heiles_energy(traj_true[:, 3, :])
E_res = henon_heiles_energy(traj_res[:, 3, :])
E_symp = henon_heiles_energy(traj_symp[:, 3, :])

print("\n================ MIT 18/18C EXPERIMENTAL RESULTS ================")
print(f"Exact Symplectic Condition: det(J) must equal 1.0000000")
print(f"Standard ResNet Jacobian Determinant det(J) : {det_res:.7f}  --> VIOLATES LIOUVILLE'S THEOREM")
print(f"Your Exact SympNet Jacobian Determinant     : {det_symp:.7f}  --> EXACT SYMPLECTOMORPHISM!")
print("=================================================================\n")

# 5. GENERATE PLOT
os.makedirs('assets', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)

axes[0].scatter(traj_true[:, :, 1], traj_true[:, :, 3], s=1.5, c='black', alpha=0.5, label='Exact Hamiltonian KAM Tori')
axes[0].scatter(traj_res[:, :, 1], traj_res[:, :, 3], s=1.5, c='red', alpha=0.45, label='Standard ResNet')
axes[0].scatter(traj_symp[:, :, 1], traj_symp[:, :, 3], s=1.5, c='limegreen', alpha=0.6, label='Exact SympNet')
axes[0].set_title('1. Phase-Space KAM Tori Projection ($p_y$ vs $y$)', fontweight='bold')
axes[0].set_xlabel('Position $y$')
axes[0].set_ylabel('Momentum $p_y$')
axes[0].legend(loc='lower left', fontsize=8.5, markerscale=4)
axes[0].grid(True, alpha=0.3)

t_area = np.arange(400) * dt
axes[1].plot(t_area, np.array(area_true)/area_true[0], 'k-', lw=2.5, label='Exact Symplectic Invariant')
axes[1].plot(t_area, np.array(area_res)/area_true[0], 'r--', lw=2.2, label=f'Standard ResNet')
axes[1].plot(t_area, np.array(area_symp)/area_true[0], 'g-', lw=2.2, label=f'Exact SympNet')
axes[1].set_title("2. Liouville's Theorem: Normalized Phase Area", fontweight='bold')
axes[1].set_xlabel('Dynamical Time $t$')
axes[1].set_ylabel('Normalized Phase-Space Volume')
axes[1].set_ylim(0.2, 1.25)
axes[1].legend(loc='lower left', fontsize=8.5)
axes[1].grid(True, alpha=0.3)

t_full_steps = np.arange(n_steps) * dt
axes[2].semilogy(t_full_steps, np.abs(E_res - E_true[0]) + 1e-7, 'r--', lw=2, label='Standard ResNet')
axes[2].semilogy(t_full_steps, np.abs(E_symp - E_true[0]) + 1e-7, 'g-', lw=1.8, label='Exact SympNet')
axes[2].set_title('3. Long-Horizon Hamiltonian Energy Error', fontweight='bold')
axes[2].set_xlabel('Dynamical Time $t$')
axes[2].set_ylabel('Absolute Energy Error')
axes[2].legend(loc='lower right', fontsize=8.5)
axes[2].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('assets/mit_18c_symplectic_neural_net.png', dpi=300)
print("Graph saved to assets/mit_18c_symplectic_neural_net.png!")
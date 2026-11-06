import numpy as np

def henon_heiles_grad_V(x, y):
    # V(x,y) = 0.5*(x^2 + y^2) + x^2*y - (1/3)*y^3
    dVdx = x + 2.0 * x * y
    dVdy = y + x**2 - y**2
    return dVdx, dVdy

def henon_heiles_energy(state):
    # state: [..., 4] -> (x, y, px, py)
    x, y, px, py = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    K = 0.5 * (px**2 + py**2)
    V = 0.5 * (x**2 + y**2) + (x**2 * y - (1.0 / 3.0) * y**3)
    return K + V

def symplectic_step_true(state, dt=0.1, substeps=10):
    h = dt / substeps
    x, y, px, py = state[..., 0].copy(), state[..., 1].copy(), state[..., 2].copy(), state[..., 3].copy()
    for _ in range(substeps):
        gx, gy = henon_heiles_grad_V(x, y)
        px -= 0.5 * h * gx
        py -= 0.5 * h * gy
        x += h * px
        y += h * py
        gx, gy = henon_heiles_grad_V(x, y)
        px -= 0.5 * h * gx
        py -= 0.5 * h * gy
    return np.stack([x, y, px, py], axis=-1)
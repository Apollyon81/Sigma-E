import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Faster, coarser runtime parameters for demo
N = 5
mu = 1.0
k_b = 0.5
omega_base = np.linspace(2.0, 3.0, N)
K_base = 0.5
gamma_decay = 1e-4
eta0 = 5e-4
I_th = 0.01
A_target = 0.6
tau_A = 0.5
tau_b = 5.0
beta_meta = 2.0
E_target = 1.0
dt_out = 0.05
T_total = 36.0 # shorter for robust run
noise_amp = 0.01
def sigmoid(x, slope=10.0):
    return 1.0 / (1.0 + np.exp(-slope * x))
def external_input(t):
    s = np.zeros(N)
    # criar N sinais senoidais (um por oscilador)
    freqs = np.linspace(0.6, 0.9, N)
    s += 0.15 * np.sin(freqs * t)
    env = 0.5 * (1.0 + np.sin(0.05 * t))
    s = s * env
    # pulsos transitórios mantidos apenas para os 3 primeiros nós
    if 5.0 < (t % 20.0) < 5.5:
        s[:3] += np.array([0.6, 0.15, 0.0])
    if 12.0 < (t % 20.0) < 12.2:
        s[:3] += np.array([0.0, 0.45, 0.1])
    s += noise_amp * np.random.randn(N)
    return s

state_size = 2*N + N*N + N + N

def sigmaE_dynamics(t, state):
    x = state[0:N]
    y = state[N:2*N]
    w_flat = state[2*N:2*N + N*N]
    w = w_flat.reshape((N, N))
    A = state[2*N + N*N : 2*N + N*N + N]
    b = state[2*N + N*N + N : 2*N + N*N + 2*N]
    E_glob = np.sum(0.5 * (y**2 + (omega_base**2) * x**2))
    eta_eff = eta0 * (1 + beta_meta * sigmoid((E_glob - E_target)/max(1e-6, E_target), slope=3.0))
    S = external_input(t)
    mu_eff = mu * (1.0 + k_b * b)
    dxdt = y
    # vectorized coupling calculation
    # coupling[i] = sum_j K_base * w[i,j] * (x[j] - x[i])
    diff = x.reshape((1, N)) - x.reshape((N, 1)) # shape (N,N): x_j - x_i? currently x_row - x_col -> transpose fix
    # we want (x_j - x_i) for each i row: compute as x - x_i across row: use (x - x[i])
    coupling = np.sum(K_base * w * (x.reshape((1,N)) - x.reshape((N,1))), axis=1)
    dydt = mu_eff * (1.0 - x**2) * y - (omega_base**2) * x + coupling + S
    # memristor updates vectorized
    # compute I_ij matrix
    Xj_minus_Xi = (x.reshape((1, N)) - x.reshape((N, 1))) # row j ? careful: this yields x_j - x_i at [j,i]
    # To have I_ij = K_base * w[i,j] * (x[j] - x[i]) we can use transpose:
    I_mat = K_base * w * (x.reshape((N,1)) - x.reshape((1,N))) # shape (N,N) -> entry [i,j] = x_i - x_j, need x_j - x_i, so negative
    I_mat = -I_mat # now I_mat[i,j] = K * w[i,j] * (x[j] - x[i])
    prog = sigmoid(np.abs(I_mat) - I_th, slope=20.0)
    dwdt = eta_eff * prog * I_mat * (1.0 - w) - gamma_decay * w
    dA_dt = (np.abs(x) - A) / max(1e-6, tau_A)
    db_dt = (A_target - A) / max(1e-6, tau_b)
    deriv = np.concatenate([dxdt, dydt, dwdt.flatten(), dA_dt, db_dt])
    return deriv
rng = np.random.default_rng(12345)
x0 = 0.05 * rng.standard_normal(N)
y0 = 0.05 * rng.standard_normal(N)
w0 = np.clip(0.2 + 0.02 * rng.standard_normal((N, N)), 0.01, 0.9)
A0 = np.abs(x0)
b0 = np.zeros(N)
state0 = np.concatenate([x0, y0, w0.flatten(), A0, b0])
# === INÍCIO DA SUBSTITUIÇÃO (cole a partir daqui) ===
t_out = np.arange(0.0, T_total + 1e-9, dt_out)  # tempos desejados de saída

sol = solve_ivp(fun=sigmaE_dynamics, t_span=(0.0, T_total), y0=state0,
                method='RK45',
                rtol=1e-3, atol=1e-6,
                max_step=0.2,
                dense_output=True)  # <--- ESSA LINHA É A CORREÇÃO OBRIGATÓRIA

# Bloco robusto: funciona com sucesso ou falha
if sol.success:
    print("Solver completed successfully until T_total.")
    X = sol.sol(t_out).T          # agora sol.sol existe sempre quando success=True
    t_eval = t_out
else:
    print("Solver failed:", sol.message)
    print(f"Stopped at t = {sol.t[-1]:.2f}")
    X = sol.y.T
    t_eval = sol.t

# Extração das variáveis
x_t = X[:, 0:N]
y_t = X[:, N:2*N]
w_t = X[:, 2*N:2*N+N*N].reshape(X.shape[0], N, N)
A_t = X[:, 2*N+N*N:2*N+N*N+N]
b_t = X[:, 2*N+N*N+N:2*N+N*N+2*N]

E_glob_t = np.sum(0.5 * (y_t**2 + (omega_base**2) * x_t**2), axis=1)
amp_t = np.sqrt(x_t**2 + y_t**2)
coherence_idx = 1.0 / (1.0 + np.std(amp_t, axis=1))
w_mean = np.mean(w_t, axis=0)

# Ajuste visual
t_max = t_eval[-1]
print(f"Data available until t = {t_max:.2f}")

import matplotlib.gridspec as gridspec
plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3,2)
ax0 = plt.subplot(gs[0,0])
for i in range(N):
    ax0.plot(t_eval, x_t[:, i], label=f'x[{i}]')
ax0.set_title('x (positions)')
ax0.legend()
ax1 = plt.subplot(gs[0,1])
for i in range(N):
    ax1.plot(t_eval, y_t[:, i], label=f'y[{i}]')
ax1.set_title('y (velocities)')
ax1.legend()
ax2 = plt.subplot(gs[1,0])
for i in range(N):
    ax2.plot(t_eval, A_t[:, i], label=f'A[{i}]')
ax2.set_title('Activity A_i (LPF of |x|)')
ax2.legend()
ax3 = plt.subplot(gs[1,1])
for i in range(N):
    ax3.plot(t_eval, b_t[:, i], label=f'b[{i}]')
ax3.set_title('Bias b_i (homeostasis integrator)')
ax3.legend()
ax4 = plt.subplot(gs[2,0])
im = ax4.imshow(w_mean, cmap='viridis', aspect='auto')
ax4.set_title('Average w (memristive coupling) over sim')
plt.colorbar(im, ax=ax4)
ax5 = plt.subplot(gs[2,1])
ax5.plot(t_eval, E_glob_t, label='E_glob proxy')
ax5.plot(t_eval, coherence_idx, label='Coherence idx')
ax5.set_title('Energy and coherence')
ax5.legend()
plt.tight_layout()
plt.show()
print("Final average w matrix:")
print(np.round(w_mean, 4))
print("Final bias b (last timestep):", np.round(b_t[-1], 4))
print("Final activity A (last timestep):", np.round(A_t[-1], 4))
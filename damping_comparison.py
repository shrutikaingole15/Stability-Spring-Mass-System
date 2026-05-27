import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = 1.0
k = 25.0
omega_n = np.sqrt(k / m)
zeta_values = [0.1, 0.3, 0.7, 1.0, 2.0]
labels = ["Underdamped (ζ=0.1)", "Underdamped (ζ=0.3)",
          "Underdamped (ζ=0.7)", "Critically Damped (ζ=1.0)", "Overdamped (ζ=2.0)"]

t_span = (0, 10)
t_eval = np.linspace(0, 10, 2000)
X0 = [0, 0]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for zeta, label in zip(zeta_values, labels):
    c = 2 * zeta * omega_n * m

    def system(t, X):
        x1, x2 = X
        return [x2, (-k / m) * x1 - (c / m) * x2 + (1 / m) * 1.0]

    sol = solve_ivp(system, t_span, X0, t_eval=t_eval)
    axes[0].plot(sol.t, sol.y[0], label=label)
    axes[1].plot(sol.y[0], sol.y[1], label=label)

# Steady state line
axes[0].axhline(1 / k, color='gray', linestyle='--', linewidth=0.8, label=f"Steady state = {1/k:.3f}")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Displacement x(t)")
axes[0].set_title("Step Response — Damping Ratio Comparison")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].set_xlabel("Displacement x")
axes[1].set_ylabel("Velocity ẋ")
axes[1].set_title("Phase Portrait — Damping Ratio Comparison")
axes[1].legend(fontsize=8)
axes[1].grid(True)

plt.tight_layout()
plt.show()

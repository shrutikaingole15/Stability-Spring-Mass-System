import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = 1.0
k = 25.0
zeta = 0.3
omega_n = np.sqrt(k / m)
c = 2 * zeta * omega_n * m
omega_d = omega_n * np.sqrt(1 - zeta**2)  # damped natural frequency

t = np.linspace(0, 10, 2000)

# Analytical solution for underdamped step response (x0=0, v0=0, F=1)
x_ss = 1 / k
A = -x_ss
B = -zeta * omega_n * A / omega_d

x_analytical = x_ss + np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

# Numerical solution
X0 = [0, 0]
def system(t, X):
    x1, x2 = X
    return [x2, (-k / m) * x1 - (c / m) * x2 + (1 / m) * 1.0]

sol = solve_ivp(system, (0, 10), X0, t_eval=t)

# Error
error = np.abs(x_analytical - sol.y[0])

print(f"Damped natural frequency ωd = {omega_d:.4f} rad/s")
print(f"Max absolute error (analytical vs numerical): {np.max(error):.2e}")
print(f"RMS error: {np.sqrt(np.mean(error**2)):.2e}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(t, x_analytical, label="Analytical", linewidth=2)
ax1.plot(sol.t, sol.y[0], '--', label="Numerical (solve_ivp)", linewidth=1.5)
ax1.axhline(1 / k, color='gray', linestyle=':', linewidth=0.8, label=f"Steady state = {1/k:.3f}")
ax1.set_ylabel("Displacement x(t)")
ax1.set_title(f"Analytical vs Numerical — Underdamped Step Response (ζ = {zeta})")
ax1.legend()
ax1.grid(True)

ax2.plot(t, error, color='red')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Absolute Error")
ax2.set_title("Error Between Analytical and Numerical Solution")
ax2.grid(True)

plt.tight_layout()
plt.show()

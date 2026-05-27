import numpy as np
import matplotlib.pyplot as plt

m = 1.0
k = 25.0
zeta_values = [0.1, 0.3, 0.7, 1.0, 2.0]

print("=" * 50)
print("STABILITY ANALYSIS — Eigenvalue / Pole Analysis")
print("=" * 50)

for zeta in zeta_values:
    omega_n = np.sqrt(k / m)
    c = 2 * zeta * omega_n * m

    # State matrix A for x'' + (c/m)x' + (k/m)x = 0
    # [x1']   [  0       1  ] [x1]
    # [x2'] = [-k/m   -c/m  ] [x2]
    A = np.array([[0, 1],
                  [-k / m, -c / m]])

    eigenvalues = np.linalg.eigvals(A)
    stable = all(np.real(ev) < 0 for ev in eigenvalues)

    print(f"\nzeta = {zeta}")
    print(f"  Poles: {eigenvalues[0]:.4f},  {eigenvalues[1]:.4f}")
    print(f"  Real parts: {np.real(eigenvalues[0]):.4f}, {np.real(eigenvalues[1]):.4f}")
    print(f"  Stable: {stable}")

# Pole-zero plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(zeta_values)))

for zeta, color in zip(zeta_values, colors):
    omega_n = np.sqrt(k / m)
    c = 2 * zeta * omega_n * m
    A = np.array([[0, 1], [-k / m, -c / m]])
    eigenvalues = np.linalg.eigvals(A)
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues),
               color=color, s=100, zorder=5, label=f"ζ = {zeta}")

ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel("Real Part")
ax.set_ylabel("Imaginary Part")
ax.set_title("Pole-Zero Plot — Effect of Damping Ratio")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

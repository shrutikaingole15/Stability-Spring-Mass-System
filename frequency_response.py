import numpy as np
import matplotlib.pyplot as plt

m = 1.0
k = 25.0
omega_n = np.sqrt(k / m)
zeta_values = [0.1, 0.3, 0.7, 1.0, 2.0]

omega = np.linspace(0.1, 3 * omega_n, 2000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for zeta in zeta_values:
    c = 2 * zeta * omega_n * m
    # Frequency response H(jw) = 1 / (-mw^2 + jcw + k)
    H = 1.0 / (k - m * omega**2 + 1j * c * omega)
    magnitude_db = 20 * np.log10(np.abs(H))
    phase_deg = np.degrees(np.angle(H))

    ax1.plot(omega, magnitude_db, label=f"ζ = {zeta}")
    ax2.plot(omega, phase_deg, label=f"ζ = {zeta}")

ax1.axvline(omega_n, color='gray', linestyle='--', linewidth=0.8, label=f"ωₙ = {omega_n:.2f} rad/s")
ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("Frequency Response (Bode Plot)")
ax1.legend(fontsize=8)
ax1.grid(True)

ax2.axvline(omega_n, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel("Frequency (rad/s)")
ax2.set_ylabel("Phase (degrees)")
ax2.legend(fontsize=8)
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print resonance peaks
print(f"Natural frequency ωₙ = {omega_n:.4f} rad/s")
for zeta in zeta_values:
    if zeta < 1 / np.sqrt(2):
        omega_r = omega_n * np.sqrt(1 - 2 * zeta**2)
        print(f"  ζ = {zeta}: Resonance peak at ω = {omega_r:.4f} rad/s")
    else:
        print(f"  ζ = {zeta}: No resonance peak (overdamped or critically damped)")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================
# SYSTEM PARAMETERS
# ==============================

m = 1.0
k = 25.0
zeta = 0.3

omega_n = np.sqrt(k/m)
c = 2 * zeta * omega_n * m

print("Natural frequency (omega_n):", omega_n)
print("Damping coefficient (c):", c)
print("Expected steady state (F/k):", 1/k)

# ==============================
# FORCE DEFINITIONS
# ==============================

def step_force(t):
    return 1.0

def zero_force(t):
    return 0.0

# ==============================
# STATE-SPACE MODEL
# ==============================

def system(t, X, force_function):
    x1, x2 = X
    dx1dt = x2
    dx2dt = (-k/m)*x1 - (c/m)*x2 + (1/m)*force_function(t)
    return [dx1dt, dx2dt]

# ==============================
# TIME SETTINGS
# ==============================

t_span = (0, 10)
t_eval = np.linspace(0, 10, 2000)

# ==============================
# STEP RESPONSE (solve_ivp)
# ==============================

X0_step = [0, 0]

sol_step = solve_ivp(
    lambda t, X: system(t, X, step_force),
    t_span,
    X0_step,
    t_eval=t_eval
)

# ==============================
# FREE RESPONSE (solve_ivp)
# ==============================

X0_free = [1, 0]

sol_free = solve_ivp(
    lambda t, X: system(t, X, zero_force),
    t_span,
    X0_free,
    t_eval=t_eval
)

# ==============================
# EULER METHOD (STEP INPUT)
# ==============================

dt = 0.01
t_euler = np.arange(0, 10, dt)

x1_euler = np.zeros(len(t_euler))
x2_euler = np.zeros(len(t_euler))

for i in range(len(t_euler) - 1):
    dx1 = x2_euler[i]
    dx2 = (-k/m)*x1_euler[i] - (c/m)*x2_euler[i] + (1/m)*1.0
    x1_euler[i+1] = x1_euler[i] + dx1 * dt
    x2_euler[i+1] = x2_euler[i] + dx2 * dt

# ==============================
# PLOTTING
# ==============================

# ---- Step Response ----
plt.figure()
plt.subplot(2,1,1)
plt.plot(sol_step.t, sol_step.y[0])
plt.ylabel("Displacement x(t)")
plt.title("Step Response (solve_ivp)")
plt.grid()

plt.subplot(2,1,2)
plt.plot(sol_step.t, sol_step.y[1])
plt.xlabel("Time (s)")
plt.ylabel("Velocity x_dot(t)")
plt.grid()
plt.tight_layout()

# ---- Free Response ----
plt.figure()
plt.subplot(2,1,1)
plt.plot(sol_free.t, sol_free.y[0])
plt.ylabel("Displacement x(t)")
plt.title("Free Response (solve_ivp)")
plt.grid()

plt.subplot(2,1,2)
plt.plot(sol_free.t, sol_free.y[1])
plt.xlabel("Time (s)")
plt.ylabel("Velocity x_dot(t)")
plt.grid()
plt.tight_layout()

# ---- Euler vs solve_ivp Comparison ----
plt.figure()
plt.plot(sol_step.t, sol_step.y[0], label="solve_ivp")
plt.plot(t_euler, x1_euler, '--', label="Euler")
plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("Euler vs solve_ivp Comparison")
plt.legend()
plt.grid()

# ---- Phase Portrait ----
plt.figure()
plt.plot(sol_step.y[0], sol_step.y[1])
plt.xlabel("Displacement x")
plt.ylabel("Velocity x_dot")
plt.title("Phase Portrait (Step Response)")
plt.grid()

plt.show()
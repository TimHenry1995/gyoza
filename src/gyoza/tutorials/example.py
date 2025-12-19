import numpy as np
import matplotlib.pyplot as plt

# --- Define Functions ---

def transform_z_to_ztilde(z_data, sigma_T):
    """Applies the transformation T: z / sigma_T (element-wise)."""
    return z_data / sigma_T[None, :]

def jacobian_determinant(sigma_T):
    """Computes the Jacobian determinant of T (1 / (sigma1 * sigma2))."""
    return 1.0 / (sigma_T[0] * sigma_T[1])

def neg_log_likelihood(z_a_data, z_b_data, sigma_T, y_ab_values):
    """Computes the mean negative log-likelihood loss for the dataset."""
    ztilde_a = transform_z_to_ztilde(z_a_data, sigma_T)
    ztilde_b = transform_z_to_ztilde(z_b_data, sigma_T)
    det_J = jacobian_determinant(sigma_T)
    
    # Log likelihood p(ztilde_a) ~ N(0, I)
    log_p_ztilde_a = -np.log(2 * np.pi) - 0.5 * np.sum(ztilde_a**2, axis=1)
    
    # Log likelihood p(ztilde_b | ztilde_a)
    # Mean mu_b_a = y_ab * ztilde_a, Covariance diag(1 - y_ab^2)
    mu_b_a = y_ab_values[None, :] * ztilde_a
    c_b_a = 1.0 - y_ab_values**2
    det_Sigma_b_a = c_b_a[0] * c_b_a[1]
    diff_b_a = ztilde_b - mu_b_a
    exponent_term_b_a = np.sum(diff_b_a**2 / c_b_a[None, :], axis=1)
    
    log_p_ztilde_b_a = -np.log(2 * np.pi * np.sqrt(det_Sigma_b_a)) - 0.5 * exponent_term_b_a
    
    # Total log likelihood p(z_a, z_b) using change of variables
    log_p_za_zb = log_p_ztilde_a + log_p_ztilde_b_a + 2 * np.log(np.abs(det_J))
    
    # Negative mean log likelihood (loss)
    nll = -np.mean(log_p_za_zb)
    return nll

# --- Simulation and Plotting Script ---

### Step 1: Generate Synthetic Dataset
# Data is generated to adhere to the model's assumptions with 'true' parameters.
# True sigmas are [2.0, 3.0]. Fixed similarity y_ab for all pairs is [0.8, 0.8].

sigma_true = np.array([2.0, 3.0])
y_ab_fixed = np.array([0.8, 0.8])
N_pairs = 1000

# Generate data in the enforced (tilde) space first
ztilde_a_samples = np.random.randn(N_pairs, 2)
c_b_a = 1.0 - y_ab_fixed**2
ztilde_b_samples = (y_ab_fixed[None, :] * ztilde_a_samples + 
                    np.random.randn(N_pairs, 2) * np.sqrt(c_b_a[None, :]))

# Transform back to the original z space using true sigmas
z_a_samples = ztilde_a_samples * sigma_true[None, :]
z_b_samples = ztilde_b_samples * sigma_true[None, :]



### Step 2: Compute Loss Surface
# Calculate NLL over a range of potential calibration parameters for sigma1 and sigma2.

sigma1_range = np.linspace(1.0, 5.0, 50)
sigma2_range = np.linspace(1.0, 5.0, 50)
nll_surface = np.zeros((len(sigma2_range), len(sigma1_range)))

for i, s1 in enumerate(sigma1_range):
    for j, s2 in enumerate(sigma2_range):
        sigma_T_params = np.array([s1, s2])
        nll_surface[j, i] = neg_log_likelihood(z_a_samples, z_b_samples, sigma_T_params, y_ab_fixed)

# Find minimum NLL location
min_nll_idx = np.unravel_index(np.argmin(nll_surface), nll_surface.shape)
optimal_sigma1 = sigma1_range[min_nll_idx[1]]
optimal_sigma2 = sigma2_range[min_nll_idx[0]]

print(f"Optimal Sigma1 found by grid search: {optimal_sigma1:.2f}")
print(f"Optimal Sigma2 found by grid search: {optimal_sigma2:.2f}")



### Step 3: Plot Loss Function Contour
# The plot visualizes the NLL landscape, highlighting the true and optimal parameters.

plt.figure(figsize=(8, 6))
contour_plot = plt.contourf(sigma1_range, sigma2_range, nll_surface, levels=50, cmap='viridis')
plt.colorbar(contour_plot, label='Negative Log-Likelihood')
plt.contour(sigma1_range, sigma2_range, nll_surface, levels=10, colors='white', linewidths=0.5)
plt.scatter(optimal_sigma1, optimal_sigma2, color='red', marker='*', s=200, label=f'Optimal: ({optimal_sigma1:.2f}, {optimal_sigma2:.2f})')
plt.scatter(sigma_true[0], sigma_true[1], color='cyan', marker='o', s=100, label=f'True: ({sigma_true[0]:.1f}, {sigma_true[1]:.1f})', edgecolors='black')
plt.title('NLL Loss Contour Plot vs Transformation Sigmas')
plt.xlabel(r'$\sigma_{T1}$')
plt.ylabel(r'$\sigma_{T2}$')
plt.legend()
plt.grid(True)
plt.show()


# Combine 'a' and 'b' samples for visualization (illustrating the overall distribution shape)
z_total = np.vstack((z_a_samples, z_b_samples))
# Transform total z data using the *true* sigmas for perfect visualization of z_tilde
ztilde_total = transform_z_to_ztilde(z_total, sigma_true) 

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)

# Plot the original Z distribution
axes[0].scatter(z_total[:, 0], z_total[:, 1], alpha=0.5, s=10)
axes[0].set_title(r'Original Data Distribution $z$')
axes[0].set_xlabel(r'$z_1$ (Std Dev 2.0)')
axes[0].set_ylabel(r'$z_2$ (Std Dev 3.0)')
axes[0].axis('equal')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot the transformed Z_tilde distribution
axes[1].scatter(ztilde_total[:, 0], ztilde_total[:, 1], alpha=0.5, s=10, color='orange')
axes[1].set_title(r'Transformed Data Distribution $\tilde{z}$')
axes[1].set_xlabel(r'$\tilde{z}_1$ (Std Dev 1.0 enforced)')
axes[1].set_ylabel(r'$\tilde{z}_2$ (Std Dev 1.0 enforced)')
axes[1].axis('equal')
axes[1].set_xlim([-4, 4])
axes[1].set_ylim([-4, 4])
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
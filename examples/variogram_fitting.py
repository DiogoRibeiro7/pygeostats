# examples/variogram_fitting.py
"""Variogram fitting example with different models."""

import numpy as np
import matplotlib.pyplot as plt
from pygeostats.variogram import EmpiricalVariogram, Variogram

def generate_data_with_known_variogram(n_samples=100, nugget=0.1, sill=1.0, range_param=2.0):
    """Generate data with a known variogram structure."""
    np.random.seed(42)
    coords = np.random.uniform(0, 10, size=(n_samples, 2))
    
    # Calculate distances
    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    
    # Create covariance matrix using exponential model
    # C(h) = (sill - nugget) * exp(-h/range) + nugget * I
    covariance = (sill - nugget) * np.exp(-distances / range_param)
    np.fill_diagonal(covariance, sill)  # C(0) = sill
    
    # Generate values
    values = np.random.multivariate_normal(
        mean=np.zeros(n_samples), cov=covariance
    )
    
    return coords, values

def main():
    """Run variogram fitting example."""
    print("PySpatialStats - Variogram Fitting Example")
    print("=" * 45)
    
    # Generate synthetic data with known parameters
    true_nugget, true_sill, true_range = 0.2, 1.2, 3.0
    coords, values = generate_data_with_known_variogram(
        n_samples=150, nugget=true_nugget, sill=true_sill, range_param=true_range
    )
    
    print(f"Generated data with true parameters:")
    print(f"  Nugget: {true_nugget}")
    print(f"  Sill: {true_sill}")
    print(f"  Range: {true_range}")
    print()
    
    # Compute empirical variogram
    emp_vario = EmpiricalVariogram(coords, values, max_distance=6.0, n_bins=20)
    emp_vario.compute()
    
    # Test different models
    models = ['exponential', 'spherical', 'gaussian']
    fitted_models = {}
    
    print("Fitting different variogram models:")
    print("-" * 35)
    
    for model_name in models:
        vario = Variogram(model=model_name)
        vario.fit(emp_vario.distances_, emp_vario.gamma_)
        fitted_models[model_name] = vario
        
        print(f"{model_name.capitalize()} Model:")
        print(f"  Nugget: {vario.nugget_:.3f} (true: {true_nugget:.3f})")
        print(f"  Sill:   {vario.sill_:.3f} (true: {true_sill:.3f})")
        print(f"  Range:  {vario.range_:.3f} (true: {true_range:.3f})")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Empirical data and sample locations
    ax = axes[0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=50)
    plt.colorbar(scatter, ax=ax, label='Value')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Sample Data Locations')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Variogram models comparison
    ax = axes[1]
    
    # Plot empirical variogram
    valid_mask = emp_vario.counts_ > 0
    distances_emp = emp_vario.distances_[valid_mask]
    gamma_emp = emp_vario.gamma_[valid_mask]
    counts = emp_vario.counts_[valid_mask]
    
    # Size points by number of pairs
    sizes = 30 + 100 * counts / np.max(counts)
    ax.scatter(distances_emp, gamma_emp, s=sizes, alpha=0.6, 
               color='black', label='Empirical', zorder=5)
    
    # Plot fitted models
    distances_smooth = np.linspace(0, 6, 200)
    colors = ['red', 'blue', 'green']
    
    for i, (model_name, vario) in enumerate(fitted_models.items()):
        fitted_gamma = vario.predict(distances_smooth)
        ax.plot(distances_smooth, fitted_gamma, color=colors[i], 
                linewidth=2.5, label=f'{model_name.capitalize()}', alpha=0.8)
    
    # Add horizontal lines for nugget and sill
    ax.axhline(y=true_sill, color='gray', linestyle='--', alpha=0.7, label='True Sill')
    ax.axhline(y=true_nugget, color='gray', linestyle=':', alpha=0.7, label='True Nugget')
    
    ax.set_xlabel('Distance')
    ax.set_ylabel('Semivariance (γ)')
    ax.set_title('Variogram Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.5 * true_sill, np.max(gamma_emp) * 1.2))
    
    plt.tight_layout()
    plt.savefig('variogram_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Model evaluation
    print("Model Evaluation:")
    print("-" * 17)
    
    for model_name, vario in fitted_models.items():
        predictions = vario.predict(distances_emp)
        rmse = np.sqrt(np.mean((predictions - gamma_emp)**2))
        mae = np.mean(np.abs(predictions - gamma_emp))
        
        print(f"{model_name.capitalize()}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print()
    
    print("Example completed! Plots saved as 'variogram_comparison.png'")

if __name__ == "__main__":
    main()

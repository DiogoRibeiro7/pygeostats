# examples/basic_kriging.py
"""Basic kriging example."""

import numpy as np
import matplotlib.pyplot as plt
from pygeostats.variogram import EmpiricalVariogram, Variogram
from pygeostats.kriging import OrdinaryKriging

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=100):
    """Generate synthetic spatial data."""
    # Random coordinates in a 10x10 grid
    coords = np.random.uniform(0, 10, size=(n_samples, 2))
    
    # Generate spatially correlated values using exponential covariance
    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    # Exponential covariance: C(h) = σ² * exp(-h/r)
    covariance = 1.0 * np.exp(-distances / 2.0)
    
    # Add nugget effect
    np.fill_diagonal(covariance, covariance.diagonal() + 0.1)
    
    # Generate correlated random values
    values = np.random.multivariate_normal(
        mean=np.zeros(n_samples), cov=covariance
    )
    
    return coords, values

def main():
    """Run basic kriging example."""
    print("PySpatialStats - Basic Kriging Example")
    print("=" * 40)
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    known_coords, known_values = generate_sample_data(80)
    print(f"   Generated {len(known_coords)} sample points")
    
    # 2. Compute empirical variogram
    print("2. Computing empirical variogram...")
    emp_vario = EmpiricalVariogram(
        known_coords, known_values, 
        max_distance=5.0, n_bins=15
    )
    emp_vario.compute()
    print(f"   Computed variogram with {emp_vario.n_bins} bins")
    
    # 3. Fit theoretical variogram
    print("3. Fitting theoretical variogram...")
    vario_model = Variogram(model='exponential')
    vario_model.fit(emp_vario.distances_, emp_vario.gamma_)
    
    print(f"   Fitted parameters:")
    print(f"   - Nugget: {vario_model.nugget_:.3f}")
    print(f"   - Sill: {vario_model.sill_:.3f}")
    print(f"   - Range: {vario_model.range_:.3f}")
    
    # 4. Set up kriging
    print("4. Setting up ordinary kriging...")
    kriging = OrdinaryKriging(vario_model)
    kriging.fit(known_coords, known_values)
    
    # 5. Create prediction grid
    print("5. Creating prediction grid...")
    x_pred = np.linspace(0, 10, 50)
    y_pred = np.linspace(0, 10, 50)
    xx, yy = np.meshgrid(x_pred, y_pred)
    pred_coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 6. Make predictions
    print("6. Making predictions...")
    predictions, variances = kriging.predict(
        pred_coords, return_variance=True
    )
    
    # Reshape for plotting
    pred_grid = predictions.reshape(50, 50)
    var_grid = variances.reshape(50, 50)
    
    print(f"   Made predictions at {len(predictions)} locations")
    print(f"   Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    print(f"   Mean kriging variance: {np.mean(variances):.3f}")
    
    # 7. Create plots
    print("7. Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot empirical vs fitted variogram
    ax = axes[0, 0]
    valid_mask = emp_vario.counts_ > 0
    ax.scatter(
        emp_vario.distances_[valid_mask], 
        emp_vario.gamma_[valid_mask],
        s=50, alpha=0.7, label='Empirical', color='blue'
    )
    
    distances_smooth = np.linspace(0, 5, 100)
    fitted_gamma = vario_model.predict(distances_smooth)
    ax.plot(distances_smooth, fitted_gamma, 'r-', label='Fitted', linewidth=2)
    
    ax.set_xlabel('Distance')
    ax.set_ylabel('Semivariance')
    ax.set_title('Variogram Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot sample data
    ax = axes[0, 1]
    scatter = ax.scatter(
        known_coords[:, 0], known_coords[:, 1], 
        c=known_values, cmap='viridis', s=50
    )
    plt.colorbar(scatter, ax=ax, label='Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Sample Data')
    ax.set_aspect('equal')
    
    # Plot kriging predictions
    ax = axes[1, 0]
    im = ax.imshow(
        pred_grid, extent=[0, 10, 0, 10], 
        origin='lower', cmap='viridis', aspect='equal'
    )
    ax.scatter(
        known_coords[:, 0], known_coords[:, 1], 
        c='white', s=20, alpha=0.8, edgecolors='black', linewidth=0.5
    )
    plt.colorbar(im, ax=ax, label='Predicted Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Kriging Predictions')
    
    # Plot kriging variance (uncertainty)
    ax = axes[1, 1]
    im = ax.imshow(
        var_grid, extent=[0, 10, 0, 10], 
        origin='lower', cmap='Reds', aspect='equal'
    )
    ax.scatter(
        known_coords[:, 0], known_coords[:, 1], 
        c='blue', s=20, alpha=0.8
    )
    plt.colorbar(im, ax=ax, label='Kriging Variance')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Prediction Uncertainty')
    
    plt.tight_layout()
    plt.savefig('kriging_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("8. Example completed!")
    print("   Plot saved as 'kriging_example.png'")

if __name__ == "__main__":
    main()

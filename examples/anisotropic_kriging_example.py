# examples/anisotropic_kriging_example.py
"""Comprehensive examples of anisotropic variogram analysis and kriging."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from pyspatialstats.variogram import DirectionalVariogram, Variogram
from pyspatialstats.kriging.anisotropic import (
    AnisotropicKriging,
    create_anisotropic_variogram_from_directional,
)
from pyspatialstats.kriging import OrdinaryKriging  # For comparison


def generate_anisotropic_data(
    n_samples=100,
    range_major=3.0,
    range_minor=1.0,
    rotation_angle=45.0,
    nugget=0.1,
    sill=1.0,
    domain_size=10.0,
    random_seed=42,
):
    """
    Generate synthetic data with known anisotropic spatial correlation.

    Parameters
    ----------
    n_samples : int
        Number of sample points
    range_major : float
        Range in major anisotropy direction
    range_minor : float
        Range in minor anisotropy direction
    rotation_angle : float
        Rotation angle in degrees
    nugget : float
        Nugget effect
    sill : float
        Total variance (sill)
    domain_size : float
        Size of sampling domain
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    coords : ndarray
        Sample coordinates
    values : ndarray
        Sample values with anisotropic correlation
    true_params : dict
        True anisotropic parameters used for generation
    """
    np.random.seed(random_seed)

    # Generate random coordinates
    coords = np.random.uniform(0, domain_size, size=(n_samples, 2))

    # Convert angle to radians
    theta = np.radians(rotation_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Build anisotropic covariance matrix
    n = len(coords)
    cov_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Vector difference
            delta = coords[j] - coords[i]

            # Rotate to anisotropy axes
            rotated_delta = R @ delta

            # Anisotropic distance
            aniso_dist = np.sqrt(
                (rotated_delta[0] / range_major) ** 2
                + (rotated_delta[1] / range_minor) ** 2
            )

            # Exponential covariance model
            if i == j:
                cov_matrix[i, j] = sill
            else:
                gamma = nugget + (sill - nugget) * (1 - np.exp(-aniso_dist))
                cov_matrix[i, j] = sill - gamma

    # Generate correlated random values
    values = np.random.multivariate_normal(np.zeros(n), cov_matrix)

    true_params = {
        "range_major": range_major,
        "range_minor": range_minor,
        "rotation_angle": rotation_angle,
        "nugget": nugget,
        "sill": sill,
        "anisotropy_ratio": range_major / range_minor,
    }

    return coords, values, true_params


def example_1_basic_anisotropic_analysis():
    """Example 1: Basic anisotropic variogram analysis and kriging."""

    print("Example 1: Basic Anisotropic Analysis")
    print("=" * 50)

    # Generate synthetic anisotropic data
    coords, values, true_params = generate_anisotropic_data(
        n_samples=150,
        range_major=4.0,
        range_minor=1.5,
        rotation_angle=30.0,
        nugget=0.1,
        sill=1.2,
    )

    print(f"Generated data with true parameters:")
    for key, value in true_params.items():
        print(f"  {key}: {value:.3f}")
    print()


def example_3_cross_validation_comparison():
    """Example 3: Cross-validation comparison of anisotropic vs isotropic kriging."""

    print("Example 3: Cross-validation Comparison")
    print("=" * 50)

    # Generate highly anisotropic data
    coords, values, true_params = generate_anisotropic_data(
        n_samples=100,
        range_major=5.0,
        range_minor=1.0,
        rotation_angle=45.0,
        nugget=0.1,
        sill=1.5,
        random_seed=123,
    )

    print(f"Testing with anisotropy ratio: {true_params['anisotropy_ratio']:.2f}")
    print()

    # Leave-one-out cross-validation
    n_samples = len(coords)
    aniso_predictions = np.zeros(n_samples)
    iso_predictions = np.zeros(n_samples)
    aniso_variances = np.zeros(n_samples)

    print("Performing leave-one-out cross-validation...")

    for i in range(n_samples):
        if i % 20 == 0:
            print(f"  Processing sample {i + 1}/{n_samples}")

        # Create training set (leave out sample i)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        train_coords = coords[train_mask]
        train_values = values[train_mask]
        test_coord = coords[i : i + 1]
        test_value = values[i]

        try:
            # Fit directional variogram on training data
            directional_vario = DirectionalVariogram(
                coordinates=train_coords,
                values=train_values,
                directions=[0, 30, 60, 90, 120, 150],
                tolerance=22.5,
                max_distance=4.0,
                n_bins=10,
            )
            directional_vario.compute()

            # Anisotropic kriging
            aniso_variogram = create_anisotropic_variogram_from_directional(
                directional_vario
            )
            aniso_kriging = AnisotropicKriging(aniso_variogram)
            aniso_kriging.fit(train_coords, train_values)
            pred, var = aniso_kriging.predict(test_coord, return_variance=True)
            aniso_predictions[i] = pred[0]
            aniso_variances[i] = var[0]

            # Isotropic kriging for comparison
            avg_range = (
                aniso_variogram.parameters[2] + aniso_variogram.parameters[3]
            ) / 2
            iso_variogram = Variogram(model="exponential")
            iso_variogram.parameters = [
                aniso_variogram.parameters[0],  # nugget
                aniso_variogram.parameters[1],  # sill
                avg_range,  # average range
            ]
            iso_variogram.is_fitted_ = True
            iso_variogram.nugget_ = aniso_variogram.parameters[0]
            iso_variogram.sill_ = aniso_variogram.parameters[1]
            iso_variogram.range_ = avg_range

            iso_kriging = OrdinaryKriging(iso_variogram)
            iso_kriging.fit(train_coords, train_values)
            iso_predictions[i] = iso_kriging.predict(test_coord)[0]

        except Exception as e:
            print(f"  Warning: Failed to process sample {i + 1}: {e}")
            aniso_predictions[i] = np.mean(train_values)
            iso_predictions[i] = np.mean(train_values)
            aniso_variances[i] = np.var(train_values)

    # Compute validation statistics
    aniso_rmse = np.sqrt(mean_squared_error(values, aniso_predictions))
    iso_rmse = np.sqrt(mean_squared_error(values, iso_predictions))
    aniso_r2 = r2_score(values, aniso_predictions)
    iso_r2 = r2_score(values, iso_predictions)

    # Mean kriging variance (uncertainty)
    mean_aniso_var = np.mean(aniso_variances)

    print("\nCross-validation Results:")
    print("-" * 30)
    print(f"Anisotropic Kriging:")
    print(f"  RMSE: {aniso_rmse:.4f}")
    print(f"  R²: {aniso_r2:.4f}")
    print(f"  Mean Variance: {mean_aniso_var:.4f}")
    print()
    print(f"Isotropic Kriging:")
    print(f"  RMSE: {iso_rmse:.4f}")
    print(f"  R²: {iso_r2:.4f}")
    print()
    print(f"Improvement:")
    print(f"  RMSE reduction: {((iso_rmse - aniso_rmse) / iso_rmse * 100):.1f}%")
    print(f"  R² increase: {((aniso_r2 - iso_r2) / iso_r2 * 100):.1f}%")
    print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot: predicted vs observed
    ax = axes[0, 0]
    min_val = min(values.min(), aniso_predictions.min(), iso_predictions.min())
    max_val = max(values.max(), aniso_predictions.max(), iso_predictions.max())

    ax.scatter(
        values,
        aniso_predictions,
        alpha=0.6,
        label=f"Anisotropic (R²={aniso_r2:.3f})",
        color="red",
    )
    ax.scatter(
        values,
        iso_predictions,
        alpha=0.6,
        label=f"Isotropic (R²={iso_r2:.3f})",
        color="blue",
    )
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Cross-validation: Predicted vs Observed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals vs predicted
    ax = axes[0, 1]
    aniso_residuals = values - aniso_predictions
    iso_residuals = values - iso_predictions

    ax.scatter(
        aniso_predictions, aniso_residuals, alpha=0.6, label="Anisotropic", color="red"
    )
    ax.scatter(
        iso_predictions, iso_residuals, alpha=0.6, label="Isotropic", color="blue"
    )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertainty vs absolute error
    ax = axes[1, 0]
    aniso_abs_error = np.abs(aniso_residuals)
    ax.scatter(np.sqrt(aniso_variances), aniso_abs_error, alpha=0.6, color="red")
    ax.set_xlabel("Kriging Standard Deviation")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Uncertainty vs Absolute Error")
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(np.sqrt(aniso_variances), aniso_abs_error, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(
        np.sqrt(aniso_variances).min(), np.sqrt(aniso_variances).max(), 100
    )
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend (slope={z[0]:.2f})")
    ax.legend()

    # Sample locations colored by error
    ax = axes[1, 1]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=aniso_abs_error, cmap="Reds", s=50, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Absolute Error")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Spatial Distribution of Errors")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("anisotropic_cross_validation.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        "Cross-validation completed! Results saved as 'anisotropic_cross_validation.png'"
    )
    print()


def example_4_real_world_workflow():
    """Example 4: Complete workflow for real-world anisotropic analysis."""

    print("Example 4: Real-world Workflow")
    print("=" * 50)

    # Simulate more realistic data with measurement noise
    np.random.seed(456)

    # Generate base anisotropic field
    coords, values, true_params = generate_anisotropic_data(
        n_samples=200,
        range_major=6.0,
        range_minor=2.0,
        rotation_angle=60.0,
        nugget=0.2,
        sill=2.0,
        domain_size=15.0,
    )

    # Add measurement noise
    measurement_noise = np.random.normal(0, 0.1, len(values))
    values = values + measurement_noise

    print(
        f"Dataset: {len(coords)} samples over {coords.max():.0f}x{coords.max():.0f} domain"
    )
    print(f"Value range: [{values.min():.2f}, {values.max():.2f}]")
    print()

    # Step 1: Exploratory data analysis
    print("Step 1: Exploratory Data Analysis")
    print("-" * 35)

    # Basic statistics
    print(f"Mean: {np.mean(values):.3f}")
    print(f"Std: {np.std(values):.3f}")
    print(f"Skewness: {_skewness(values):.3f}")
    print(f"Kurtosis: {_kurtosis(values):.3f}")

    # Check for spatial clustering
    from scipy.spatial.distance import pdist

    distances = pdist(coords)
    print(f"Mean nearest neighbor distance: {np.min(distances):.3f}")
    print()

    # Step 2: Automatic direction detection
    print("Step 2: Automatic Direction Detection")
    print("-" * 40)

    optimal_directions = DirectionalVariogram.automatic_direction_set(
        coords, n_directions=8
    )
    print(f"Optimal directions: {[f'{d:.0f}°' for d in optimal_directions]}")

    # Step 3: Comprehensive directional analysis
    print("Step 3: Directional Variogram Analysis")
    print("-" * 42)

    directional_vario = DirectionalVariogram(
        coordinates=coords,
        values=values,
        directions=optimal_directions,
        tolerance=20.0,
        max_distance=8.0,
        n_bins=20,
    )
    directional_vario.compute()

    # Anisotropy detection with different thresholds
    anisotropy_results = []
    thresholds = [1.2, 1.5, 2.0]

    for threshold in thresholds:
        result = directional_vario.detect_anisotropy(ratio_threshold=threshold)
        anisotropy_results.append(result)
        print(
            f"Threshold {threshold}: Anisotropic = {result.is_anisotropic}, "
            f"Ratio = {result.anisotropy_ratio:.2f}"
        )

    # Use most conservative threshold
    final_anisotropy = anisotropy_results[0]  # threshold = 1.2
    print(f"\nFinal anisotropy assessment:")
    print(f"  Is anisotropic: {final_anisotropy.is_anisotropic}")
    print(f"  Major direction: {final_anisotropy.major_direction:.1f}°")
    print(f"  Anisotropy ratio: {final_anisotropy.anisotropy_ratio:.2f}")
    print()

    # Step 4: Model comparison and selection
    print("Step 4: Model Comparison")
    print("-" * 25)

    models_to_test = ["exponential", "spherical", "gaussian"]
    model_results = {}

    for model in models_to_test:
        try:
            # Create anisotropic variogram
            aniso_variogram = create_anisotropic_variogram_from_directional(
                directional_vario, model=model
            )

            # Fit kriging model
            kriging = AnisotropicKriging(aniso_variogram)
            kriging.fit(coords, values)

            # Simple validation on training data
            predictions = kriging.predict(coords)
            r2 = r2_score(values, predictions)
            rmse = np.sqrt(mean_squared_error(values, predictions))

            model_results[model] = {
                "r2": r2,
                "rmse": rmse,
                "variogram": aniso_variogram,
                "kriging": kriging,
            }

            print(f"  {model.capitalize()}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

        except Exception as e:
            print(f"  {model.capitalize()}: Failed ({str(e)[:50]}...)")

    # Select best model
    if model_results:
        best_model = max(model_results.keys(), key=lambda k: model_results[k]["r2"])
        print(f"\nBest model: {best_model}")
        best_kriging = model_results[best_model]["kriging"]
        print()
    else:
        print("No models fitted successfully!")
        return

    # Step 5: Final prediction and uncertainty mapping
    print("Step 5: Prediction and Uncertainty Mapping")
    print("-" * 45)

    # Create high-resolution prediction grid
    x_pred = np.linspace(0, 15, 60)
    y_pred = np.linspace(0, 15, 60)
    xx, yy = np.meshgrid(x_pred, y_pred)
    pred_coords = np.column_stack([xx.ravel(), yy.ravel()])

    print(f"Predicting at {len(pred_coords)} locations...")
    predictions, variances = best_kriging.predict(pred_coords, return_variance=True)

    # Reshape for plotting
    pred_grid = predictions.reshape(60, 60)
    var_grid = variances.reshape(60, 60)

    # Step 6: Comprehensive visualization
    print("Step 6: Final Visualization")
    print("-" * 30)

    fig = plt.figure(figsize=(20, 16))

    # Main prediction map
    ax1 = plt.subplot(3, 3, (1, 3))
    im1 = ax1.imshow(
        pred_grid, extent=[0, 15, 0, 15], origin="lower", cmap="viridis", aspect="equal"
    )
    scatter1 = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap="viridis",
        s=30,
        edgecolors="white",
        linewidth=0.5,
    )
    plt.colorbar(im1, ax=ax1, label="Predicted Value", shrink=0.8)
    ax1.set_title(f"Anisotropic Kriging Predictions ({best_model})", fontsize=14)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Add anisotropy ellipse
    from pyspatialstats.kriging.anisotropic import plot_anisotropy_ellipse

    plot_anisotropy_ellipse(best_kriging, center=(7.5, 7.5), scale=0.5, ax=ax1)

    # Uncertainty map
    ax2 = plt.subplot(3, 3, (4, 6))
    im2 = ax2.imshow(
        var_grid, extent=[0, 15, 0, 15], origin="lower", cmap="Reds", aspect="equal"
    )
    ax2.scatter(coords[:, 0], coords[:, 1], c="blue", s=15, alpha=0.6)
    plt.colorbar(im2, ax=ax2, label="Kriging Variance", shrink=0.8)
    ax2.set_title("Prediction Uncertainty", fontsize=14)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Directional variograms
    ax3 = plt.subplot(3, 3, 7)
    for angle, result in directional_vario.directional_results_.items():
        valid = result.counts > 0
        if np.any(valid):
            ax3.plot(
                result.bin_centers[valid],
                result.gamma[valid],
                "o-",
                label=f"{angle:.0f}°",
                alpha=0.7,
                markersize=4,
            )
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Semivariance")
    ax3.set_title("Directional Variograms")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Rose diagram
    ax4 = plt.subplot(3, 3, 8, projection="polar")
    angles, ranges = directional_vario.anisotropy_rose_data()
    ax4.bar(
        np.radians(angles),
        ranges,
        width=np.radians(360 / len(angles)),
        alpha=0.7,
        color="purple",
    )
    ax4.set_title("Anisotropy Rose", pad=20)

    # Summary statistics
    ax5 = plt.subplot(3, 3, 9)
    ax5.axis("off")

    # Model info
    aniso_info = best_kriging.get_anisotropy_info()
    summary_text = f"""
Model Summary:
• Model: {best_model.capitalize()}
• R²: {model_results[best_model]["r2"]:.3f}
• RMSE: {model_results[best_model]["rmse"]:.3f}

Anisotropy:
• Ratio: {aniso_info["anisotropy_ratio"]:.2f}
• Major Range: {aniso_info["range_major"]:.2f}
• Minor Range: {aniso_info["range_minor"]:.2f}
• Rotation: {aniso_info["rotation_angle_degrees"]:.1f}°

Parameters:
• Nugget: {aniso_info["nugget"]:.3f}
• Sill: {aniso_info["sill"]:.3f}

Validation:
• Mean Variance: {np.mean(variances):.3f}
• Max Variance: {np.max(variances):.3f}
"""
    ax5.text(
        0.05,
        0.95,
        summary_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig("anisotropic_workflow_complete.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Complete workflow finished!")
    print("Results saved as 'anisotropic_workflow_complete.png'")
    print()

    return {
        "coordinates": coords,
        "values": values,
        "kriging_model": best_kriging,
        "predictions": predictions,
        "variances": variances,
        "anisotropy_info": aniso_info,
        "model_comparison": model_results,
    }


def _skewness(data):
    """Calculate skewness of data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)


def _kurtosis(data):
    """Calculate kurtosis of data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(
        ((data - mean) / std) ** 4
    ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


if __name__ == "__main__":
    # Run all examples
    print("PySpatialStats - Anisotropic Kriging Examples")
    print("=" * 60)
    print()

    # Example 1: Basic anisotropic analysis
    example_1_basic_anisotropic_analysis()

    # Example 2: Parameter recovery validation
    example_2_parameter_recovery_validation()

    # Example 3: Cross-validation comparison
    example_3_cross_validation_comparison()

    # Example 4: Complete real-world workflow
    workflow_results = example_4_real_world_workflow()

    print("All examples completed successfully!")
    print("\nFiles generated:")
    print("• anisotropic_kriging_example.png")
    print("• anisotropic_cross_validation.png")
    print("• anisotropic_workflow_complete.png")
    print("\nFor more advanced usage, see the workflow_results dictionary.")

    # Step 1: Directional variogram analysis
    print("Step 1: Computing directional variograms...")
    directions = [0, 30, 60, 90, 120, 150]  # degrees

    directional_vario = DirectionalVariogram(
        coordinates=coords,
        values=values,
        directions=directions,
        tolerance=22.5,
        max_distance=5.0,
        n_bins=15,
    )
    directional_vario.compute()

    # Detect anisotropy
    anisotropy_result = directional_vario.detect_anisotropy()
    print(f"Anisotropy detected: {anisotropy_result.is_anisotropic}")
    print(f"Major direction: {anisotropy_result.major_direction:.1f}°")
    print(f"Minor direction: {anisotropy_result.minor_direction:.1f}°")
    print(f"Anisotropy ratio: {anisotropy_result.anisotropy_ratio:.2f}")
    print()

    # Step 2: Create anisotropic variogram model
    print("Step 2: Fitting anisotropic variogram model...")
    aniso_variogram = create_anisotropic_variogram_from_directional(
        directional_vario, model="exponential"
    )

    print("Fitted anisotropic parameters:")
    info = {
        "nugget": aniso_variogram.parameters[0],
        "sill": aniso_variogram.parameters[1],
        "range_major": aniso_variogram.parameters[2],
        "range_minor": aniso_variogram.parameters[3],
        "rotation_angle_deg": np.degrees(aniso_variogram.parameters[4]),
    }
    for key, value in info.items():
        print(f"  {key}: {value:.3f}")
    print()

    # Step 3: Anisotropic kriging
    print("Step 3: Performing anisotropic kriging...")
    aniso_kriging = AnisotropicKriging(aniso_variogram)
    aniso_kriging.fit(coords, values)

    # Create prediction grid
    x_pred = np.linspace(0, 10, 30)
    y_pred = np.linspace(0, 10, 30)
    xx, yy = np.meshgrid(x_pred, y_pred)
    pred_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Make predictions
    aniso_predictions, aniso_variance = aniso_kriging.predict(
        pred_coords, return_variance=True
    )

    # Step 4: Compare with isotropic kriging
    print("Step 4: Comparing with isotropic kriging...")

    # Create isotropic variogram (using average range)
    avg_range = (true_params["range_major"] + true_params["range_minor"]) / 2
    iso_variogram = Variogram(model="exponential")
    iso_variogram.parameters = [true_params["nugget"], true_params["sill"], avg_range]
    iso_variogram.is_fitted_ = True
    iso_variogram.nugget_ = true_params["nugget"]
    iso_variogram.sill_ = true_params["sill"]
    iso_variogram.range_ = avg_range

    iso_kriging = OrdinaryKriging(iso_variogram)
    iso_kriging.fit(coords, values)
    iso_predictions = iso_kriging.predict(pred_coords)

    # Step 5: Visualization
    print("Step 5: Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Sample data
    ax = axes[0, 0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=50)
    plt.colorbar(scatter, ax=ax, label="Value")
    ax.set_title("Sample Data")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    # Plot 2: Directional variograms
    ax = axes[0, 1]
    for angle, result in directional_vario.directional_results_.items():
        valid = result.counts > 0
        ax.plot(
            result.bin_centers[valid],
            result.gamma[valid],
            "o-",
            label=f"{angle:.0f}°",
            alpha=0.7,
        )
    ax.set_xlabel("Distance")
    ax.set_ylabel("Semivariance")
    ax.set_title("Directional Variograms")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot 3: Anisotropic predictions
    ax = axes[0, 2]
    pred_grid = aniso_predictions.reshape(30, 30)
    im = ax.imshow(
        pred_grid, extent=[0, 10, 0, 10], origin="lower", cmap="viridis", aspect="equal"
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="white",
        s=20,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(im, ax=ax, label="Predicted Value")
    ax.set_title("Anisotropic Kriging")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot 4: Isotropic predictions
    ax = axes[1, 0]
    iso_grid = iso_predictions.reshape(30, 30)
    im = ax.imshow(
        iso_grid, extent=[0, 10, 0, 10], origin="lower", cmap="viridis", aspect="equal"
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="white",
        s=20,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(im, ax=ax, label="Predicted Value")
    ax.set_title("Isotropic Kriging")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot 5: Difference map
    ax = axes[1, 1]
    diff_grid = pred_grid - iso_grid
    im = ax.imshow(
        diff_grid, extent=[0, 10, 0, 10], origin="lower", cmap="RdBu_r", aspect="equal"
    )
    plt.colorbar(im, ax=ax, label="Anisotropic - Isotropic")
    ax.set_title("Prediction Difference")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot 6: Anisotropic variance
    ax = axes[1, 2]
    var_grid = aniso_variance.reshape(30, 30)
    im = ax.imshow(
        var_grid, extent=[0, 10, 0, 10], origin="lower", cmap="Reds", aspect="equal"
    )
    plt.colorbar(im, ax=ax, label="Kriging Variance")
    ax.set_title("Anisotropic Uncertainty")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("anisotropic_kriging_example.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Example completed! Results saved as 'anisotropic_kriging_example.png'")
    print()


def example_2_parameter_recovery_validation():
    """Example 2: Validate parameter recovery with different anisotropy levels."""

    print("Example 2: Parameter Recovery Validation")
    print("=" * 50)

    # Test different anisotropy ratios
    ratios_to_test = [1.0, 1.5, 2.0, 3.0, 5.0]
    angles_to_test = [0, 30, 60, 90]

    results = []

    for ratio in ratios_to_test:
        for angle in angles_to_test:
            print(f"Testing ratio={ratio:.1f}, angle={angle}°...")

            # Generate data
            coords, values, true_params = generate_anisotropic_data(
                n_samples=200,
                range_major=3.0,
                range_minor=3.0 / ratio,
                rotation_angle=angle,
                nugget=0.05,
                sill=1.0,
                random_seed=42 + int(ratio * 10) + angle,
            )

            # Fit directional variogram
            directional_vario = DirectionalVariogram(
                coordinates=coords,
                values=values,
                directions=np.arange(0, 180, 15),  # Every 15 degrees
                tolerance=15.0,
                max_distance=4.0,
                n_bins=12,
            )
            directional_vario.compute()

            # Create anisotropic variogram
            aniso_variogram = create_anisotropic_variogram_from_directional(
                directional_vario
            )

            # Extract fitted parameters
            fitted_params = {
                "nugget": aniso_variogram.parameters[0],
                "sill": aniso_variogram.parameters[1],
                "range_major": aniso_variogram.parameters[2],
                "range_minor": aniso_variogram.parameters[3],
                "rotation_angle": np.degrees(aniso_variogram.parameters[4]) % 180,
                "fitted_ratio": aniso_variogram.parameters[2]
                / aniso_variogram.parameters[3],
            }

            # Compute errors
            errors = {
                "nugget_error": abs(fitted_params["nugget"] - true_params["nugget"]),
                "sill_error": abs(fitted_params["sill"] - true_params["sill"]),
                "ratio_error": abs(
                    fitted_params["fitted_ratio"] - true_params["anisotropy_ratio"]
                ),
                "angle_error": min(
                    abs(
                        fitted_params["rotation_angle"] - true_params["rotation_angle"]
                    ),
                    180
                    - abs(
                        fitted_params["rotation_angle"] - true_params["rotation_angle"]
                    ),
                ),
            }

            result = {
                "true_ratio": ratio,
                "true_angle": angle,
                "fitted_ratio": fitted_params["fitted_ratio"],
                "fitted_angle": fitted_params["rotation_angle"],
                **errors,
            }
            results.append(result)

    # Summary statistics
    print("\nParameter Recovery Summary:")
    print("-" * 40)

    import pandas as pd

    df = pd.DataFrame(results)

    print("Mean Absolute Errors:")
    print(f"  Nugget: {df['nugget_error'].mean():.4f}")
    print(f"  Sill: {df['sill_error'].mean():.4f}")
    print(f"  Ratio: {df['ratio_error'].mean():.4f}")
    print(f"  Angle: {df['angle_error'].mean():.2f}°")
    print()

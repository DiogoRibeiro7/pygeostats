# examples/anisotropic_kriging_example.py
"""Anisotropic kriging, from directional variograms to predictions.

Simulates a field whose correlation reaches three times further along 30 degrees
than across it, and then:

1. computes directional variograms and estimates the anisotropy axis,
2. derives starting values for an anisotropic model,
3. kriges with AnisotropicKriging and compares it, on held-out samples, with
   isotropic ordinary kriging,
4. plots the directional variograms and both prediction maps.

Fitting an anisotropic model from directional variograms is not implemented yet,
so the anisotropic parameters are set by hand from the estimates. The guide
explains each step and its caveats:
https://diogoribeiro7.github.io/pygeostats/guide/anisotropy/

    python examples/anisotropic_kriging_example.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pygeostats.kriging import AnisotropicKriging, OrdinaryKriging
from pygeostats.variogram import DirectionalVariogram, EmpiricalVariogram, Variogram
from pygeostats.variogram.initialization import RangeInitializer

AXIS_DEGREES = 30.0
RANGE_MAJOR, RANGE_MINOR = 1.5, 0.5


def simulate_field(n_samples: int = 300, seed: int = 0):
    """Samples of a Gaussian field with geometric anisotropy along AXIS_DEGREES."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_samples, 2))

    axis = np.radians(AXIS_DEGREES)
    delta = coords[:, None, :] - coords[None, :, :]
    along = delta[..., 0] * np.cos(axis) + delta[..., 1] * np.sin(axis)
    across = -delta[..., 0] * np.sin(axis) + delta[..., 1] * np.cos(axis)
    scaled_distance = np.sqrt((along / RANGE_MAJOR) ** 2 + (across / RANGE_MINOR) ** 2)

    covariance = np.exp(-scaled_distance) + 0.05 * np.eye(n_samples)
    values = rng.multivariate_normal(np.zeros(n_samples), covariance)
    return coords, values


def main() -> None:
    coords, values = simulate_field()
    order = np.random.default_rng(1).permutation(len(coords))
    train, test = order[:240], order[240:]

    # 1. Directional variograms and the anisotropy axis, from the training samples.
    directional = DirectionalVariogram(
        coords[train],
        values[train],
        directions=[0, 30, 60, 90, 120, 150],
        tolerance=15.0,
        max_distance=5.0,
        n_bins=12,
    ).compute()
    anisotropy = directional.detect_anisotropy()
    print(
        f"Axis: true {AXIS_DEGREES:.0f} degrees, "
        f"estimated {anisotropy.major_direction:.1f} degrees"
    )
    print(
        f"Anisotropic: {anisotropy.is_anisotropic}, ratio "
        f"{anisotropy.anisotropy_ratio:.2f} (true {RANGE_MAJOR / RANGE_MINOR:.1f}). "
        "Treat the ratio as a detection statistic, not an estimate."
    )

    # 2. Starting values for the anisotropic model.
    starting = RangeInitializer(directional.directional_results_).estimate()

    # 3. Anisotropic kriging with those values, against isotropic ordinary kriging.
    anisotropic_variogram = Variogram(model="exponential")
    anisotropic_variogram.parameters = np.array(
        [
            starting.nugget,
            starting.sill,
            starting.range_major,
            starting.range_minor,
            # AnisotropicKriging measures its rotation clockwise, and
            # detect_anisotropy measures directions counter-clockwise.
            -np.radians(anisotropy.major_direction),
        ]
    )
    anisotropic_variogram.is_fitted_ = True
    anisotropic = AnisotropicKriging(anisotropic_variogram).fit(
        coords[train], values[train]
    )

    empirical = EmpiricalVariogram(coords[train], values[train], n_bins=12).compute()
    isotropic_variogram = Variogram(model="exponential").fit(
        empirical.distances_, empirical.gamma_, weights=empirical.counts_
    )
    isotropic = OrdinaryKriging(isotropic_variogram).fit(coords[train], values[train])

    # The field was simulated, so its true parameters are known. Kriging with them
    # shows how much of any gap comes from estimating the axis and ranges, rather
    # than from anisotropic kriging itself.
    true_variogram = Variogram(model="exponential")
    true_variogram.parameters = np.array(
        [0.05, 1.05, RANGE_MAJOR, RANGE_MINOR, -np.radians(AXIS_DEGREES)]
    )
    true_variogram.is_fitted_ = True
    with_true_parameters = AnisotropicKriging(true_variogram).fit(
        coords[train], values[train]
    )

    print("Held-out R^2:")
    print(
        "  anisotropic, estimated parameters "
        f"{anisotropic.score(coords[test], values[test]):.3f}"
    )
    print(
        "  anisotropic, true parameters      "
        f"{with_true_parameters.score(coords[test], values[test]):.3f}"
    )
    print(
        "  isotropic                         "
        f"{isotropic.score(coords[test], values[test]):.3f}"
    )

    # 4. Plots.
    xs = np.linspace(0, 10, 60)
    grid_x, grid_y = np.meshgrid(xs, xs)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for direction, curve in directional.directional_summary().items():
        populated = curve["counts"] > 0
        axes[0].plot(
            curve["bin_centers"][populated],
            curve["gamma"][populated],
            "o-",
            label=f"{direction:.0f} degrees",
        )
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Semivariance")
    axes[0].set_title("Directional variograms")
    axes[0].legend(fontsize=8)

    for ax, (title, model) in zip(
        axes[1:],
        (
            ("Anisotropic kriging", anisotropic),
            ("Isotropic ordinary kriging", isotropic),
        ),
        strict=True,
    ):
        image = ax.imshow(
            model.predict(grid).reshape(grid_x.shape),
            extent=(0, 10, 0, 10),
            origin="lower",
            cmap="viridis",
        )
        ax.scatter(coords[train, 0], coords[train, 1], c="white", s=6)
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    fig.tight_layout()
    fig.savefig("anisotropic_kriging_example.png", dpi=150)
    print("Saved anisotropic_kriging_example.png")


if __name__ == "__main__":
    main()

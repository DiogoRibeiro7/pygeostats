Quick Start Guide
=================

This guide walks you through the basic workflow of PySpatialStats in just a few minutes.

Basic Workflow
--------------

The typical spatial statistics workflow consists of:

1. **Data preparation**: Load and validate your spatial data
2. **Exploratory analysis**: Examine data distribution and spatial patterns  
3. **Variogram modeling**: Compute empirical variogram and fit theoretical models
4. **Kriging interpolation**: Predict values at unsampled locations
5. **Validation**: Assess model performance and uncertainty

Let's walk through each step:

1. Data Preparation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   import geopandas as gpd
   from pyspatialstats.variogram import EmpiricalVariogram, Variogram
   from pyspatialstats.kriging import OrdinaryKriging

   # Create sample data (or load your own)
   np.random.seed(42)
   n_samples = 100
   
   # Random coordinates
   coords = np.random.uniform(0, 10, size=(n_samples, 2))
   
   # Generate spatially correlated values
   distances = np.linalg.norm(coords[:, None] - coords, axis=2)
   correlation = np.exp(-distances / 2.0)
   values = np.random.multivariate_normal(np.zeros(n_samples), correlation)

2. Exploratory Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Basic statistics
   print(f"Value range: [{values.min():.2f}, {values.max():.2f}]")
   print(f"Mean: {values.mean():.2f}, Std: {values.std():.2f}")

   # Plot data locations
   plt.figure(figsize=(10, 4))
   
   plt.subplot(1, 2, 1)
   scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis')
   plt.colorbar(scatter, label='Value')
   plt.xlabel('X coordinate')
   plt.ylabel('Y coordinate') 
   plt.title('Sample Locations')
   
   plt.subplot(1, 2, 2)
   plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
   plt.xlabel('Value')
   plt.ylabel('Frequency')
   plt.title('Value Distribution')
   
   plt.tight_layout()
   plt.show()

3. Variogram Modeling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute empirical variogram
   emp_vario = EmpiricalVariogram(coords, values, max_distance=5.0, n_bins=15)
   emp_vario.compute()

   # Plot empirical variogram
   plt.figure(figsize=(8, 5))
   emp_vario.plot()
   plt.title('Empirical Variogram')
   plt.show()

   # Fit theoretical model
   vario_model = Variogram(model='exponential')
   vario_model.fit(emp_vario.distances_, emp_vario.gamma_)

   print(f"Fitted parameters:")
   print(f"  Nugget: {vario_model.nugget_:.3f}")
   print(f"  Sill:   {vario_model.sill_:.3f}")  
   print(f"  Range:  {vario_model.range_:.3f}")

4. Kriging Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set up kriging
   kriging = OrdinaryKriging(vario_model)
   kriging.fit(coords, values)

   # Create prediction grid
   x_pred = np.linspace(0, 10, 30)
   y_pred = np.linspace(0, 10, 30)
   xx, yy = np.meshgrid(x_pred, y_pred)
   pred_coords = np.column_stack([xx.ravel(), yy.ravel()])

   # Make predictions
   predictions, variance = kriging.predict(pred_coords, return_variance=True)

   # Reshape for plotting
   pred_grid = predictions.reshape(30, 30)
   var_grid = variance.reshape(30, 30)

5. Results Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Kriging predictions
   ax = axes[0]
   im1 = ax.imshow(pred_grid, extent=[0, 10, 0, 10], origin='lower', 
                   cmap='viridis', aspect='equal')
   ax.scatter(coords[:, 0], coords[:, 1], c='white', s=20, 
              edgecolors='black', linewidth=0.5, alpha=0.8)
   plt.colorbar(im1, ax=ax, label='Predicted Value')
   ax.set_title('Kriging Predictions')
   ax.set_xlabel('X')
   ax.set_ylabel('Y')

   # Prediction uncertainty
   ax = axes[1]
   im2 = ax.imshow(var_grid, extent=[0, 10, 0, 10], origin='lower', 
                   cmap='Reds', aspect='equal')
   ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=20, alpha=0.8)
   plt.colorbar(im2, ax=ax, label='Kriging Variance')
   ax.set_title('Prediction Uncertainty')
   ax.set_xlabel('X')
   ax.set_ylabel('Y')

   plt.tight_layout()
   plt.show()

6. Model Validation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cross-validation score
   cv_score = kriging.score(coords, values)
   print(f"Cross-validation R² score: {cv_score:.3f}")

   # Summary statistics
   print(f"\nPrediction summary:")
   print(f"  Mean prediction: {predictions.mean():.3f}")
   print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
   print(f"  Mean kriging variance: {variance.mean():.3f}")

Next Steps
----------

Now that you've completed the basic workflow, explore:

* **Different variogram models**: Try 'spherical', 'gaussian', or 'matern'
* **Advanced kriging**: Universal kriging with trend modeling
* **Real data**: Load your own spatial datasets using GeoPandas
* **Performance**: Benchmark against other spatial statistics packages

For more detailed examples and advanced features, check out our `tutorials <tutorials/index.html>`_ and `API documentation <api/index.html>`_.

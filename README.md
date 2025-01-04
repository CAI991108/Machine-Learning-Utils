# Machine Learning -- Python Code and Solutions

This repository contains the Python code and solutions for machine learning,
including supervised and unsupervised learning, regularization, cross-validation, PCA, and support vector machines (SVM).

## Repository Structure
```
.
├── Gradient Descent Loss Min/
│   ├── least_square_huber.py (.ipynb)
├── Pocket Algorithm/
│   ├── pocket_alg.py (.ipynb)
├── Multi-Class Logistic Regression/
│   ├── lr_coffee.py 
│   ├── lr_weather.py
│   ├── lr_test.ipynb 
├── Subgradient Non-Smooth Optimization/
│   ├── subgrad.py (.ipynb)
├── Overfitting Behavior/
│   ├── overfit_b.py (.ipynb)
├── Validation and Regularization/
│   ├── val_regularization.ipynb
├── Face Reconstruction by PCA/
│   ├── face_pca.ipynb
├── SVM and Kernel Methods/
│   ├── svm_kernel.ipynb
├── reports/
└── README.md
```


**Gradient descent for minimizing the loss function**

- The script `least_squre_huber.py` and the `least_square_huber.ipynb` performs two main tasks: 
**Least Squares Estimation** and **L1 Estimation using Huber Approximation**. 
It loads data (`X`, `y`, `theta_star`), computes the least squares estimator, and calculates its error. For the L1 estimator, 
it uses gradient descent with a smoothing parameter (`mu`) and step size (`alpha`) to approximate the solution, 
plotting the L2 error over iterations. The notebook outputs the least squares error and a convergence plot for the L1 estimator.

**Pocket algorithm for non-separable data**

- The Python script `pocket_alg.py` and the `pocket_alg.ipynb`implements a **Perceptron algorithm** for binary classification 
using image data from the MNIST dataset. 
It calculates features like **intensity** and **symmetry** from the images, trains the Perceptron model, 
and evaluates its performance by plotting training and test errors over iterations. 
The script also visualizes the decision boundary and feature distributions. Key steps include feature extraction, 
Perceptron training, error calculation, and plotting the decision boundary and error curves. 
The goal is to classify digits (e.g., 1 vs. 6) based on intensity and symmetry features.

**Multi-Class Logistic Regression**
- The scripts `lr_coffee.py` and `lr_weather.py` implement **multinomial logistic regression** 
using **Gradient Descent (GD)** and **Accelerated Gradient Descent (AGD)** for image classification. 
They load image datasets (coffee roast levels and weather conditions), preprocess the data (normalization, one-hot encoding),
and train models to classify images. Both scripts evaluate training and test losses/accuracies over epochs and plot the results.
Key steps include data loading, softmax loss calculation, optimization (GD/AGD), and performance visualization.
- The `lr_test.ipynb` is a Python script for image classification using logistic regression with 
gradient descent (GD) and accelerated gradient descent (AGD) on two datasets: a **Coffee Dataset** and a **Weather Dataset**.
It includes functions for loading and preprocessing image data, normalizing features, and adding bias terms. 
The script trains models using softmax regression, evaluates performance every 5 epochs, and tracks training/test losses and accuracies.
For the Coffee Dataset, the model achieves around 90% accuracy, while the Weather Dataset shows slower convergence with lower accuracy.
The script also includes visualizations of losses and accuracies over epochs for both optimization methods.

**Subgradient method for optimizing non-smooth functions**
- The `subgrad.py` and `subgrad.ipynb` implements the **subgradient method** to solve an optimization problem 
using three different learning rate strategies:**constant**, **polynomial diminishing**, and **geometrically diminishing**. 
The data is loaded from `.npy` files, including matrix `A`, vector `b`, and the optimal solution `x_star`. 
The subgradient is computed using a custom function, and the optimality gap is calculated to measure the distance 
between the current solution and the optimal solution. 
The script runs the subgradient method for 300 iterations, tracking the optimality gaps for each learning rate strategy. 
Finally, it visualizes the results by plotting the optimality gaps over iterations, both in normal and log scales, 
to compare the convergence behavior of the different learning rate strategies.

**Experiments on overfitting behavior**
- The `overfit_b.py` and the `overfit_b.ipynb` explores **overfitting** in polynomial regression using **Legendre polynomials** 
by generating synthetic data with varying polynomial complexity `Q_g` and noise levels `sigma`. 
It fits models of different complexities (degree 2 and 10) and calculates the **out-of-sample error** to measure overfitting 
as the difference in errors between the two models. 
The script conducts two experiments: (1) varying `Q_g` and the number of data points `n` with fixed noise, 
and (2) varying `sigma` and `Q_g` with fixed `n`. 
The results are visualized using contour plots to illustrate how overfitting is influenced by model complexity, data size, and noise, 
providing insights into balancing these factors to avoid overfitting.

**Overfitting, Validation and Regularization**
- The `val_regularization.ipynb` describes a process of fitting an 8th-order polynomial to training data using the least squares 
and evaluating its performance through Ridge regression. 
Training and test data are loaded, and a Vandermonde matrix is created for polynomial fitting. 
The model is trained, and the fitted curve is plotted. 
Cross-validation performed to determine the optimal regularization parameter (λ = 2)for Ridge regression. 
The model's performance evaluated using cross-validation and test errors, with visualizations of fitted curves for different λ values.
The best λ minimizes cross-validation error, and test errors reported for various λ values, demonstrating the model's generalization capability.

**Face Reconstruction by PCA**
- The `face_pca.ipynb` outlines an analysis of the ORL faces database using Singular Value Decomposition (SVD) 
to extract eigenfaces and reconstruct images. 
The process involves centering the data, performing SVD, and visualizing the top 40 eigenfaces.
Five random images are reconstructed using varying numbers of eigenfaces (k = 20, 40, 100, 200, 300), 
and the results are displayed alongside the original images. 
The Signal-to-Noise Ratio (SNR) is calculated for different k values to evaluate reconstruction quality, 
and a plot of SNR versus the number of eigenfaces is generated, showing how SNR improves with more eigenfaces.
This demonstrates the trade-off between reconstruction accuracy and the number of eigenfaces used.

**Support Vector Machine and Kernel Methods**
- The `svm_kernel.ipynb` describes an experiment using Support Vector Machines (SVMs) with different kernels (linear, quadratic, and RBF) 
to classify the MNIST dataset.
The data is preprocessed, split into training, validation, and test sets, and normalized.
GridSearchCV is used to tune hyperparameters (C and gamma) for each kernel, and the best models are selected based on validation error. 
The linear and quadratic kernels are evaluated first, with their validation and test errors reported. 
The RBF kernel is then tuned with specific gamma values, and the best model retrained and evaluated. 
Results show that the RBF kernel achieves the lowest validation and test errors (0.0610 and 0.0494, respectively), 
outperforming the linear and quadratic kernels.

### Dependencies

- Python 3.x, NumPy, SciPy, Matplotlib, Scikit-learn, Jupyter Notebook

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Machine-Learning-Utils

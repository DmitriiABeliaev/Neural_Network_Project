# Deep L-layer Neural Network

This project had two main goals:
- Implementing a deep L-layer neural network from scratch for binary image classification (Cat Photo vs Non-Cat Image).
- A complete machine learning pipeline using tabular health data (Heart Disease dataset), including preprocessing, model building, optimization, and hyperparameter tuning.

## L-Layer Deep Neural Network for Image Classification
- Goal: Classify 64x64 RGB images into Cat or Non-Cat using a 4-layer neural network.
- Model Architecture: [LINEAR → RELU] × (L-1) → LINEAR → SIGMOID.
- Tools: numpy, matplotlib, h5py, scipy, PIL, custom helper modules (nn_functions, extra_functions).
- Task Structure:
  - Data flattening and normalization.
  - Forward and backward propagation.
  - Cost function and gradient descent.
  - Final test accuracy (significantly higher than logistic regression baseline).
 
## Full ML Pipeline: Heart Disease Prediction
- Dataset: Heart Disease Dataset with 11 features and 1 target label.
- Goal: Predict the presence of heart disease using categorical and numerical features.

### Data Preprocessing:
- Label encoding of categorical variables.
- Data normalization using L1 norm.
- Train-test split (90%/10%).

### Model Implementation
Four model versions were tested for comparison:
| Model Version                | Techniques Used                        |
| ---------------------------- | -------------------------------------- |
| Basic Model                  | Standard feedforward + backpropagation |
| L2 Regularized Model         | L2 penalty added to cost and gradients |
| Momentum Optimized Model     | Gradient descent with momentum         |
| L2 + Momentum Combined Model | Both L2 regularization + momentum      |

Improvements Observed:
- Momentum helped smooth convergence.
- L2 slightly improved generalization.
- Best results achieved with combined approach and deeper architecture.

### Hyperparameter Tuning
Used randomized search for:
- Number of iterations (best ≈ 4800)
- Learning rate (best ≈ 0.265)
- Hidden layer dimensions (best: [11, 120, 50, 19, 7, 4, 1])

### Results Summary

| Model                 | Train Accuracy | Test Accuracy |
| --------------------- | -------------- | ------------- |
| Basic L-Layer         | \~74%          | \~70%         |
| L2 Regularization     | \~75%          | \~72%         |
| Momentum Optimization | \~78%          | \~74%         |
| L2 + Momentum (Best)  | \~85%          | \~84%         |

## Lessons Learned
- Proper data preprocessing has a big impact on model performance.
- Simple models can work surprisingly well on small datasets.
- L2 regularization helped minimally due to low variance in the dataset.
- Momentum optimization significantly improved convergence speed and final accuracy.
- Hyperparameter tuning via random search provided consistent performance boosts.

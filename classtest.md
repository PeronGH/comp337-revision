# Class Test Revision

## Feature Normalization

### [0, 1]-scaling

- **Formula:** $\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$

- **Core Concept:** rescaling data to the [0,1] interval using the provided formula

### Gaussian Normalization

- **Formula:** $\hat{x} = \frac{x - \mu}{\sigma}$
  - $u$: mean
  - $\sigma$: standard deviation

## Linear Algebra

### Dot Product

$C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, the dot product, $C = AB$, is an $m \times p$ matrix whose entries are $c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$.

### Linearly Independent

- Only solution to $\lambda_1A_1 + \lambda_2A_2 + ... + \lambda_nA_n = 0$ is $\lambda_1 = \lambda_2 = ... = \lambda_n = 0$

- **Proof:** Solve $\lambda_1A_1 + \lambda_2A_2 + ... + \lambda_nA_n = 0$ and the only solution is $\lambda_1 = \lambda_2 = ... = \lambda_n = 0$.

### Rank

Maximum number of linearly independent columns (or rows) in $A$.

- **Computation:**
  - (square matrix only) If $\det(A) \neq 0$, the rank is the size of the matrix.
  - (quick path) identify Rows that are linearly independent, which gives the minimum value of the rank.
  - Gaussian Elimination to row-echelon form, then count non-zero rows.
  - Operations of **Gaussian Elimination**:
    1.  Swapping two rows.
    2.  Multiplying a row by a non-zero scalar.
    3.  Adding a multiple of one row to another row.
  - **Row-Echelon Form** (e.g., $\begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}$) is defined by:
    1.  All non-zero rows are above any rows of all zeros.
    2.  The leading coefficient (the first non-zero number from the left, also called the *pivot*) of a non-zero row is always strictly to the right of the leading coefficient of the row above it.
    3.  All entries in a column below a leading entry are zeros.

### Trace

The sum of diagonal elements.

### Eigenvalues and Eigenvectors

- $Av = \lambda v$, where $v$ is the eigenvector and $\lambda$ is the eigenvalue.

- **Computation (If $A$ is 2×2):**
  - Solve $\text{det}(A - \lambda I) = \text{det}\begin{bmatrix} a-\lambda & b \\ c & d-\lambda \end{bmatrix} = (a-\lambda)(d-\lambda) - bc = 0$.
  - The solutions, $\lambda_1$ and $\lambda_2$, are the eigenvalues.
  - Solve $\begin{bmatrix} a-\lambda_1 & b \\ c & d-\lambda_1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. Non-trivial solution is eigenvector $\mathbf{v_1}$.
  - Solve $\begin{bmatrix} a-\lambda_2 & b \\ c & d-\lambda_2 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. Non-trivial solution is eigenvector $\mathbf{v_2}$.

## Differential Calculus

### Common Derivatives

- $\frac{d}{dx}c = 0$ ($c$ is constant)
- $\frac{d}{dx}x^n=nx^{n-1}$
- $\frac{d}{dx}e^x=e^x$
- $\frac{d}{dx}\ln{x}=\frac{1}{x}$ ($x > 0$)
- $\frac{d}{dx}\sin{x}=\cos{x}$
- $\frac{d}{dx}\cos{x}=-\sin{x}$

### Differentiation Rules

- $\frac{d}{dx}[f(x) \pm g(x)] = \frac{d}{dx}f(x) \pm \frac{d}{dx}g(x)$
- $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
- $\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$
- $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$

### Partial Derivative

A partial derivative is the derivative of a multi-variable function *with respect to only one of those variables*, **treating all other variables as constant**.

## Perceptron

### Algorithm

**Input:**

*   A dataset of labeled examples, $D = \{(\vec{X}_1, y_1), ..., (\vec{X}_n, y_n)\}$.
*   Each $\vec{X}$ is a vector of features, and $x_0=1$.
*   Each $y$ is a binary label: $+1$ or $-1$. It is defined $\text{sign}(\vec{W}^T\vec{X} + b)$.

**Algorithm:**

1.  **Initialization:** Weights $\vec{W}$ and bias $b$ are initialized to zero, and $w_0=b$.
2.  **PerceptronTrain:** For a fixed number of iterations (or until convergence):
    *   For each training example $(\vec{X}, y)$:
        *   Compute the activation: $a = \vec{W}^T\vec{X} + b$.
        *   If the prediction is incorrect ($y \cdot a \le 0$):
            *   Update weights: $\vec{W} \leftarrow \vec{W} + y\vec{X}$.
            *   Update bias: $b \leftarrow b + y$.

**Output:**

*   The trained weights, $\vec{W}$, and bias, $b$. These define the hyperplane that separates the classes.
* The algorithm can detect incorrect predictions.

### Geometric Interpretation

- Linearly separable data can be perfectly divided by the perceptron's hyperplane.

## Confusion Matrix

|                        |   Actual Positive   |   Actual Negative   |
| :--------------------- | :-----------------: | :-----------------: |
| **Predicted Positive** | True Positive (TP)  | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN)  |

- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision**: $\frac{\text{TP}}{\text{Predicted Positive}} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\frac{\text{TP}}{\text{Actual Positive}} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F-score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$

- **Marco F-score:** Mean of F-scores for all classes

## Multiclass Classification 

### One-vs-Rest

- Train $K$ binary classifiers, where $K$ is the number of classes.
- $k$-th classifier is trained to distinguish class $k$ from all other classes.
- Prediction: classify with the classifier that outputs the highest confidence score.

### One-vs-One

- Train $\frac{K(K-1)}{2}$ binary classifiers, one for each pair of classes.
- Prediction: each classifier votes for a class, and the majority wins.

## Regularization

- Regularization adds a penalty term based on the *magnitude* of the weights.

- Assume a simple linear regression: $y = w^Tx + b$, where $y$ is the prediction, $x$ are the features, $w$ are the weights, and $b$ is the bias.
- The origianl loss is MSE ($\sum_{i=1}^{n}(y_i - \hat{y_i})^2$).

### L2 Regularization

- $\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{m}w_j^2$

### L1 Regularization

- $\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{m}|w_j|$

### L1+L2 Regularization

- $\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2 + \lambda_1 \sum_{j=1}^{m}|w_j| + \lambda_2 \sum_{j=1}^{m}w_j^2$

## K-Nearest Neighbors

1. Select the number ($k$) of nearest neighbors to consider.
2. For a new, unlabeled data point, calculate its distance to all points in the training data.
3. Identify the $k$ training data points closest to the new point.
4. Assign the new point the class label that is most frequent among its $k$ nearest neighbors.

### Vector Norms

- **L0 Norm:** Counts non-zero elements.

- **L1 Norm:** $\left \| x \right \|_{1} = \sum_{i=1}^{n}\left | x_{i} \right |$
- **L2 Norm:** $\left \| x \right \|_{2} = \sqrt{\sum_{i=1}^{n} x_{i}^{2}}$
- **L-Infinity Norm:** $||x||_\infty = \max_{i} |x_i|$

### Similarity / Distance

- **Cosine Similarity:** $\frac{A \cdot B}{||A||_2 ||B||_2}$
- **Cosine Distance:** $1 - \text{Cosine Similarity}$
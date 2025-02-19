# Class Test Revision

## Feature Normalization

### [0, 1]-scaling

- **Formula:** $\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$

- **Core Concept:** rescaling data to the [0,1] interval using the provided formula

### Gaussian Normalization

- **Formula:** $\hat{x} = \frac{x - \mu}{\sigma}$
  - $u$: mean
  - $\sigma$: standard deviation

## Mathematical Preliminaries

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

- **Computation (If A is 2×2):**
  - Solve $\text{det}(A - \lambda I) = \text{det}\begin{bmatrix} a-\lambda & b \\ c & d-\lambda \end{bmatrix} = (a-\lambda)(d-\lambda) - bc = 0$.
  - The solutions, $\lambda_1$ and $\lambda_2$, are the eigenvalues.
  - Solve $\begin{bmatrix} a-\lambda_1 & b \\ c & d-\lambda_1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. Non-trivial solution is eigenvector $\mathbf{v_1}$.
  - Solve $\begin{bmatrix} a-\lambda_2 & b \\ c & d-\lambda_2 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. Non-trivial solution is eigenvector $\mathbf{v_2}$.

### Differential Calculus

#### Common Derivatives

- $\frac{d}{dx}c = 0$ ($c$ is constant)
- $\frac{d}{dx}x^n=nx^{n-1}$
- $\frac{d}{dx}e^x=e^x$
- $\frac{d}{dx}\ln{x}=\frac{1}{x}$ ($x > 0$)
- $\frac{d}{dx}\sin{x}=\cos{x}$
- $\frac{d}{dx}\cos{x}=-\sin{x}$

#### Differentiation Rules

- $\frac{d}{dx}[f(x) \pm g(x)] = \frac{d}{dx}f(x) \pm \frac{d}{dx}g(x)$
- $\frac{d}{dx}[f(x)g(x)] = g(x)f'(x) + f(x)g'(x)$
- $\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{g(x)f'(x) - f(x)g'(x)}{[g(x)]^2}$
- $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$

#### Partial Derivative

A partial derivative is the derivative of a multi-variable function *with respect to only one of those variables*, **treating all other variables as constant**.
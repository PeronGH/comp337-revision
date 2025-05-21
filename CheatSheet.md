# COMP337/COMP527 Exam Cheat Sheet

## 1. Mathematical Preliminaries

### 1.1. Linear Algebra

*   **Vectors:** Ordered sets of coordinates. Denoted by $\bar{X}$, $\bar{Y}$, etc.
    *   **Column Vector:** Default representation. Transposed ($\bar{X}^T$) for row vector.
    *   **$d$-dimensional vector with real coordinates:** $\bar{X} \in \mathbb{R}^d$.
*   **Matrices:** Collection of vectors arranged by columns or rows. Denoted by $\bar{M}$.
    *   **$n \times m$ matrix:** $\bar{M} \in \mathbb{R}^{n \times m}$ has $n$ rows and $m$ columns.
    *   **Square Matrix:** $n=m$.
    *   **Element:** $(i,j)$ element of $\bar{M}$ is $\bar{M}_{i,j}$.
    *   **Symmetric Matrix:** $\bar{M}_{i,j} = \bar{M}_{j,i}$ for all $i,j$.
*   **Vector Arithmetic:**
    *   **Addition:** $\bar{Z} = \bar{X} + \bar{Y} \implies z_i = x_i + y_i$.
        *   *Calculator Tip:* Use vector/matrix mode for element-wise addition.
    *   **Inner-product (Dot Product):** $\bar{X}^T \bar{Y} = \sum_{i=1}^d x_i y_i$.
        *   *Calculator Tip:* Use vector/matrix mode for dot product.
    *   **Outer-product:** $\bar{X} \bar{Y}^T$ (results in a matrix). $(\bar{X} \bar{Y}^T)_{i,j} = x_i y_j$.
        *   *Calculator Tip:* Use matrix multiplication.
*   **Matrix Arithmetic:**
    *   **Addition (element-wise):** $\bar{A} + \bar{B} = \bar{C} \implies \bar{C}_{i,j} = \bar{A}_{i,j} + \bar{B}_{i,j}$.
        *   *Property:* Matrix addition is commutative: $\bar{A} + \bar{B} = \bar{B} + \bar{A}$.
        *   *Calculator Tip:* Use matrix mode for addition.
    *   **Multiplication:** $\bar{C} = \bar{A} \bar{B}$ (if columns of $\bar{A}$ = rows of $\bar{B}$). $\bar{C}_{i,j} = \sum_{k=1}^m \bar{A}_{i,k} \bar{B}_{k,j}$.
        *   *Property:* Matrix multiplication is generally NOT commutative: $\bar{A} \bar{B} \neq \bar{B} \bar{A}$.
        *   *Calculator Tip:* Use matrix mode for multiplication.
*   **Transpose:** $\bar{A}^T$ where $\bar{A}^T_{i,k} = \bar{A}_{k,i}$.
    *   *Property:* $(\bar{A}\bar{B})^T = \bar{B}^T \bar{A}^T$.
    *   *Calculator Tip:* Use matrix transpose function.
*   **Inverse:** $\bar{A}^{-1}$ for a square matrix $\bar{A}$ such that $\bar{A}\bar{A}^{-1} = \bar{A}^{-1}\bar{A} = \bar{I}$ (unit matrix).
    *   **For $2 \times 2$ matrix $\bar{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:** $\bar{A}^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.
    *   *Calculator Tip:* For $2 \times 2$ matrices, calculate determinant and adjoint manually, or use calculator's inverse function if available for small matrices.
*   **Linear Independence:** Vectors $\bar{X}_1, \dots, \bar{X}_k$ are linearly independent if $\lambda_1 \bar{X}_1 + \dots + \lambda_k \bar{X}_k = \bar{0}$ implies $\lambda_1 = \dots = \lambda_k = 0$.
    *   *Solving:* Set up a system of linear equations and solve for $\lambda_i$.
*   **Rank:** Number of linearly independent columns (or rows) of a matrix. $\text{rank}(\bar{A}) \le \min\{n, m\}$.
    *   *Solving:* For small matrices, check if rows/columns are scalar multiples of each other.
*   **Trace:** Sum of diagonal elements of a square matrix. $\text{tr}(\bar{A}) = \sum_i \bar{A}_{i,i}$.
*   **Eigenvalues and Eigenvectors:** For a square matrix $\bar{A}$, a non-zero vector $\bar{X}$ is an eigenvector with eigenvalue $\lambda$ if $\bar{A}\bar{X} = \lambda\bar{X}$.
    *   *Solving for Eigenvalues (for $2 \times 2$ matrix $\bar{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$):* Solve the characteristic equation $\text{det}(\bar{A} - \lambda\bar{I}) = 0 \implies (a-\lambda)(d-\lambda) - bc = 0$. This is a quadratic equation for $\lambda$.
    *   *Calculator Tip:* Your calculator cannot compute eigenvalues directly, but it can solve quadratic equations.

### 1.2. Differential Calculus

*   **Basic Derivatives:**
    *   $\frac{d}{dx} a = 0$ (for constant $a$)
    *   $\frac{d}{dx} x^a = a \cdot x^{a-1}$
    *   $\frac{d}{dx} e^x = e^x$
    *   $\frac{d}{dx} \log(x) = \frac{1}{x}$ (natural logarithm)
    *   $\frac{d}{dx} \sin(x) = \cos(x)$
    *   $\frac{d}{dx} \cos(x) = -\sin(x)$
*   **Differentiation Rules:**
    *   **Sum Rule:** $(\alpha f + \beta g)' = \alpha f' + \beta g'$
    *   **Product Rule:** $(fg)' = f'g + fg'$
    *   **Quotient Rule:** $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$
    *   **Chain Rule:** If $f(x) = h(g(x))$, then $f'(x) = h'(g(x)) \cdot g'(x) = \frac{d}{dg(x)} h \cdot \frac{d}{dx} g$.
*   **Partial Derivative:** Derivative of a function of several variables with respect to one variable, holding others constant.
    *   Example: $f(x,y) = 5x + y^2 \implies \frac{\partial f}{\partial x} = 5$, $\frac{\partial f}{\partial y} = 2y$.
*   **Gradient:** Vector of all partial derivatives. $\nabla_{\bar{X}} f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_d}\right)^T$.

### 1.3. Optimisation (Conceptual)

*   **Unconstrained Optimisation:** Find $\min_{\bar{X}} f(\bar{X})$.
    *   **Gradient Descent:** Iterative method: $\bar{X}_{i+1} = \bar{X}_i - \gamma_i \cdot (\nabla_{\bar{X}} f)(\bar{X}_i)$.
        *   $\nabla_{\bar{X}} f$ points in the direction of steepest ascent. Negative gradient points to steepest descent.
*   **Constrained Optimisation:** Find $\min_{\bar{X}} f(\bar{X})$ subject to $g(\bar{X}) = 0$.
    *   **Lagrange Multipliers:** Form Lagrangian $\mathcal{L}(\bar{X}, \lambda) = f(\bar{X}) - \lambda \cdot g(\bar{X})$. Find stationary points by setting $\nabla_{(\bar{X},\lambda)}\mathcal{L} = \bar{0}$.

## 2. Data Preprocessing & Representation

### 2.1. Data Types

*   **Nondependency-oriented (Multidimensional):** Objects do not have dependencies (e.g., table data).
    *   **Numerical/Quantitative:** Values with natural ordering (integer, real).
    *   **Categorical/Unordered Discrete-valued:** Discrete unordered values (e.g., colors).
    *   **Binary:** Two values (0 and 1).
    *   **Text:** Document as string or set of words/terms (vector-space representation).
*   **Dependency-oriented:** Implicit or explicit dependencies between objects.
    *   **Implicit Dependencies:** Not explicitly specified but known to exist.
        *   **Time-series:** Sequential measurements over time (time-stamp/index value is contextual).
        *   **Discrete Sequences/Strings:** Categorical analog of time-series.
        *   **Spatial Data:** Records with location attributes.
        *   **Spatiotemporal Data:** Contain both spatial and temporal attributes.
    *   **Explicit Dependencies:** Graphs or network data (edges specify relationships).
        *   **Network/Graph Data:** Objects are nodes, relationships are edges (directed/undirected).

### 2.2. Data Representation

*   **Categorical Data:** Represent each category as a separate dimension of a vector (e.g., one-hot encoding).
*   **Text Data:**
    *   List of words: `["the", "burger", "I", "ate", ...]`
    *   Set of words: `{"the", "burger", "I", "ate", ...}`
    *   Vector of word frequency (Bag of Words): `{"the":1, "burger":2, ...}`
    *   Vector of letter frequency.

### 2.3. Missing Values

*   **Problem:** Data may have erroneous or missing fields.
*   **Handling Strategies:**
    1.  **Discard:** Remove entire records with missing values. (May lose important info if missing data is not random).
    2.  **Fill in by hand:** Manual annotation/re-measurement. (Reliable but slow, costly, often impractical).
    3.  **Set "missingValue":** Treat "missing" as a new category. (Not suitable for numerical data, doesn't solve problem).
    4.  **Replace with mean/median/mode:** Compute mean/median/mode of available values and impute. (Good if data is representative, inaccurate if outliers).
    5.  **Predict:** Train a model to predict missing values, then use imputed data for main task.
    6.  **Accept missing values:** Let the algorithm handle them (some algorithms can).

### 2.4. Noisy Data

*   **Definition:** Random errors scattered in data (inaccurate recording, corruption).
*   **Problem:** Leads to **overfitting**.
    *   **Overfitting:** Model learns "too much" from training data, including noise, and does not generalize well to unseen data. High accuracy on train, low on test.
    *   **Underfitting:** Model is not sufficiently "fitted" to the training data. Poor performance on both train and test.
*   **Solutions to Underfitting:**
    *   Train for more iterations.
    *   Implement more/better features.
    *   Cleanse/re-annotate train data.
    *   Select a different training algorithm.
*   **Solutions to Overfitting:**
    *   Reduce model flexibility (e.g., **Regularisation**).
    *   Remove features.
    *   **Early stopping:** Premature termination of training.
    *   Train with more data.
    *   **Cross-validation.**
*   **Detecting Noisy Data:**
    *   Obvious: Incorrect type (string in numeric), very dissimilar values (outliers).
    *   Non-obvious: Typing errors (e.g., 0.52 instead of 0.25).
*   **Handling Noisy Values:**
    *   Manual inspection/removal.
    *   Clustering/Outlier detection to remove instances/features.
    *   Linear regression to predict and remove far-off values.
    *   Ignore values below frequency threshold (for text, misspellings).
    *   Apply missing value techniques after identification/removal.

### 2.5. Feature Normalisation

*   **Purpose:** Scale numerical features to a common range.
*   **Method 1: [0,1]-scaling (Min-Max Scaling):**
    *   $\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$
    *   Values will be in $[0,1]$.
    *   *Calculator Tip:* Find min/max values from data, then apply formula.
*   **Method 2: Gaussian Normalisation (Z-score Normalization):**
    *   $\hat{x} = \frac{x - \mu}{\sigma}$
    *   $\mu$: mean of feature values. $\sigma$: standard deviation of feature values.
    *   Resulting feature has zero mean and unit variance.
    *   *Calculator Tip:* Use calculator's statistics mode to find mean ($\mu$) and standard deviation ($\sigma$).

## 3. Core Data Mining Problems & Algorithms

### 3.1. Classification

*   **Goal:** Learn relationships between a fixed feature (class label) and others to predict unknown labels.
*   **Supervised Learning:** Training data has labels.
*   **Perceptron (Binary Classifier):**
    *   **Model:** Bio-inspired, single neuron. Activation score $a = \sum_{i=1}^d w_i x_i + b = \bar{W}^T \bar{X} + b$.
    *   **Prediction:** Output $1$ if $a > 0$, else $-1$. (Threshold $\theta$ is set to $0$ by incorporating bias $b = -\theta$).
    *   **Training Algorithm (Online, Error-driven):**
        *   Initialize $w_i = 0$, $b = 0$.
        *   For each training instance $(\bar{X}, y)$:
            *   Compute $a = \bar{W}^T \bar{X} + b$.
            *   If $y \cdot a \le 0$ (misclassification):
                *   Update weights: $w_i \leftarrow w_i + y \cdot x_i$ for all $i$.
                *   Update bias: $b \leftarrow b + y$.
    *   **Geometric Interpretation:** Hyperplane $\bar{W}^T \bar{X} + b = 0$ separates classes. $\bar{W}$ is perpendicular to the hyperplane. Update rule moves $\bar{W}$ towards misclassified points.
    *   **Linear Separability:** If a dataset can be separated by a hyperplane, it's linearly separable. Perceptron converges for linearly separable data.
*   **Logistic Regression (Probabilistic Discriminative Classifier):**
    *   **Model:** Predicts probability of class membership using a sigmoid function.
    *   **Sigmoid Function:** $\sigma(x) = \frac{1}{1 + e^{-x}}$.
        *   *Properties:* $\sigma(x) \in [0,1]$, $1 - \sigma(x) = \sigma(-x)$, $\frac{\partial \sigma}{\partial x} = \sigma(x)(1-\sigma(x))$.
    *   **Probability Prediction:** $P(y=+1|\bar{X}) = \sigma(a)$ and $P(y=-1|\bar{X}) = \sigma(-a)$, where $a = \bar{W}^T \bar{X} + b$.
    *   **Training (MLE with Gradient Descent):** Minimizes negative log-likelihood.
        *   Update rules (Online/SGD version, for misclassified $(\bar{X}, y)$):
            *   $w_s \leftarrow w_s + \mu \cdot y \cdot \sigma(-y \cdot a) \cdot x_s^{(i)}$
            *   $b \leftarrow b + \mu \cdot y \cdot \sigma(-y \cdot a)$
            *   (With $\mu=1$, $\sigma(-y \cdot a)$ is the probability of misclassification).
*   **K-Nearest Neighbors (k-NN):**
    *   **Training:** Store entire training dataset.
    *   **Classification:** For new $\bar{X}'$:
        1.  Find $k$ closest training objects to $\bar{X}'$ using a distance metric.
        2.  Find the majority class label among these $k$ neighbors.
        3.  Predict this majority label for $\bar{X}'$.
    *   **Distance Measures:**
        *   **Euclidean Distance ($L^2$-norm):** $\text{EucDist}(\bar{X}, \bar{Y}) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$.
        *   **Manhattan Distance ($L^1$-norm):** $\text{ManDist}(\bar{X}, \bar{Y}) = \sum_{i=1}^d |x_i - y_i|$.
        *   **$L^0$-norm:** Number of non-zero elements.
        *   **$L^\infty$-norm:** $\max\{|x_1|, \dots, |x_d|\}$.
        *   **Cosine Similarity:** $\text{CosSim}(\bar{X}, \bar{Y}) = \frac{\bar{X}^T \bar{Y}}{\|\bar{X}\|_2 \|\bar{Y}\|_2}$. (Higher value = more similar).
        *   **Cosine Distance:** $\text{CosDist}(\bar{X}, \bar{Y}) = 1 - \text{CosSim}(\bar{X}, \bar{Y})$.
        *   **Jaccard Similarity (for sets $A, B$):** $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$.
        *   **Jaccard Distance:** $d_J(A,B) = 1 - J(A,B)$.
        *   **Overlap Coefficient (for sets $A, B$):** $\text{overlap}(A,B) = \frac{|A \cap B|}{\min(|A|, |B|)}$.
        *   **Hamming Distance (for binary vectors):** Number of coordinates where vectors differ.
        *   **Categorical Data Similarity:** $\text{Sim}(\bar{X}, \bar{Y}) = \sum_{i=1}^d S(x_i, y_i)$.
            *   $S(x_i, y_i) = 1$ if $x_i = y_i$, else $0$.
            *   Smoothed: $S(x_i, y_i) = 1/p_k(x_i)^2$ if $x_i = y_i$, else $0$.
    *   **Choosing $k$ (Hyperparameter):** Use validation data/cross-validation.
    *   **Complexity:** Training is cheap (just storage). Classification is expensive (distance calculation for each test instance).
*   **Naive Bayes Classifier (Probabilistic Generative Classifier):**
    *   **Bayes' Rule:** $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$.
        *   $P(H|E)$: Posterior (what we want to estimate).
        *   $P(E|H)$: Likelihood.
        *   $P(H)$: Prior.
        *   $P(E)$: Evidence.
    *   **Naive Assumption:** Features are independent of one another *conditional on the class*.
        *   $P(x_1=a_1, \dots, x_d=a_d | C=c) = \prod_{i=1}^d P(x_i=a_i | C=c)$.
    *   **Classification Task:** Predict class $c$ that maximizes $P(C=c) \prod_{i=1}^d P(x_i=a_i | C=c)$.
    *   **Probability Estimation:** $P(x_i=a|C=c) = \frac{n(a,c)}{N(c)}$ (fraction of training objects in class $c$ with feature $x_i=a$).
    *   **Zero Probabilities & Laplace Smoothing:** If $n(a,c)=0$, the product becomes $0$.
        *   **Laplace Smoothing:** $P(x_i=a|C=c) = \frac{n(a,c)+1}{N(c)+m_i}$, where $m_i$ is the number of possible values for feature $x_i$.
            *   *Calculator Tip:* Apply the formula directly.
*   **Multiclass Classification (using Binary Classifiers):**
    *   **One-vs.-one:** Train $\frac{k(k-1)}{2}$ binary classifiers, one for each pair of classes. For prediction, each classifier votes, and the class with most votes wins.
    *   **One-vs.-rest:** Train $k$ binary classifiers, one for each class (class $i$ vs. all other classes). For prediction, choose the class whose classifier outputs the highest confidence score.

### 3.2. Clustering

*   **Goal:** Partition objects into sets (clusters) such that objects in a cluster are "similar".
*   **Unsupervised Learning:** No labels in training data.
*   **Evaluation:**
    *   **Extrinsic Methods (Supervised):** Compare clusters to ground truth labels.
        *   Assign most frequent label to each cluster, merge, then compute Precision, Recall, F-score (macro-average).
        *   **B-CUBED Measures:**
            *   $\text{precision}(x) = \frac{\text{No. of items in C(x) with A(x)}}{\text{No. of items in C(x)}}$
            *   $\text{recall}(x) = \frac{\text{No. of items in C(x) with A(x)}}{\text{Total no. of items with A(x)}}$
            *   Overall Precision/Recall/F-score are averages over all items.
    *   **Intrinsic Methods (Unsupervised):** Use only the partition.
        *   **Silhouette Coefficient:** For object $x$ in cluster $C_i$:
            *   $a(x)$: mean distance between $x$ and all other points in $C_i$.
            *   $b(x)$: mean distance between $x$ and all other points in the next nearest cluster.
            *   $s(x) = \frac{b(x) - a(x)}{\max\{a(x), b(x)\}}$ (if $|C_i|>1$). $s(x)=0$ if $|C_i|=1$.
            *   Range: $[-1, 1]$. Close to 1 = well-clustered; close to -1 = misclassified; close to 0 = on border.
            *   Overall Silhouette Coefficient is average over all objects.
            *   *Calculator Tip:* Calculate $a(x)$ and $b(x)$ using distances, then apply formula.
*   **K-Means Algorithm (Representative-based):**
    *   **Objective:** Minimize Within Cluster Sum of Squares (WCSS): $\sum_{i=1}^k \sum_{\bar{X} \in C_i} \|\bar{X} - \bar{Y}_i\|^2$ (squared Euclidean distance).
    *   **Representatives:** Cluster centroids (means), not necessarily data points.
    *   **Algorithm:**
        1.  **Initialization:** Choose $k$ initial cluster representatives (randomly, k-means++).
        2.  **Assignment:** Assign each object to its closest representative.
        3.  **Optimization:** Update representatives to be the mean of assigned objects: $\bar{Y}_j = \frac{1}{|C_j|} \sum_{\bar{X} \in C_j} \bar{X}$.
        4.  Repeat steps 2-3 until convergence.
    *   **Issues:** Sensitive to initial choices (local minima), outliers affect means, Euclidean distance inappropriate for categorical data.
    *   **Initialisation Strategies:** Randomly, Randomly repeat several times (choose best WCSS), Sampling + Hierarchical Clustering, Furthest Points, k-means++.
*   **K-Medoids Algorithm (Representative-based):**
    *   **Differences from K-Means:**
        *   Uses any dissimilarity measure (not just Euclidean).
        *   Representatives (medoids) are actual data points from the dataset.
    *   **Pros:** More interpretable representatives, more robust to noise/outliers, applicable to complex data types.
    *   **Cons:** Slower than k-Means.
*   **K-Medians Algorithm (Representative-based):**
    *   **Differences from K-Means:**
        *   Uses Manhattan distance ($L^1$-norm).
        *   Representatives are component-wise medians of the cluster points.
*   **Hierarchical Clustering:**
    *   **No need to specify $k$.** Creates a hierarchy of clusters (dendrogram).
    *   **Agglomerative (Bottom-up):** Start with singletons, iteratively merge closest clusters.
        *   **Proximity Measures (Linkage):**
            *   **Single-linkage:** $\text{dist}(P,Q) = \min_{\bar{X} \in P, \bar{Y} \in Q} d(\bar{X}, \bar{Y})$ (minimum distance between any two points in different clusters).
            *   **Complete-linkage:** $\text{dist}(P,Q) = \max_{\bar{X} \in P, \bar{Y} \in Q} d(\bar{X}, \bar{Y})$ (maximum distance).
            *   **Group-average linkage:** $\text{dist}(P,Q) = \frac{1}{p \cdot q} \sum_{\bar{X} \in P, \bar{Y} \in Q} d(\bar{X}, \bar{Y})$ (average distance).
    *   **Divisive (Top-down):** Start with one big cluster, repeatedly partition it.
        *   **Bisecting K-Means:** Iteratively splits a cluster into two using k-Means.

### 3.3. Association Pattern Mining

*   **Terminology:**
    *   **Transactions:** Dataset objects (e.g., customer purchases).
    *   **Itemset:** A set of items.
    *   **Support of Itemset $I$ ($\text{sup}(I)$):** Fraction of transactions that contain $I$ as a subset.
        *   *Calculator Tip:* Count transactions containing the itemset, divide by total transactions.
    *   **Frequent Itemset:** Itemset whose support is at least a given frequency threshold $f$.
    *   **Maximal Frequent Itemset:** Frequent itemset for which no superset is frequent.
*   **Association Rules:** $X \implies Y$ (if transaction contains $X$, it's likely to contain $Y$).
    *   **Confidence of Rule $X \implies Y$ ($\text{conf}(X \implies Y)$):** Conditional probability that a transaction contains $Y$ given it contains $X$.
        *   $\text{conf}(X \implies Y) = \frac{\text{sup}(X \cup Y)}{\text{sup}(X)}$.
        *   *Calculator Tip:* Calculate supports, then apply formula.
    *   **Definition of Association Rule:** Rule $X \implies Y$ is an association rule if $\text{sup}(X \cup Y) \ge f$ AND $\text{conf}(X \implies Y) \ge c$.
*   **Properties:**
    *   **Support Monotonicity:** $\text{sup}(J) \ge \text{sup}(I)$ for $J \subseteq I$.
    *   **Downward Closure Property:** Every subset of a frequent itemset is also frequent. (Used for pruning search space).
    *   **Confidence Monotonicity:** $\text{conf}(X_2 \implies I - X_2) \ge \text{conf}(X_1 \implies I - X_1)$ if $X_1 \subset X_2 \subset I$.
*   **Generation Framework:**
    1.  **Phase 1 (Frequent Itemset Generation):** Find all frequent itemsets (e.g., Apriori algorithm).
    2.  **Phase 2 (Association Rule Generation):** From frequent itemsets, generate rules (partition $I$ into $X, Y=I-X$, compute confidence, keep if $\ge c$).

### 3.4. Outlier Detection

*   **Goal:** Identify objects significantly different from others (noise or exception).
*   **Examples:** Credit card fraud, sensor events, medical diagnosis.

## 4. Graph Mining & Social Network Analysis

### 4.1. Graph Terminology

*   **Graph:** $G=(V,E)$ where $V$ is set of nodes (vertices), $E$ is set of edges (links).
*   **Undirected Graph:** Edges are unordered pairs $(u,v) = (v,u)$.
*   **Directed Graph:** Edges are ordered pairs $(u,v) \neq (v,u)$ (called arcs).
*   **Multigraph:** Allows multiple edges between same pair of nodes.
*   **Loop:** Edge connecting vertex to itself $(u,u)$.
*   **Weighted Graph:** Edges/vertices assigned a numerical weight.
*   **Adjacency Matrix ($\bar{A}$):** $n \times n$ matrix for $n$ vertices.
    *   **Undirected:** $\bar{A}_{i,j}=1$ if $(i,j) \in E$, else $0$. Symmetric.
    *   **Directed:** $\bar{A}_{i,j}=1$ if $(i,j) \in E$ (arc from $i$ to $j$), else $0$. Not necessarily symmetric.
    *   **Weighted:** $\bar{A}_{i,j}=w_{i,j}$ (weight) if $(i,j) \in E$, else $0$.
*   **Neighbours & Degree:**
    *   **Neighbourhood $N(v)$:** Set of all neighbours of $v$.
    *   **Degree $\text{deg}(v)$ (undirected):** Number of neighbours of $v$.
    *   **In-degree $\text{deg}_+(v)$ (directed):** Number of in-neighbours (arcs pointing to $v$).
    *   **Out-degree $\text{deg}_-(v)$ (directed):** Number of out-neighbours (arcs pointing from $v$).
*   **Path:** Sequence of distinct vertices $v_0, v_1, \dots, v_k$ where consecutive vertices are adjacent.
    *   **Length:** Number of edges/arcs in path.
*   **Distance:** Length of shortest path between two vertices. $\infty$ if no path.
*   **Connected Graph (undirected):** Path between any pair of vertices.
*   **Connected Component (undirected):** Maximal connected subgraph.
*   **Strongly Connected Graph (directed):** Directed path between any ordered pair of vertices.
*   **Strongly Connected Component (directed):** Maximal strongly connected subgraph.

### 4.2. Measures of Centrality (Undirected Graphs)

*   **Degree Centrality ($C_D(i)$):** Normalized degree.
    *   $C_D(i) = \frac{\text{deg}(i)}{n-1}$ (where $n$ is total nodes).
    *   *Motivation:* Hub nodes, local importance.
*   **Closeness Centrality ($C_C(i)$):** Inverse of average shortest path distance.
    *   $\text{AvDist}(i) = \frac{\sum_{j=1}^n \text{dist}(i,j)}{n-1}$.
    *   $C_C(i) = \frac{1}{\text{AvDist}(i)}$.
    *   *Motivation:* How quickly information can spread from a node.
*   **Betweenness Centrality ($C_B(i)$):** Fraction of shortest paths between other pairs of nodes that pass through node $i$.
    *   $f_{jk}(i) = \frac{q_{jk}(i)}{q_{jk}}$ ($q_{jk}(i)$ is number of shortest paths between $j,k$ passing through $i$; $q_{jk}$ is total shortest paths between $j,k$).
    *   $C_B(i) = \frac{\sum_{j<k} f_{jk}(i)}{\binom{n}{2}}$.
    *   *Motivation:* Control over information flow, critical nodes.

### 4.3. Measures of Prestige (Directed Graphs)

*   **Degree Prestige ($P_D(i)$):** Normalized in-degree.
    *   $P_D(i) = \frac{\text{deg}_+(i)}{n-1}$.
    *   *Motivation:* Popularity (votes).
*   **Proximity Prestige ($P_P(i)$):** Combines influence and average distance.
    *   $\text{Influence}(i)$: Set of nodes that can reach node $i$ with a direct path.
    *   $\text{AvDist}(i) = \frac{\sum_{j \in \text{Influence}(i)} \text{dist}(j,i)}{|\text{Influence}(i)|}$.
    *   $\text{InfluenceFraction}(i) = \frac{|\text{Influence}(i)|}{n-1}$.
    *   $P_P(i) = \frac{\text{InfluenceFraction}(i)}{\text{AvDist}(i)}$.
    *   *Motivation:* How much a node is influenced by others, considering distance.

### 4.4. PageRank Algorithm

*   **Goal:** Rank web pages (or nodes in any graph) based on importance.
*   **Idea:** A page is important if it is linked by other important pages.
*   **Markov Chain Perspective:** Random surfer model.
    *   **State:** Each webpage.
    *   **Transition:** Hyperlink.
    *   **Transition Probability Matrix ($\bar{A}$):** $\bar{A}_{i,j} = \frac{1}{O_i}$ if $(i,j) \in E$ (link from $i$ to $j$), else $0$. ($O_i$ is out-degree of page $i$).
*   **Problems with real Web graph:**
    1.  **Not Stochastic:** Dangling pages (no out-links) cause rows to sum to 0.
    2.  **Not Irreducible:** Disconnected components.
    3.  **Not Aperiodic:** Cycles.
*   **Modifications to $\bar{A}$ (to make it stochastic, irreducible, aperiodic):**
    1.  **Handle Dangling Pages:** For dangling page $i$, add outgoing links to all pages with probability $1/n$.
    2.  **Add Random Jumps (Damping Factor $d$):** With probability $d$, surfer follows a link. With probability $1-d$, surfer jumps to a random page.
        *   Modified transition matrix: $\bar{A}' = (1-d)\frac{\bar{E}}{n} + d\bar{A}^T$ (where $\bar{E}$ is matrix of all 1s).
*   **PageRank Vector ($\bar{P}$):** The stationary probability distribution of the modified Markov chain.
    *   $\bar{P} = (\bar{A}')^T \bar{P}$ (eigenvector with eigenvalue 1).
*   **Power Iteration Algorithm:** Iteratively compute $\bar{P}_k = (\bar{A}')^T \bar{P}_{k-1}$ until convergence ($\|\bar{P}_k - \bar{P}_{k-1}\|_1 \le \epsilon$).
    *   **Initial $\bar{P}_0$:** Usually uniform, e.g., $P_0(i) = 1/n$ for all $i$.
    *   **Typical $d$ value:** $0.85$.

## 5. Evaluation Metrics (for Classifiers)

*   **Gold Standard (Test Data):** Dataset with correct labels for evaluation. **NEVER TRAIN ON TEST DATA!**
*   **Confusion Matrix (Error Matrix):**
    |                      |  **Actual YES(+)**   |   **Actual NO(-)**   |
    | :------------------- | :------------------: | :------------------: |
    | **Predicted YES(+)** | True Positives (TP)  | False Positives (FP) |
    | **Predicted NO(-)**  | False Negatives (FN) | True Negatives (TN)  |
    *   **TP:** Predicted positive, actual positive.
    *   **TN:** Predicted negative, actual negative.
    *   **FP:** Predicted positive, actual negative (Type I error).
    *   **FN:** Predicted negative, actual positive (Type II error).
*   **Measures:**
    *   **Accuracy:** Proportion of correctly classified objects.
        *   $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
    *   **Precision:** Proportion of predicted positives that are truly positive.
        *   $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
    *   **Recall (Sensitivity):** Proportion of actual positives that are correctly classified.
        *   $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
    *   **F-score (F1-score):** Harmonic mean of Precision and Recall.
        *   $\text{F-score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
        *   $\text{F-score} = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}$
    *   **Precision-Recall Trade-off:** Improving one often lowers the other.
*   **Measures for Multiple Classes (Class A):**
    *   $\text{Precision}_A = \frac{\text{no. objects correctly classified A}}{\text{no. objects classified A}}$
    *   $\text{Recall}_A = \frac{\text{no. objects correctly classified A}}{\text{no. objects that belong to class A}}$
    *   $\text{F-score}_A = \frac{2 \times \text{Precision}_A \times \text{Recall}_A}{\text{Precision}_A + \text{Recall}_A}$
    *   **Macro F-score:** Average of F-scores for all classes.
        *   $\text{Macro F-score} = \frac{1}{C} \sum_{i=1}^C \text{F-score}_i$ (where $C$ is number of classes).
        *   *Calculator Tip:* Calculate individual F-scores, then average.

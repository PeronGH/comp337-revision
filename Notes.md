# COMP337/COMP527 - Data Mining and Visualisation: Exam Revision Notes

## Part 1: Introduction to the Module & Data Mining Fundamentals

### 1. Introduction to the Module (Procheta Sen)

*   **Lecturer:** Procheta Sen
    *   Lecturer at the Department of Computer Science since July 2022.
    *   Postdoc at University College London.
    *   Research interests: Natural Language Processing (NLP), Explainability of Large Language Models.
*   **Module Code:** COMP337/COMP527
*   **Module Title:** Data Mining and Visualisation

#### 1.1. Schedule
*   **Week materials release:** Monday
*   **Lab sessions:** Monday, Tuesday, Thursday, Friday

#### 1.2. Materials
*   Reading materials (use module’s Reading List)
*   Quizzes / Problem sets
*   Lab tasks

#### 1.3. Lab Sessions & Tasks
*   **Lab tasks release:** Monday
*   Work on tasks (individually or in groups) during lab sessions.
*   Continue working on lab tasks during the week.
*   **Solutions:** Released on Canvas on Friday.
*   **Questions:** Can be asked on the next Monday.

#### 1.4. Office Hours
*   **Time:** Friday 14.00-15.00
*   **Office Address:** Ashton Building, Ground Floor, Room No. G.12B
*   **Communication:**
    *   Reply to emails
    *   Individual one-to-one discussions via MS Teams chats
    *   `campuswire.com` / shared google doc link

#### 1.5. Assessments
*   **Two assessments during the course:**
    | Assessment    | Contribution to final mark | Release date    | Submission deadline | Notes                                                        |
    | :------------ | :------------------------- | :-------------- | :------------------ | :----------------------------------------------------------- |
    | Class Test    | 15%                        | Friday 21 Feb   | Friday 21 Feb       |                                                              |
    | Course work 2 | 15%                        | Monday 17th Mar | Wednesday 7th May   |                                                              |
    | Final Exam    | 70%                        | Date: N/A       | N/A                 | A useful link with answers to common questions is available [here]. It includes a direct link to the timetable and information about available support. |
*   **Individual Work Policy:**
    1.  **Do it yourself.**
    2.  **Don’t share your work with others.**

#### 1.6. Learning Objectives
*   Learning objectives are explicit statements of observable actions students should be able to complete.
*   **Key Learning Objectives (Examples from slide):**
    *   Identify the main phases of the data mining process.
    *   Multiply, add matrices and vectors (by hand and with python).
    *   Compute the derivative of a function.
    *   Compute the gradient of a function.
    *   Find a minimum of a function using the gradient descent method (with python).
*   Consider Learning Objectives as a study guide.
*   Ensure achievement of Learning Objectives to perform well in assessments.

### 2. Introduction to Data Mining

#### 2.1. What is Data Mining?
*   Automated analysis of massive data.

#### 2.2. Why Data Mining?
*   Exponential growth of data (terabytes to petabytes).
*   Major sources of abundant data (e.g., social media, YouTube).
*   Data collection has become easier.

#### 2.3. The Data Mining Process
A sequential process with an optional feedback loop:

1.  **Data Collection:**
    *   Gathering raw data.
    *   Highly application-specific (e.g., sensor networks, user surveys, automatically collected documents).
    *   Critically important: significantly impacts the whole process.
    *   Data can be stored in structured or flat format.
2.  **Data Preprocessing:** Transforming raw data into an understandable format.
    *   **Feature Extraction:**
        *   Convert data into a format friendly to data mining algorithms.
        *   Analyst abstracts out features most relevant to a particular application.
        *   Formats: Multidimensional, Time series, Semistructured, etc.
        *   **Multidimensional format:** Every data point (object) is characterized by a sequence of fields (features, attributes, or dimensions).
        *   **Features:**
            *   Central to data mining.
            *   Allow abstraction of data points and learning rules for prediction.
            *   Example: `if colour == red then flower = rose`. This rule can classify flowers beyond the training set.
            *   Coming up with good features is an art (feature engineering: `https://en.wikipedia.org/wiki/Feature_engineering`).
            *   Deep learning focuses on automatically discovering good features.
    *   **Data Cleaning:**
        *   Handle missing and erroneous parts of the data.
        *   Extracted data may have erroneous or missing fields.
        *   Possible actions:
            *   Drop a record.
            *   Estimate a missing value.
            *   Remove inconsistencies.
    *   **Feature Selection & Transformation:**
        *   Choosing meaningful and effective features is important.
        *   Remove irrelevant features.
        *   Transform existing features to features of different scale or format.
        *   **Data transformation example:** Numerical age -> {young, middle-aged, elderly}.
        *   **Feature Pruning:**
            *   Remove irrelevant features (uncorrelated with prediction) or redundant features.
            *   Text data: words appearing almost always or never.
            *   Numerical data: low variance features.
3.  **Analytical Processing:**
    *   Design and apply analytical methods to preprocessed data.
    *   The core of data mining, where algorithms are applied.
    *   Output for analysts (e.g., reports, visualizations).
    *   **Main types of subproblems:**
        *   Association pattern mining
        *   Clustering
        *   Classification
        *   Outlier detection
4.  **Feedback (optional):** Results from analytical processing can feedback to earlier stages for refinement.

**Objects and Features:**
*   **Object (Record, Point, Case, Sample, Entity, Instance):** An individual data item. Described by a collection of attributes.
*   **Feature (Attribute, Variable, Field, Characteristic, Dimension):** A property or characteristic of an object.

**Example of Data Mining Process: Retailer Web Logs**
*   **Task:** Design a system of targeted product recommendations using customer demographics and buying behavior.
*   **Data Collection:** Extract records from a log file (e.g., IP address, timestamp, product page accessed).
*   **Data Preprocessing:**
    *   **Feature Extraction:**
        *   Create one record per customer: CustomerID, Location (from IP), Product1_access_count, Product2_access_count, ..., ProductN_access_count.
        *   Map Geographic Locations from IP Address.
    *   **Cleaning:** Remove log records useless for user identification.
    *   **Feature Selection & Transformation:** (Implicit in the feature extraction choices).
    *   Result: A cleaned dataset (table of CustomerId, Location, Product access counts).
*   **Analytical Processing:**
    *   Determine "similar" groups of customers by **clustering**.
    *   For a given customer, recommend the most frequent items accessed by customers from their group.

### 3. Types of Data

#### 3.1. Two General Classes
1.  **Nondependency-oriented data:** Objects do not have dependencies.
    *   Simplest form of data.
    *   A multidimensional data set $\mathcal{D}$ typically contains a set of records $\bar{X_1}, \dots, \bar{X_n}$.
    *   Each record $\bar{X_i}$ contains a set of $d$ features $(x_i^1, \dots, x_i^d)$.
    *   Can be represented by an $n \times d$ data matrix.
    *   Example: Table with ID, Height, Weight, Marital Status, Employed.
2.  **Dependency-oriented data:** Implicit or explicit dependencies between objects may exist.
    *   **Implicit dependencies:** Not explicitly specified but known to exist.
        *   Example: Temperature values collected by a sensor over time.
        *   **Types with implicit dependencies:**
            *   **Time-series:** Values generated by sequential measurements over time (timestamp/index is contextual, measurement is behavioral).
            *   **Discrete Sequences and Strings:** Categorical analog of time-series.
            *   **Spatial data:** Every record has a location attribute (e.g., temperature, pressure at spatial locations).
            *   **Spatiotemporal data:** Contain both spatial and temporal attributes.
    *   **Explicit dependencies:**
        *   Example: Networks where nodes (objects) are connected by edges (relationships).
        *   **Types with explicit dependencies (Network/Graph data):**
            *   Objects correspond to nodes.
            *   Relationships correspond to edges (directed or undirected).
            *   Nodes/edges can have attributes.
            *   Examples: Web graph, Facebook/Instagram/LinkedIn social networks.

#### 3.2. Specific Data Types
*   **Numerical or quantitative:** Values have natural ordering.
    *   Integer values (e.g., number of petals).
    *   Real values (e.g., length of a petal).
*   **Categorical or unordered discrete-valued:** Discrete unordered values/categories.
    *   Example: Colour of a flower petal.
*   **Binary data:** Two values (0 and 1).
    *   Can be seen as categorical (two categories) or numerical (0 < 1).
    *   Can represent Set Data via characteristic vectors.
*   **Text data:**
    *   Document as a string (dependency-oriented).
    *   Document as a set of words or terms (vector-space representation: frequencies of words).

### 4. Data Representation

*   One of the first things to do in data mining.
*   What can be mined is largely determined by data representation.
*   No single best representation method for all tasks.

#### 4.1. Representing Categorical Data
*   Represent each category as a separate dimension of a vector (One-Hot Encoding).
*   Example: For categories {awesome, burger, terrible}:
    *   `(awesome=1, burger=1, terrible=0)`
    *   `(awesome=0, burger=1, terrible=1)`

#### 4.2. Representing Text Data: Example
Sentence: "The burger I ate was an awesome burger!"
*   **Method 1: List of words:** `["the", "burger", "i", "ate", "was", "an", "awesome", "burger"]` (Order preserved, duplicates kept)
*   **Method 2: Set of words (Bag of Words):** `{"the", "burger", "i", "ate", "was", "an", "awesome"}` (Order lost, duplicates removed)
*   **Method 3: Vector of word frequency (Term Frequency):** `("the":1, "burger":2, "i":1, "ate":1, "was":1, "an":1, "awesome":1)`
*   **Method 4: Vector of letter frequency:** `{'a': 4, ' ': 7, 'b': 2, 'e': 6, ...}`

### 5. Main Data Mining Problems

Four fundamental problems:
1.  **Association Pattern Mining:**
    *   Discover interesting relationships (associations/correlations) among items in a large dataset.
    *   Special case: **Frequent Pattern Mining** (binary data sets).
        *   Given an $n \times d$ data matrix, identify all subsets of columns (features) such that at least a fraction $s$ (support threshold) of rows (objects) have all these features enabled (value of 1).
        *   Example: Transactions (Milk, Butter, Bread, Mushrooms, Onion, Carrot). If $s=0.65$, and {Milk, Butter, Bread} appear together in $\ge 65\%$ of transactions, they are frequently bought together.
2.  **Classification:**
    *   Goal: Use **training data** to learn relationships between a fixed feature (called **class label**) and remaining features.
    *   The resulting **learned model** is used to predict class labels for **test objects** (test data) where the label is unknown.
    *   **Supervised learning.**
    *   Example Algorithms: Decision Tree, Naive Bayes.
    *   Examples: Targeted marketing, text recognition.
    *   Illustration: SVM classification plot separating data points (e.g., salary vs. age) into regions.
3.  **Clustering:**
    *   Given a data set (data matrix), partition its objects (rows) into sets (clusters) $C_1, C_2, \dots, C_k$ such that objects in each cluster are "similar" to one another and dissimilar to objects in other clusters.
    *   Specific definitions depend on how **similarity** is defined.
    *   **Unsupervised learning** (no predefined class labels).
    *   Primary objective: Increase intra-class similarity and minimize inter-class similarity.
    *   Examples: Customer segmentation, data summarization.
    *   Illustration: Plot showing data points grouped into distinct clusters (e.g., BMI vs. Age).
4.  **Outlier Detection:**
    *   Given a data set, determine the **outliers**, i.e., objects significantly different from remaining objects.
    *   Can be noise or an exception (potentially interesting).
    *   Examples: Credit card fraud, detecting sensor events, medical diagnosis, earth science.
    *   Illustration: Box plot showing outlier values for BMI.

### 6. Data Quality Issues

#### 6.1. Missing Values
*   Occur when the value of a particular feature for an instance is unknown.
*   Example: Height of 10th student is missing. Important if height is essential for obesity classification.

    **Handling Missing Values:**
    1.  **Discard the entire training instance:**
        *   May work if data is redundant.
        *   Risky if the instance is unique or important for a class (e.g., only instance of an obese student).
    2.  **Fill in values by hand (re-annotate/re-measure):**
        *   Reliable but often impractical (slow, costly, subjects may be inaccessible).
    3.  **Set "missingValue" as a distinct category:**
        *   Treat "missing" as another category for the feature.
        *   Not directly applicable to numerical data (is 0 measured or missing?).
        *   Doesn't truly solve the problem.
    4.  **Replace with the mean (or median/mode):**
        *   Compute the mean of available feature values for the entire training dataset and use it.
        *   Good if missing data points are representative.
        *   Inaccurate if data points are outliers.
    5.  **Predict the missing value:**
        *   Train a classifier to predict missing values. Then use the completed dataset to train the main classifier.
    6.  **Accept missing values:**
        *   Let the algorithm (e.g., classifier) handle them.
        *   Some algorithms can inherently ignore missing values or use features without missing values.

#### 6.2. Noisy Data
*   Random errors scattered in the data (e.g., inaccurate recording, data corruption).
*   **Problem:** Can lead to **Overfitting**.
    *   If noisy values are assumed correct, the model might learn the noise.
    *   Overfitting: Model learns "too much" from training data (including noise) and doesn't generalize well to unseen test data.

    **Detecting Noisy Data:**
    *   Obvious noise: Incorrect data type (string in numeric attribute), value very dissimilar to others (e.g., feature value 10 when range is [0,1]).
    *   Non-obvious noise: Typo (0.52 instead of 0.25).

    **Handling Noisy Values:**
    1.  **Manual inspection and removal.**
    2.  **Clustering:** Find instances/features outside main clusters (outlier detection) and remove them.
    3.  **Linear regression:** Determine function, remove points far from predicted value.
    4.  **Frequency threshold:** Ignore values occurring below a certain frequency (effective for misspellings in text).
    5.  If noisy points identified and removed, use missing value techniques to fill gaps.

### 7. Overfitting vs. Underfitting

*   **Overfitting:**
    *   Model $M$, trained on $D_{train}$, performs well on $D_{train}$ but poorly on a separate $D_{test}$.
    *   Typically: 90-99% accuracy on $D_{train}$, 40-60% on $D_{test}$ (for balanced binary classification).
    *   Cause: Model $M$ has too many parameters ("too much flexibility"), fits noise in $D_{train}$.
    *   **Solutions to Overfitting:**
        *   **Reduce model flexibility:**
            *   Regularization (L1, L2).
            *   Remove features.
        *   **Early stopping:** Premature termination of training to prevent parameter overfitting.
        *   **Training with more data.**
        *   **Cross-validation.**
*   **Underfitting:**
    *   Poor performance on $D_{train}$ because the model is not sufficiently "fitted" to the train data.
    *   **Solutions to Underfitting:**
        *   **Learning has not converged:** Let training proceed for more iterations.
        *   **Feature space too small/inadequate:** Implement more/better features.
        *   **Train data is bad/noisy/missing values:** Cleanse/re-annotate train data.
        *   **Algorithm not training well:** Select a different training algorithm.

### 8. Feature Normalisation (Scaling)
Transforms numerical features into a common scale. Important for distance-based algorithms or gradient descent.

1.  **[0,1]-scaling (Min-Max Scaling):**
    *   Formula: $\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$
    *   $\min(x)$ and $\max(x)$ are computed over all training data points.
    *   Scaled feature values will be in the interval $[0,1]$.
2.  **Gaussian Normalisation (Standardization or Z-score Normalization):**
    *   Formula: $\hat{x} = \frac{x - \mu}{\sigma}$
    *   $\mu$ (mean) and $\sigma$ (standard deviation) are computed over all training data points.
    *   Transformed feature will have zero mean and unit variance.
    *   Makes it "easier" to compare features, ignoring their absolute scales.

## Part 2: Mathematical Preliminaries

### 9. Linear Algebra

*   Data points are often represented using **vectors** (ordered sets of coordinates/features).
*   Reference: Chapter 02 of MML book (`https://mml-book.github.io/book/mml-book.pdf`)

#### 9.1. Vectors
*   Notation: $\bar{X}, \bar{Y}, \bar{W}$ (uppercase letters with a bar).
*   Typically column vectors. Row vectors are transposes, e.g., $\bar{X} = (3.2, -9.1, 0.1)^T$.
*   $\bar{X} \in \mathbb{R}^d$ means $\bar{X}$ is a $d$-dimensional vector with real coordinates.

#### 9.2. Matrices
*   Collections of vectors arranged by columns or rows. Notation: $\bar{M}$.
*   $\bar{M} \in \mathbb{R}^{n \times m}$ means $\bar{M}$ is a matrix with $n$ rows and $m$ columns.
*   If $n=m$, $\bar{M}$ is **square**.
*   $(i,j)$ element of $\bar{M}$ is $\bar{M}_{i,j}$.
*   If $\bar{M}_{i,j} = \bar{M}_{j,i}$ for all $i,j$, $\bar{M}$ is **symmetric**. Otherwise, asymmetric.

#### 9.3. Vector Arithmetic
Given $\bar{X} = (x_1, \dots, x_d)^T$ and $\bar{Y} = (y_1, \dots, y_d)^T$:
*   **Addition:** $\bar{Z} = \bar{X} + \bar{Y}$, where $z_i = x_i + y_i$.
*   **Inner-product (dot product):** $\bar{X}^T \bar{Y} = \sum_{i=1}^{d} x_i y_i$.
*   **Outer-product:** $\bar{X} \bar{Y}^T$ is a $d \times d$ matrix $\bar{M}$ where $\bar{M}_{i,j} = x_i \cdot y_j$.

#### 9.4. Matrix Arithmetic
*   **Addition:** Matrices of the same shape can be added element-wise: $\bar{A} + \bar{B} = \bar{C}$, where $\bar{C}_{i,j} = \bar{A}_{i,j} + \bar{B}_{i,j}$.
*   **Multiplication:** If $\bar{A} \in \mathbb{R}^{n \times m}$ and $\bar{B} \in \mathbb{R}^{m \times d}$, then $\bar{C} = \bar{A}\bar{B}$ is an $n \times d$ matrix where $\bar{C}_{i,j} = \sum_{k=1}^{m} \bar{A}_{i,k} \bar{B}_{k,j}$.

#### 9.5. Transpose and Inverse
*   **Transpose:** Transpose of $\bar{A} \in \mathbb{R}^{n \times d}$ is $\bar{A}^T \in \mathbb{R}^{d \times n}$, where $(\bar{A}^T)_{i,k} = \bar{A}_{k,i}$.
    *   Property: $(\bar{A}\bar{B})^T = \bar{B}^T \bar{A}^T$.
*   **Inverse:** For a square matrix $\bar{A} \in \mathbb{R}^{n \times n}$, its inverse $\bar{A}^{-1}$ satisfies $\bar{A}\bar{A}^{-1} = \bar{A}^{-1}\bar{A} = \bar{I}$, where $\bar{I}$ is the unit matrix (1s on diagonal, 0s elsewhere).
    *   Only full-rank square matrices are invertible.

#### 9.6. Linear Independence
*   A vector $\bar{V}$ formed as $\bar{V} = \lambda_1 \bar{X_1} + \dots + \lambda_k \bar{X_k} = \sum_{i=1}^{k} \lambda_i \bar{X_i}$ is a **linear combination** of $\bar{X_1}, \dots, \bar{X_k}$.
*   Vectors $\bar{X_1}, \dots, \bar{X_k}$ are **linearly dependent** if there exist $\lambda_1, \dots, \lambda_k$, not all zero, such that $\bar{0} = \lambda_1 \bar{X_1} + \dots + \lambda_k \bar{X_k}$.
*   Otherwise, they are **linearly independent**.

#### 9.7. Rank
*   The **rank** of a matrix $\bar{A} \in \mathbb{R}^{m \times n}$ (denoted $rank(\bar{A})$) is the number of linearly independent columns (or rows).
*   $rank(\bar{A}) \le \min(m,n)$. If $m \le n$, then $rank(\bar{A}) \le m$.
*   If $rank(\bar{A}) = m$ (assuming $m \le n$), $\bar{A}$ is **full-rank**. Otherwise, **rank-deficient**.
*   Only full-rank square matrices are invertible.

#### 9.8. Matrix Trace
*   The sum of diagonal elements of a square matrix $\bar{A}$: $tr(\bar{A}) = \sum_i \bar{A}_{i,i}$.
*   Example: For $\bar{A} = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$, $tr(\bar{A}) = 1+1+0 = 2$.

#### 9.9. Eigenvalues and Eigenvectors
*   Let $\bar{A} \in \mathbb{R}^{n \times n}$ be a square matrix.
*   A non-zero vector $\bar{X} \in \mathbb{R}^n$ is an **eigenvector** of $\bar{A}$ if $\bar{A}\bar{X} = \lambda\bar{X}$ for some scalar $\lambda \in \mathbb{R}$.
*   $\lambda$ is called the **eigenvalue** of $\bar{A}$ corresponding to $\bar{X}$.

### 10. Differential Calculus

*   Reference: Chapter 06 of MML book.

#### 10.1. Derivatives of Basic Functions
*   $\frac{d}{dx} a = 0$ (where $a$ is a constant)
*   $\frac{d}{dx} x^a = a \cdot x^{a-1}$
*   $\frac{d}{dx} e^x = e^x$ (where $e \approx 2.71$ is Euler's number)
*   $\frac{d}{dx} \log(x) = \frac{1}{x}$ (for $x > 0$)
*   $\frac{d}{dx} \sin(x) = \cos(x)$
*   $\frac{d}{dx} \cos(x) = -\sin(x)$
*   Note: $\frac{d}{dx} f(x)$ is the same as $f'(x)$.

#### 10.2. Differentiation Rules
*   **Sum rule:** $(\alpha f + \beta g)' = \alpha f' + \beta g'$
*   **Product rule:** $(fg)' = f'g + fg'$
*   **Quotient rule:** $(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$
*   **Chain rule:** If $f(x) = h(g(x))$, then $f'(x) = h'(g(x)) \cdot g'(x) = \frac{d}{dg(x)}h \cdot \frac{d}{dx}g$

#### 10.3. Partial Derivative
*   A partial derivative of a function of several variables is its derivative with respect to one variable, with others held constant.
*   Example: $f(x,y) = 5x + y^2$
    *   $\frac{\partial f}{\partial x} = 5$
    *   $\frac{\partial f}{\partial y} = 2y$
    *   Gradient: $\nabla_{(x,y)}f = (5, 2y)^T$

### 11. Optimisation

*   Reference: Chapter 07 of MML book.
*   **Continuous optimisation:**
    *   Unconstrained optimisation
    *   Constrained optimisation

#### 11.1. Unconstrained Optimisation: Gradient Descent Method
*   **Problem formulation:** Find $\min_{\bar{X}} f(\bar{X})$
    *   where $\bar{X} = (x_1, x_2, \dots, x_d)$ and $f: \mathbb{R}^d \to \mathbb{R}$.
    *   $f$ is differentiable, and analytical solution in closed form is hard.
*   **Gradient:** For $f(\bar{X}) = f(x_1, \dots, x_d)$, the gradient is $\nabla_{\bar{X}} f = \frac{\partial f}{\partial \bar{X}} = (\frac{\partial f(\bar{X})}{\partial x_1}, \frac{\partial f(\bar{X})}{\partial x_2}, \dots, \frac{\partial f(\bar{X})}{\partial x_d})^T$.
*   The gradient $\nabla_{\bar{X}} f$ evaluated at $\bar{X_0}$ points in the direction of steepest ascent.
*   Moving in the direction of the **negative gradient** $-\nabla_{\bar{X}} f(\bar{X_0})$ decreases $f$ fastest.
*   For a small step-size $\gamma \ge 0$, $\bar{X_1} = \bar{X_0} - \gamma \cdot (\nabla_{\bar{X}} f)(\bar{X_0})$ gives $f(\bar{X_1}) \le f(\bar{X_0})$.
*   **Algorithm for finding local minimum of $f(\bar{X})$:**
    1.  Pick an initial point $\bar{X_0}$.
    2.  Iterate: $\bar{X}_{i+1} = \bar{X}_i - \gamma_i \cdot ((\nabla_{\bar{X}} f)(\bar{X}_i))$.
    3.  For suitable step-sizes $\gamma_1, \gamma_2, \dots$, the sequence $f(\bar{X_0}) \ge f(\bar{X_1}) \ge \dots$ converges to a local minimum.
*   **Moral:** Gradient is a useful tool for finding local optimal points.

#### 11.2. Constrained Optimisation: Method of Lagrange Multipliers
*   **Problem formulation:** Find $\min_{\bar{X}} f(\bar{X})$ subject to $g(\bar{X}) = 0$.
    *   $\bar{X} = (x_1, \dots, x_d)$, $f: \mathbb{R}^d \to \mathbb{R}$, $g: \mathbb{R}^d \to \mathbb{R}$.
    *   $f$ is differentiable, analytical solution hard.
*   **Solution Steps:**
    1.  Form the **Lagrangian function:** $\mathcal{L}(\bar{X}, \lambda) = f(\bar{X}) - \lambda \cdot g(\bar{X})$.
    2.  Find all **stationary points** $(\bar{X_0}, \lambda_0)$ of $\mathcal{L}(\bar{X}, \lambda)$ by setting all partial derivatives to 0: $\nabla_{(\bar{X},\lambda)}\mathcal{L} = \bar{0}$.
        *   This means $\frac{\partial \mathcal{L}}{\partial x_i} = 0$ for all $i$, and $\frac{\partial \mathcal{L}}{\partial \lambda} = 0$ (which recovers $g(\bar{X})=0$).
    3.  Examine stationary points to find the solution.

### 12. Probability

#### 12.1. Common Discrete Probability Distributions
*   **Bernoulli distribution:** Models binary outcomes (e.g., coin flip).
    *   $P(X=\text{head}) = p$, $P(X=\text{tail}) = 1-p$.
*   **Generalised Bernoulli distribution (Categorical distribution):** Models $k > 2$ outcomes (e.g., $k$-sided die).
    *   $P(X=1)=p_1, P(X=2)=p_2, \dots, P(X=k)=p_k$, such that $\sum_{i=1}^k p_i = 1$.
*   **Binomial distribution:** Models a sequence of $n$ multiple independent Bernoulli trials (flips of a coin).
    *   $P(\text{in } n \text{ flips there are exactly } k \text{ heads}) = \binom{n}{k} p^k (1-p)^{n-k}$.
*   **Multinomial distribution:** Models a sequence of $n$ multiple independent generalised Bernoulli trials (rolls of a $k$-sided die, $k>2$).
    *   If there are $n$ rolls and $n_i$ is the number of times side $i$ came up (where $\sum n_i = n$), then the probability of this specific sequence of outcomes is $\frac{n!}{\prod_{i=1}^k n_i!} \cdot \prod_{i=1}^k p_i^{n_i}$.

## Part 3: Classification Algorithms

### 13. Perceptron

*   A binary classification algorithm.
*   **Bio-inspired model:**
    *   Neural networks simulate the human nervous system.
    *   Nervous system composed of nerve cells (neurons) connected at synapses.
    *   Learning occurs by changing synaptic connection strengths in response to external stimuli.
    *   **Perceptron is a model of a single neuron.**

#### 13.1. Perceptron Model
*   Inputs $x_1, \dots, x_5$ are weighted by $w_1, \dots, w_5$.
*   **Activation score:** $a = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + w_5x_5$.
*   If score $a > \theta$ (threshold), neuron fires (Output = 1). Else, Output = -1.
*   Computation function at a neuron is defined by **weights**.
*   Weights correspond to strengths of synaptic connections.
*   Computation function is **learned** by changing weights.
*   "External stimulus" is provided by **training data**.
*   **Idea:** Incrementally modify weights whenever incorrect predictions are made.

#### 13.2. Mathematical Notation
*   Input object: $\bar{X}^T = (x_1, x_2, \dots, x_d)$.
*   Weights: $\bar{W}^T = (w_1, w_2, \dots, w_d)$.
*   Activation score: $a = \sum_{i=1}^d w_i x_i = \bar{W}^T \bar{X}$.
*   Output 1 if $a > \theta$, Output -1 if $a \le \theta$.

#### 13.3. Bias Term
*   Convenient to make threshold $\theta = 0$.
*   Achieved by introducing a bias term $b = -\theta$.
*   Activation: $a = b + \sum_{i=1}^d w_i x_i$.
*   Output 1 if $a > 0$, Output -1 if $a \le 0$.
*   Equivalently, output $sign(\bar{W}^T \bar{X} + b)$.

#### 13.4. Notational Trick for Bias
*   Introduce a feature $x_0$ that is always ON (i.e., $x_0=1$ for all objects).
*   Squeeze bias term $b$ into the weight vector by setting $w_0 = b$.
*   Activation: $a = \sum_{i=0}^d w_i x_i = \bar{W}^T \bar{X}$ (where $\bar{X}$ now includes $x_0=1$ and $\bar{W}$ includes $w_0=b$).
*   This is elegant, but remember the bias term is still conceptually present.

#### 13.5. Perceptron Training Algorithm
`PerceptronTrain(Training data: D, MaxIter)`
1.  Initialize weights and bias: $w_i = 0$ for all $i=1, \dots, d$; $b=0$.
    (If using notational trick, $w_i=0$ for $i=0, \dots, d$).
2.  For `iter` = 1 to `MaxIter` do:
3.  For each training instance $(\bar{X}, y) \in D$ (where $y \in \{+1, -1\}$) do:
4.  Compute activation score: $a = \bar{W}^T \bar{X} + b$.
5.  If $y \cdot a \le 0$ (misclassification):
6.  Update weights: $w_i = w_i + y \cdot x_i$ for all $i=1, \dots, d$.
7.  Update bias: $b = b + y$.
    (If using notational trick, $w_i = w_i + y \cdot x_i$ for $i=0, \dots, d$, where $x_0=1$).
8.  Return $b, w_1, \dots, w_d$.

#### 13.6. Perceptron Test Algorithm
`PerceptronTest(b, w_1, \dots, w_d, \bar{X}_{test})`
1.  $a = \bar{W}^T \bar{X}_{test} + b$.
2.  Return $sign(a)$.

#### 13.7. Important Features of Perceptron
*   **Online algorithm:** Processes objects from training data one by one (vs. batch learning like k-NN).
*   **Error-driven:** Parameters updated **only** when a training object is misclassified.

#### 13.8. Detecting Misclassification
*   Predicted class $sign(a)$ is different from true class $y$ if and only if $y \cdot a \le 0$.
    *   If $y=+1$ and $a \le 0 \implies y \cdot a \le 0$ (misclassified).
    *   If $y=-1$ and $a > 0 \implies y \cdot a < 0$ (misclassified).

#### 13.9. Update Rule - Intuitive Explanation
Perceptron update rule: $\bar{W} = \bar{W} + y\bar{X}$, $b = b + y$.
*   **If incorrectly classify a positive instance ($y=+1$) as negative ($a \le 0$):**
    *   We need a higher activation.
    *   Increase $\bar{W}^T\bar{X}$ and $b$ by adding $\bar{X}$ to $\bar{W}$ (since $y=+1$) and $1$ to $b$.
*   **If incorrectly classify a negative instance ($y=-1$) as positive ($a > 0$):**
    *   We need a lower activation.
    *   Decrease $\bar{W}^T\bar{X}$ and $b$ by subtracting $\bar{X}$ from $\bar{W}$ (since $y=-1$) and $-1$ from $b$.

#### 13.10. Update Rule - Math Explanation
Consider $y=+1$ and $a \le 0$ (misclassification).
Current parameters: $b, w_1, \dots, w_d$. Incoming object: $(\bar{X}, y=+1)$.
New parameters: $b' = b+y = b+1$, $w_i' = w_i + y \cdot x_i = w_i + x_i$.
New activation $a'$ for this $\bar{X}$:
$a' = \sum w_i' x_i + b' = \sum (w_i+x_i)x_i + (b+1)$
$a' = (\sum w_i x_i + b) + \sum x_i^2 + 1 = a + \sum x_i^2 + 1$.
Since $\sum x_i^2 \ge 0$, $a' > a$. The activation score for this instance increases.

#### 13.11. Remarks on Perceptron
*   **Activation Adjustment:** No guarantee of correct classification in the next round for the same instance; adjustment might not be sufficient. (Passive Aggressive Classifier enforces this more strongly).
*   **Ordering of Training Instances:** Matters. Showing all positives then all negatives is bad. Random ordering within each iteration is good practice.
*   **Hyperparameter `MaxIter`:** Chosen experimentally.
    *   Too many passes: likely to **overfit**.
    *   Too few passes: might **underfit**.

#### 13.12. Geometric Interpretation
*   Decision boundary: $\bar{W}^T\bar{X} + b = 0$. This defines a **hyperplane**.
    *   In 2D (ignoring bias): $w_1x_1 + w_2x_2 = 0$ is a line through the origin.
    *   In N-D space: $(N-1)$-dimensional hyperplane.
*   The weight vector $\bar{W}$ is **perpendicular** to the hyperplane $\bar{W}^T\bar{X}=0$.
*   If a positive instance $\bar{X}$ ($y=+1$) is misclassified, its dot product with $\bar{W}$ is negative (angle > 90°).
*   Update: $\bar{W}' = \bar{W} + \bar{X}$. The new weight vector $\bar{W}'$ is "pulled" towards $\bar{X}$, reducing the angle.
*   **Linear Separability:**
    *   If a dataset is **linearly separable**, positive and negative instances can be separated by a hyperplane.
    *   Perceptron is guaranteed to find a separating hyperplane if one exists (for linearly separable data).
    *   The separating hyperplane might not be unique.
    *   If non-linearly separable, no such hyperplane exists.
*   **Further Remarks:**
    *   Final weight vector influenced by final training instances.
    *   **Averaged Perceptron:** Average all weight vectors during training for potentially better generalization.

#### 13.13. Perceptron as Loss Function Minimisation
*   Training data $\mathcal{D} = \{(\bar{X_1}, y_1), \dots, (\bar{X_n}, y_n)\}$.
*   Model defined by parameters $\bar{W} = (w_0, w_1, \dots, w_d)$.
*   Goal: Minimize a loss function $L(\bar{W}, \mathcal{D})$ by changing $\bar{W}$.

    **Loss Function 1: Step Function (Number of Misclassifications)**
    *   Let $a_k = b + \sum_{i=1}^d w_i x_k^{(i)}$.
    *   Loss for single object $(\bar{X_k}, y_k)$: $L(b, \bar{W}, \bar{X_k}, y_k) = 1$ if $\bar{X_k}$ misclassified, $0$ otherwise.
    *   Total loss: $L(b, \bar{W}, \mathcal{D}) = \sum_{k=1}^n L(b, \bar{W}, \bar{X_k}, y_k) = \text{no. of misclassifications}$.
    *   This function is piecewise-constant with many discontinuities. Derivative (where it exists) is 0. Gradient descent is not applicable.

    **Loss Function 2: Hinge-like Loss $h(t) = \max(0,t)$ (Perceptron Criterion)**
    *   Let $a_k = b + \sum_{i=1}^d w_i x_k^{(i)}$.
    *   Define $h(t) = \max(0,t)$.
    *   Loss for single object $(\bar{X_k}, y_k)$: $L(b, \bar{W}, \bar{X_k}, y_k) = h(-y_k \cdot a_k)$.
        *   If correctly classified ($y_k \cdot a_k > 0$), then $-y_k \cdot a_k < 0$, so $h(-y_k \cdot a_k) = 0$.
        *   If misclassified ($y_k \cdot a_k \le 0$), then $-y_k \cdot a_k \ge 0$, so $L(b, \bar{W}, \bar{X_k}, y_k) = -y_k \cdot a_k \ge 0$.
    *   Total loss: $L(b, \bar{W}, \mathcal{D}) = \sum_{k=1}^n h(-y_k \cdot a_k)$.
    *   We want to find parameters $b, \bar{W}$ that minimize this total loss.

    **Gradient Descent for Hinge-like Loss:**
    *   Derivative of $h(t)$: $h'(t) = 0$ if $t < 0$; $h'(t) = 1$ if $t \ge 0$. (Extend $h'(0)=1$).
    *   Using chain rule for $\frac{\partial h(-y_k \cdot a_k)}{\partial b}$:
        *   $\frac{\partial h(-y_k \cdot a_k)}{\partial b} = h'(-y_k \cdot a_k) \cdot \frac{\partial}{\partial b}(-y_k \cdot a_k) = h'(-y_k \cdot a_k) \cdot (-y_k \cdot \frac{\partial a_k}{\partial b}) = h'(-y_k \cdot a_k) \cdot (-y_k \cdot 1)$.
        *   This is $-y_k$ if $\bar{X_k}$ is misclassified (since $-y_k a_k \ge 0 \implies h'(-y_k a_k)=1$), and $0$ otherwise.
    *   Similarly for $\frac{\partial h(-y_k \cdot a_k)}{\partial w_j}$:
        *   $\frac{\partial h(-y_k \cdot a_k)}{\partial w_j} = -y_k \cdot x_k^{(j)}$ if $\bar{X_k}$ is misclassified, and $0$ otherwise.
    *   Gradient for a single misclassified instance $(\bar{X_k}, y_k)$: $\nabla_{b, \bar{W}} h(-y_k \cdot a_k) = (-y_k, -y_k x_k^{(1)}, \dots, -y_k x_k^{(d)})^T = -y_k (1, \bar{X_k})^T$.
    *   **Batch Gradient Descent Update:**
        *   $(b, \bar{W})^T \leftarrow (b, \bar{W})^T - \mu \sum_{k: \bar{X_k} \text{ misc.}} \nabla_{b, \bar{W}} h(-y_k \cdot a_k)$
        *   $(b, \bar{W})^T \leftarrow (b, \bar{W})^T + \mu \sum_{k: \bar{X_k} \text{ misc.}} y_k (1, \bar{X_k})^T$.
        *   This uses the whole training dataset for one update; slow for large datasets.
    *   **Online (Stochastic) Gradient Descent (SGD):** Update after each misclassification.
        *   For a misclassified object $(\bar{X_k}, y_k)$:
            *   $(b, \bar{W})^T \leftarrow (b, \bar{W})^T + \mu \cdot y_k (1, \bar{X_k})^T$.
            *   $b \leftarrow b + \mu \cdot y_k$
            *   $w_j \leftarrow w_j + \mu \cdot y_k \cdot x_k^{(j)}$
        *   If learning rate $\mu=1$, this is exactly the Perceptron update rule.

### 14. Classifier Evaluation

*   How good is a classifier/model/system?
    *   **Absolute goodness:** Does it do what we expect in the "wild"? Hard to know beforehand.
    *   **Relative goodness:** Use a small representative sample of **test data** (gold standard). Compare classifier output to true labels.
*   **Gold Standard (Test Data):**
    *   Dataset used for evaluation.
    *   Each test instance has its correct label annotated.
    *   **Never train on test data!**

#### 14.1. Confusion Matrix (Error Matrix)
For binary classification (Positive/Negative classes):

|                  | Actual YES (+)       | Actual NO (-)        |
| :--------------- | :------------------- | :------------------- |
| Predicted YES(+) | True Positives (TP)  | False Positives (FP) |
| Predicted NO(-)  | False Negatives (FN) | True Negatives (TN)  |

*   **True Positive (TP):** Predicted positive, actually positive.
*   **True Negative (TN):** Predicted negative, actually negative.
*   **False Positive (FP) (Type I error):** Predicted positive, actually negative. (e.g., predict cancer, patient is healthy).
*   **False Negative (FN) (Type II error):** Predicted negative, actually positive. (e.g., predict healthy, patient has cancer).
*   FP and FN can have very different importance depending on the task.

#### 14.2. Evaluation Measures
*   **Accuracy:** Proportion of all objects correctly classified.
    *   $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
*   **Precision (Positive Predictive Value):** Proportion of predicted Positives that are truly Positive.
    *   $Precision = \frac{TP}{TP + FP}$
    *   High precision means few false positives. Important when cost of FP is high (e.g., product recommendation - don't show irrelevant items).
*   **Recall (Sensitivity, True Positive Rate):** Proportion of actual Positives that are correctly classified.
    *   $Recall = \frac{TP}{TP + FN}$
    *   High recall means few false negatives. Important when cost of FN is high (e.g., cancer detection - don't miss actual cases).
*   **Precision-Recall Trade-off:**
    *   Improving precision often results in lowering recall, and vice versa.
    *   Can be controlled by varying a classifier's threshold.
*   **F-score (F1-score):** Harmonic mean of Precision and Recall.
    *   $F_1 = \frac{2 \times Precision \times Recall}{Precision + Recall} = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$
    *   Gives a larger weight to lower numbers; good balance if both P and R are important.

#### 14.3. Evaluation for Multiple Classes
*   **Precision for a class A:** $\frac{\text{no. objects correctly classified as A}}{\text{no. objects classified as A}}$
*   **Recall for a class A:** $\frac{\text{no. objects correctly classified as A}}{\text{no. objects that belong to class A}}$
*   **F-score for class A ($F_A$):** Calculated from $Precision_A$ and $Recall_A$.
*   **Macro F-score:** Average F-score across all classes.
    *   $Macro F_1 = \frac{1}{C} \sum_{i=1}^C F_i$ (where $C$ is number of classes, $F_i$ is F-score for class $i$).

### 15. Multiclass Classification from Binary Classifiers

*   Some classifiers are inherently multiclass (k-NN, Naive Bayes).
*   Others are binary (Perceptron, Logistic Regression).
*   Goal: Use a binary classification algorithm $\mathcal{A}$ to make $k$-class predictions.

#### 15.1. One-vs.-One (OvO or All-vs-All) Approach
1.  For each pair of classes $(i, j)$, train a binary classifier $\mathcal{A}_{i,j}$ to distinguish between class $i$ and class $j$, using only data from these two classes.
2.  This results in $\frac{k(k-1)}{2}$ prediction models.
3.  For a new object $\bar{X}$:
    *   Each classifier $\mathcal{A}_{i,j}$ votes for either class $i$ or class $j$.
    *   The class label with the most votes is declared the winner.
*   **Drawback:** Ambiguity if classes get the same number of votes (can use confidence scores from $\mathcal{A}$ to break ties if available).

#### 15.2. One-vs.-Rest (OvR or One-vs-All) Approach
Requires the binary classifier $\mathcal{A}$ to output a numeric "confidence" score.
1.  For each class $i$, train a binary classifier $\mathcal{A}_i$ with objects of class $i$ as positive samples and all other objects (from all other $k-1$ classes) as negative samples.
2.  This results in $k$ prediction models.
3.  For a new object $\bar{X}$:
    *   Apply all $k$ models $\mathcal{A}_1, \dots, \mathcal{A}_k$.
    *   Output the class label $y$ corresponding to the model $\mathcal{A}_y$ with the highest confidence score: $y = \text{argmax}_{i \in \{1, \dots, k\}} \mathcal{A}_i(\bar{X})$.
*   **Confidence score choice:**
    *   Perceptron: Activation score $a = b + \bar{W}^T\bar{X}$.
    *   Logistic Regression: Probability $\sigma(a)$, where $a = b + \bar{W}^T\bar{X}$.
*   **Drawbacks:**
    *   Scale of confidence scores may differ between binary classifiers.
    *   Binary classifiers are trained on unbalanced datasets (negative set is much larger).

### 16. Regularization

*   Process of reducing overfitting by constraining a model (reducing complexity/number of parameters).
*   For classifiers using a weight vector $\bar{W}$, regularization often involves minimizing the norm (length) of $\bar{W}$.
*   **Popular methods:**
    *   **L2 regularization (Ridge regression, Tikhonov regularization):** Adds $\lambda ||\bar{W}||_2^2$ to the loss.
    *   **L1 regularization (Lasso regression):** Adds $\lambda ||\bar{W}||_1$ to the loss (can lead to sparse weights).
    *   L1+L2 regularization (Elastic Net).

#### 16.1. Example: Polynomial Approximation & Overfitting
*   Dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ sampled from $f(x) = x^3 - 4x^2 + 3x - 2$ with noise.
*   Approximate $f(x)$ with a polynomial of degree $d$: $\hat{y}(x, \bar{W}) = w_0 + \sum_{j=1}^d w_j x^j = (1, x, x^2, \dots, x^d) \cdot \bar{W}$.
*   Loss function (Residual Sum of Squares - RSS): $L(\mathcal{D}, \bar{W}) = \sum_{i=1}^n (\hat{y}(x_i, \bar{W}) - y_i)^2$.
*   As degree $d$ increases, parameters (weights $w_j$) can grow very large, leading to overfitting.
    *   $d=0$: Underfit (horizontal line).
    *   $d=1,2$: Better fit.
    *   $d=9$: Overfit (wild oscillations, very large parameter values).

#### 16.2. L2 Regularization
*   Modified loss function: $J(\mathcal{D}, \bar{W}) = L(\mathcal{D}, \bar{W}) + \lambda ||\bar{W}||_2^2 = \sum_{i=1}^n (\hat{y}(x_i, \bar{W}) - y_i)^2 + \lambda \sum_{j=0}^d w_j^2$.
    *   (Note: Often $w_0$ is not regularized, so sum for $\lambda$ term would be $j=1$ to $d$).
*   $\lambda$ is the **regularization coefficient** (hyperparameter).
    *   Increasing $\lambda$ restricts parameter growth, reduces overfitting, but can cause underfitting if too large.
*   **Gradient of regularized objective:** $\nabla_{\bar{W}} J(\mathcal{D}, \bar{W}) = \nabla_{\bar{W}} L(\mathcal{D}, \bar{W}) + 2\lambda\bar{W}$.
*   **SGD Update Rule with L2 Regularization:**
    *   Original SGD update for loss $L$: $\bar{W} \leftarrow \bar{W} - \mu \nabla_{\bar{W}} L$.
    *   L2 regularized SGD update: $\bar{W} \leftarrow \bar{W} - \mu (\nabla_{\bar{W}} L + 2\lambda\bar{W}) = \bar{W} - \mu \nabla_{\bar{W}} L - 2\mu\lambda\bar{W} = (1-2\mu\lambda)\bar{W} - \mu \nabla_{\bar{W}} L$.
    *   This is a form of "weight decay".
*   **L2 Regularized Perceptron Update Rule (assuming $\mu=1$ for Perceptron):**
    *   Original: $\bar{W} \leftarrow \bar{W} + y_i \bar{X_i}$ (for misclassified $(\bar{X_i}, y_i)$).
    *   Loss gradient for Perceptron criterion (misclassified): $-y_i \bar{X_i}$.
    *   Regularized: $\bar{W} \leftarrow \bar{W} - 1 \cdot (-y_i \bar{X_i} + 2\lambda\bar{W}) = \bar{W} + y_i \bar{X_i} - 2\lambda\bar{W} = (1-2\lambda)\bar{W} + y_i \bar{X_i}$.
    *   Bias term $b$ is typically not regularized. So, $b \leftarrow b+y_i$.
    *   For correctly classified instances, only weight decay applies: $\bar{W} \leftarrow (1-2\lambda)\bar{W}$, $b \leftarrow b$. (If updates are only on error, this part is implicit).

#### 16.3. How to set $\lambda$
1.  Split training dataset into **training** and **validation** parts (e.g., 80%-20%).
2.  Try different values for $\lambda$ (typically on a logarithmic scale, e.g., $10^{-5}, 10^{-4}, \dots, 10^5$).
3.  Train a different classification model for each $\lambda$ on the training part.
4.  Select the $\lambda$ that gives the best performance (e.g., accuracy, RSS) on the **validation part**.
5.  Finally, evaluate the model with the chosen $\lambda$ on the unseen **test set**.

## Part 4: More Classification Algorithms

(Covered Perceptron already. Now k-NN, Naive Bayes, Logistic Regression)

### 17. K-Nearest Neighbours (k-NN)

*   A non-parametric, instance-based (memory-based) learning algorithm.

#### 17.1. Simplest Classifier (Lookup)
*   Given training dataset $D_{train}$ of $N$ instances $(\bar{X}, y)$.
*   "Remember" all $N$ instances.
*   To classify a test instance $\bar{X}'$:
    *   If $\bar{X}' \in D_{train}$, return its label.
    *   Otherwise, make a random guess (not very useful).

#### 17.2. Nearest Neighbour (1-NN) Classifier
*   **Training:** Store entire training set $D_{train}$.
*   **Classification:** For an input object $\bar{X}'$:
    1.  Find the object $\bar{X} \in D_{train}$ that is "closest" (most similar) to $\bar{X}'$.
    2.  Classify $\bar{X}'$ with the same label as $\bar{X}$.

#### 17.3. k-Nearest Neighbour (k-NN) Classifier
*   **Training:** Store entire training set.
*   **Classification:** For an input object $\bar{X}'$:
    1.  Find the $k$ closest (nearest) objects in $D_{train}$ to $\bar{X}'$.
    2.  Find the majority label among these $k$ nearest neighbours.
    3.  This majority label is predicted for $\bar{X}'$.
    *   Example: 5-NN. If 5 closest points to a new black point are 3 blue and 2 red, predict blue.

#### 17.4. Measures of Similarity/Distance
*   **Similarity functions:** Larger values imply greater similarity.
*   **Distance functions:** Smaller values imply greater similarity (distance is inverse of similarity).
*   Choice depends on domain (distance for spatial, similarity for text).
*   Can be closed-form (formulas) or algorithmic (computationally expensive).

    **For Numerical Data:**
    *   **Euclidean Distance ($L_2$ distance):**
        *   $EucDist(\bar{X}, \bar{Y}) = ||\bar{X} - \bar{Y}||_2 = \sqrt{\sum_{i=1}^d (x_i - y_i)^2} = \sqrt{(\bar{X} - \bar{Y})^T (\bar{X} - \bar{Y})}$.
        *   $||\bar{T}||_2 = \sqrt{\sum t_i^2}$ is the $L_2$-norm.
    *   **Manhattan Distance ($L_1$ distance):**
        *   $ManDist(\bar{X}, \bar{Y}) = ||\bar{X} - \bar{Y}||_1 = \sum_{i=1}^d |x_i - y_i|$.
        *   $||\bar{T}||_1 = \sum |t_i|$ is the $L_1$-norm.
        *   Represents distance if movement is restricted to a grid (like city blocks).
    *   **Vector Norms:** Express "size/length" of a vector.
        *   $L_1$-norm: $||\bar{X}||_1 = \sum |x_i|$.
        *   $L_2$-norm: $||\bar{X}||_2 = \sqrt{\sum x_i^2}$.
        *   $L_0$-norm: $||\bar{X}||_0 = \text{no. of non-zero elements in } \bar{X}$.
        *   $L_\infty$-norm: $||\bar{X}||_\infty = \max\{|x_1|, \dots, |x_d|\}$.
        *   Distance from norm: $Dist_p(\bar{X}, \bar{Y}) = ||\bar{X} - \bar{Y}||_p$.
    *   **Cosine Similarity:** Measures the cosine of the angle between two vectors.
        *   $CosSim(\bar{X}, \bar{Y}) = \frac{\bar{X}^T \bar{Y}}{||\bar{X}||_2 ||\bar{Y}||_2} = \cos(\theta)$.
        *   Ranges from -1 (opposite) to 1 (same direction). 0 means orthogonal.
        *   Useful for text mining (documents as term frequency vectors).
    *   **Cosine Distance:** $CosDist(\bar{X}, \bar{Y}) = 1 - CosSim(\bar{X}, \bar{Y})$. Ranges from 0 to 2.

    **For Set Data:** (A, B are sets)
    *   **Jaccard Similarity Coefficient:** $J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$.
    *   **Jaccard Distance:** $d_J(A,B) = 1 - J(A,B)$.
    *   **Overlap Coefficient:** $overlap(A,B) = \frac{|A \cap B|}{\min(|A|, |B|)}$.

    **For Binary Data:** (Vectors $\bar{X}, \bar{Y}$ of same length)
    *   **Hamming Distance:** Number of coordinates where $\bar{X}$ and $\bar{Y}$ differ.
        *   $Hamming(\bar{X}, \bar{Y}) = |\{i : x_i \ne y_i\}|$.
        *   Example: $\bar{X}=(1,0,1,0,1,1)^T, \bar{Y}=(0,0,1,1,1,1)^T \implies Hamming(\bar{X}, \bar{Y})=2$.

    **For Categorical Data:**
    *   No natural ordering or 'difference' operation.
    *   General approach: $Sim(\bar{X}, \bar{Y}) = \sum_{i=1}^d S(x_i, y_i)$, where $S(x_i, y_i)$ is similarity of feature values.
    *   **Example 1 (Simple Match):** $S(x_i, y_i) = 1$ if $x_i = y_i$, $0$ otherwise.
        *   Drawback: Doesn't account for relative frequencies of feature values. (e.g., matching on a common value is less informative than matching on a rare value).
    *   **Example 2 (Frequency-weighted):** Let $p_k(x)$ be fraction of objects where $k$-th feature is $x$.
        *   $S(x_i, y_i) = 1/(p_i(x_i))^2$ if $x_i=y_i$, $0$ otherwise. (This gives higher similarity for matches on rarer values).

#### 17.5. Choosing the Parameter $k$
*   $k$ is a hyperparameter.
*   Value depends on dataset size (larger datasets might need higher $k$). High $k$ for small datasets can cross class boundaries.
*   **How to choose $k$:**
    *   **Bad option:** Try different $k$ when evaluating on **test data** (leads to overfitting the test set).
    *   **Good option:** Use a **validation dataset** to find a good value of $k$.
*   **Train/Validation/Test Approach:**
    1.  Split available data into Training Set and Test Set.
    2.  Further split Training Set into a smaller Training part and a Validation part.
    3.  Train k-NN models with different $k$ values on the Training part.
    4.  Evaluate each model on the Validation part. Choose the $k$ that performs best.
    5.  Finally, evaluate the chosen model (with best $k$) on the Test Set for an unbiased performance estimate.
*   **Cross-Validation (if validation set is too small):**
    1.  Divide the Training Set into $M$ folds (e.g., $M=5$ or $M=10$).
    2.  For each $k$ to test:
        *   For each fold $m = 1, \dots, M$:
            *   Train k-NN on $M-1$ folds.
            *   Validate on fold $m$.
        *   Average performance across $M$ folds for this $k$.
    3.  Choose $k$ with best average performance.
    4.  Train final model with best $k$ on the entire Training Set.
    5.  Evaluate on Test Set.

#### 17.6. Complexity of k-NN
*   **Training:**
    *   Time: Just store data points (constant time per object if $n$ objects, $d$ features). Total $O(nd)$ to store.
    *   Space: $O(nd)$ for the training data.
    *   Computationally cheap.
*   **Classification (for one test example):**
    *   Find $k$ closest neighbours:
        *   Calculate distance to all $n$ training examples: $O(nd)$ (assuming $O(d)$ per distance calc).
        *   Find $k$ smallest distances: $O(n \log k)$ or $O(n)$ with selection algorithm.
    *   Computationally expensive, especially for large $n$.
*   **Practical Implication:** Classification time is often more critical than training time.
*   **Possible Solution:** Approximate Nearest Neighbor (ANN) algorithms (e.g., FLANN) can speed up lookup at the cost of some accuracy.

#### 17.7. Inductive Bias and Feature Importance
*   **k-NN classifier assumes:**
    *   Nearby points should have the same label (locality principle).
    *   **All features are equally important!** This is a major limitation. If many irrelevant features exist, k-NN performs poorly. Feature scaling (normalization) is crucial. Feature selection/weighting can help.

#### 17.8. Summary on k-NN Classifier
1.  **Preprocess data:** Normalize features (zero mean, unit variance).
2.  If data is very high-dimensional, consider dimensionality reduction.
3.  Split training data into training (50-90%) and validation (10-50%), or use cross-validation.
4.  Train and evaluate k-NN on validation data for many choices of $k$ and different distance types (start with $L_1, L_2$).

### 18. Probabilistic Classifiers

*   **"Ordinary" classifier:** A function $f$ that assigns an input object $\bar{X}$ to a predicted class $c$ from $\{c_1, \dots, c_k\}$. $c = f(\bar{X})$.
*   **Probabilistic classifier:** Models a conditional probability distribution $P(C|\bar{X})$. For an input $\bar{X}$, it gives probabilities $p_1, \dots, p_k$ where $p_i = P(C=c_i|\bar{X})$ and $\sum p_i = 1$.

#### 18.1. Two Types of Probabilistic Models
1.  **Discriminative Models:**
    *   Assume the conditional distribution $P(C|\bar{X})$ has a specific form $P_\theta(C|\bar{X})$ depending on parameters $\theta = (\theta_1, \dots, \theta_k)$.
    *   Use training data to find/learn $\theta$ such that $P_\theta(C|\bar{X})$ is "best possible".
    *   Example: Logistic Regression.
2.  **Generative Models:**
    *   Assume data comes from a specific joint distribution $P_\theta(\bar{X}, C)$ depending on parameters $\theta$.
    *   Use training data to find/learn $\theta$ such that $P_\theta(\bar{X}, C)$ is "best possible".
    *   Use $P_\theta(\bar{X}, C)$ to classify new objects (often via Bayes' rule to get $P(C|\bar{X})$).
    *   Example: Naive Bayes.

#### 18.2. Data Generating Distribution & Bayes Optimal Classifier
*   **Assumption:** Data $(\bar{X}, c)$ comes from an unknown probability distribution $P$ over object-class pairs.
*   If we knew $P(\bar{X}, c)$, we could use the **Bayes Optimal Classifier**:
    *   $f^*(\bar{X}) = \text{arg max}_{c \in \mathcal{C}} P(\bar{X}, c)$.
    *   This classifier is "best possible": its probability of error is less than or equal to any other classifier.
*   **In practice:** We don't know $P$. We learn an estimate $\hat{P}$ from training data.
    *   **Model assumption:** Assume $P$ belongs to a family of parametric distributions (e.g., Normal).
    *   **Estimate parameters:** Find parameters for the specific distribution in the family that is most likely to generate the training data.
    *   **I.I.D. assumption:** Training data instances $(\bar{X}, c)$ are drawn independently and identically distributed from $P$.

#### 18.3. Maximum Likelihood Estimation (MLE)
*   A common method to estimate parameters.
*   **Principle:** Choose parameters that **maximize the probability (likelihood)** of observing the given training data.

    **MLE Example 1: Single Parameter (Biased Coin)**
    *   Observe data $THHH$ ($H$=heads, $T$=tails).
    *   Assume Bernoulli distribution with parameter $\beta = P(H)$. $P(T) = 1-\beta$.
    *   I.I.D. assumption: flips are independent.
    *   Likelihood of $THHH$: $P_\beta(THHH) = P_\beta(T)P_\beta(H)P_\beta(H)P_\beta(H) = (1-\beta)\beta^3 = \beta^3 - \beta^4$.
    *   To find $\beta$ that maximizes this, take derivative w.r.t. $\beta$, set to 0:
        *   $\frac{d}{d\beta}(\beta^3 - \beta^4) = 3\beta^2 - 4\beta^3$.
        *   $3\beta^2 - 4\beta^3 = 0 \implies \beta^2(3-4\beta)=0$. Since $\beta \ne 0$ (for non-trivial case), $3-4\beta=0 \implies \beta = 3/4$.
    *   MLE estimate $\hat{\beta} = 3/4$.

    **MLE Example 2: General Biased Coin ($h$ heads, $t$ tails)**
    *   Likelihood: $L(\beta) = \beta^h (1-\beta)^t$.
    *   Often easier to maximize **log-likelihood**: $\ell(\beta) = \log L(\beta) = h \log\beta + t \log(1-\beta)$.
    *   $\frac{d\ell}{d\beta} = \frac{h}{\beta} - \frac{t}{1-\beta} = 0 \implies \frac{h}{\beta} = \frac{t}{1-\beta} \implies h(1-\beta) = t\beta \implies h - h\beta = t\beta \implies h = (h+t)\beta \implies \hat{\beta} = \frac{h}{h+t}$.

    **MLE Example 3: Multiple Parameters (K-sided Die)**
    *   Parameters $\beta_1, \dots, \beta_K$ where $\beta_i = P(\text{side } i)$.
    *   Constraints: $\beta_i \ge 0$ for all $i$, and $\sum_{i=1}^K \beta_i = 1$. (Generalized Bernoulli).
    *   Observed data: $x_1$ rolls of 1, $x_2$ rolls of 2, ..., $x_K$ rolls of K.
    *   Likelihood: $L(\bar{\beta}) = \prod_{i=1}^K \beta_i^{x_i}$.
    *   Log-likelihood: $\ell(\bar{\beta}) = \sum_{i=1}^K x_i \log\beta_i$.
    *   Maximize $\ell(\bar{\beta})$ subject to $\sum \beta_i - 1 = 0$. Use Lagrange Multipliers:
        *   $\mathcal{L}(\bar{\beta}, \lambda) = \sum x_i \log\beta_i - \lambda(\sum \beta_i - 1)$.
        *   $\frac{\partial\mathcal{L}}{\partial\beta_i} = \frac{x_i}{\beta_i} - \lambda = 0 \implies \beta_i = \frac{x_i}{\lambda}$.
        *   $\frac{\partial\mathcal{L}}{\partial\lambda} = -(\sum \beta_i - 1) = 0 \implies \sum \beta_i = 1$.
        *   Substitute $\beta_i$: $\sum \frac{x_i}{\lambda} = 1 \implies \frac{1}{\lambda} \sum x_i = 1 \implies \lambda = \sum x_i$.
        *   MLE estimate: $\hat{\beta_i} = \frac{x_i}{\sum_{j=1}^K x_j} = \frac{x_i}{N}$ (where $N$ is total number of rolls).

### 19. Naive Bayes Classifier

*   A generative probabilistic classifier based on Bayes' theorem with a "naive" independence assumption.

#### 19.1. Bayes' Rule (Theorem)
*   $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$, where $P(E) \ne 0$.
    *   $P(H|E)$: **Posterior** probability of hypothesis $H$ given evidence $E$.
    *   $P(E|H)$: **Likelihood** of evidence $E$ given hypothesis $H$.
    *   $P(H)$: **Prior** probability of hypothesis $H$.
    *   $P(E)$: **Marginal** probability of evidence $E$.
*   Useful when $P(H|E)$ is hard to estimate directly, but $P(E|H)$, $P(H)$, $P(E)$ are easier.
*   Derivation: From $P(H,E) = P(H|E)P(E)$ and $P(H,E) = P(E|H)P(H)$.

#### 19.2. Naive Bayes Approximation for Classification
*   Goal: Given a $d$-dimensional test object $\bar{X}=(a_1, \dots, a_d)$, estimate $P(C=c|\bar{X}=(a_1, \dots, a_d))$ for each class $c$.
*   Using Bayes' Rule:
    $P(C=c|\bar{X}=\bar{a}) = \frac{P(\bar{X}=\bar{a}|C=c)P(C=c)}{P(\bar{X}=\bar{a})}$.
*   $P(\bar{X}=\bar{a})$ is a normalizing constant (doesn't depend on class $c$), so we can ignore it for finding the most probable class:
    $P(C=c|\bar{X}=\bar{a}) \propto P(\bar{X}=\bar{a}|C=c)P(C=c)$.
*   **Estimating terms:**
    *   $P(C=c)$ (Prior): Fraction of training objects belonging to class $c$.
    *   $P(\bar{X}=\bar{a}|C=c)$ (Class-conditional likelihood): Hard to estimate directly for high-dimensional $\bar{X}$ due to data sparsity (curse of dimensionality).

*   **Naive Assumption:** Features $x_1, \dots, x_d$ are **conditionally independent** given the class $C=c$.
    *   $P(\bar{X}=\bar{a}|C=c) = P(x_1=a_1, \dots, x_d=a_d|C=c) = \prod_{j=1}^d P(x_j=a_j|C=c)$.
*   **Estimating $P(x_j=a_j|C=c)$:** Fraction of training objects in class $c$ that have feature $x_j=a_j$. This is much easier.
*   **Final Proportional Form:**
    $P(C=c|\bar{X}) \propto P(C=c) \prod_{j=1}^d P(x_j=a_j|C=c)$.
    Posterior $\propto$ Prior $\times$ Product of individual Likelihoods.
*   **Classification:** Choose class $c$ that maximizes this product.

#### 19.3. Example: Predicting Play Tennis
(Based on slide 247-249 data)
*   Given features: Outlook, Temperature, Humidity, Windy. Class: Play (Yes/No).
*   Test instance $\bar{X} = (\text{Outlook=sunny, Temp=cool, Humidity=high, Windy=true})$.
*   Calculate $P(\text{Play=yes}|\bar{X}) \propto P(\text{Play=yes}) \times P(\text{Outlook=sunny}|\text{Play=yes}) \times P(\text{Temp=cool}|\text{Play=yes}) \times \dots$
*   Calculate $P(\text{Play=no}|\bar{X}) \propto P(\text{Play=no}) \times P(\text{Outlook=sunny}|\text{Play=no}) \times P(\text{Temp=cool}|\text{Play=no}) \times \dots$
*   Compare the two values and pick the class with the higher proportional posterior.
    *   $P(\text{Play=yes}) = 9/14$, $P(\text{Play=no}) = 5/14$.
    *   $P(\text{Outlook=sunny}|\text{Play=yes}) = 2/9$, $P(\text{Temp=cool}|\text{Play=yes}) = 3/9$, etc.
    *   $P(\text{Outlook=sunny}|\text{Play=no}) = 3/5$, $P(\text{Temp=cool}|\text{Play=no}) = 1/5$, etc.
    *   Slide 249 result: $P(\text{Play=yes}|\bar{X}) \propto 0.00529$, $P(\text{Play=no}|\bar{X}) \propto 0.020$. Predict Play=no.

#### 19.4. Zero Probabilities and Laplace Smoothing
*   **Issue:** If a feature value $a_j$ never co-occurs with a class $c$ in training data, then $P(x_j=a_j|C=c) = 0$.
*   This makes the entire product $\prod P(x_j=a_j|C=c)$ zero, even if other features strongly suggest class $c$.
*   Example: If $P(\text{Outlook=overcast}|\text{Play=no}) = 0/5 = 0$, then $P(\text{Play=no}|\bar{X}_{\text{overcast}}) \propto 0$.
*   **Laplace Smoothing (Add-one smoothing):**
    *   "Borrow" probability mass from high-probability features and distribute it.
    *   Modified probability estimate: $P(x_j=a|C=c) = \frac{n(a,c) + 1}{N(c) + m_j}$.
        *   $n(a,c)$: Number of training objects in class $c$ where feature $j$ has value $a$.
        *   $N(c)$: Total number of training objects in class $c$.
        *   $m_j$: Number of possible values for feature $j$. (If feature $j$ is binary, $m_j=2$).
    *   This ensures no probability is zero.
    *   Example (slide 257-258): If $N(c)=9$, $m_j=5$ (5 possible values for feature $x_j$).
        *   Before smoothing: $P(x_j=a_3|C=c) = 0/9$.
        *   After smoothing: $P(x_j=a_3|C=c) = (0+1)/(9+5) = 1/14$.
        *   Other counts also get adjusted, e.g., $P(x_j=a_1|C=c) = (count(a_1,c)+1)/(9+5)$.

### 20. Logistic Regression

*   A discriminative probabilistic classifier for binary classification.

#### 20.1. Main Idea
*   Define a separating hyperplane $H$ parameterized by weights $\bar{W}=(w_1, \dots, w_d)$ and bias $b$: $H = \{\bar{X} : b + \bar{W}^T\bar{X} = 0\}$.
*   Perceptron uses only the *sign* of $a = b + \bar{W}^T\bar{X}$.
*   Logistic Regression uses:
    *   **Sign** of $a$ to classify.
    *   **Magnitude** $|a|$ to quantify confidence (larger $|a|$ means further from hyperplane, more confident).
*   To interpret confidence score $a \in [-\infty, \infty]$ as probability in $[0,1]$: use **logistic sigmoid function**.

#### 20.2. Logistic Sigmoid Function
*   $\sigma(x) = \frac{1}{1 + e^{-x}}$.
*   **Properties:**
    *   $\sigma(x) \in [0,1]$ for any $x \in [-\infty, \infty]$.
    *   $\sigma(0) = 0.5$. $\sigma(x) \to 1$ as $x \to \infty$. $\sigma(x) \to 0$ as $x \to -\infty$.
    *   $1 - \sigma(x) = \sigma(-x)$. (Useful property: $1 - \frac{1}{1+e^{-x}} = \frac{1+e^{-x}-1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{e^x+1} = \frac{1}{1+e^{-(-x)}} = \sigma(-x)$).
    *   $\frac{d\sigma}{dx} = \sigma(x)(1-\sigma(x))$.

#### 20.3. Model Assumption (Discriminative)
*   For an object $\bar{X}$, probability of belonging to positive class ($y=+1$):
    *   $P(y=+1|\bar{X}) = \sigma(a) = \frac{1}{1+e^{-a}}$, where $a = b + \bar{W}^T\bar{X}$.
*   Probability of belonging to negative class ($y=-1$):
    *   $P(y=-1|\bar{X}) = 1 - \sigma(a) = \sigma(-a) = \frac{1}{1+e^{a}}$.
*   Conveniently written as: $P(y=t|\bar{X}) = \sigma(t \cdot a) = \frac{1}{1+e^{-t \cdot a}}$ for $t \in \{-1, +1\}$.

#### 20.4. Choosing/Fitting Parameters (MLE)
*   Training data $\mathcal{D} = \{(\bar{X_1}, y_1), \dots, (\bar{X_n}, y_n)\}$.
*   Find parameters $b, w_1, \dots, w_d$ that maximize the likelihood function (assuming I.I.D.):
    *   $L(b, \bar{W}, \mathcal{D}) = \prod_{i=1}^n P(y_i|\bar{X_i}) = \prod_{i=1}^n \sigma(y_i(b + \bar{W}^T\bar{X_i}))$.
*   Equivalent to minimizing the **negative log-likelihood (NLL)**:
    *   $NLL = -\ell\ell = -\log L = -\sum_{i=1}^n \log \sigma(y_i(b + \bar{W}^T\bar{X_i})) = -\sum_{i=1}^n \log \sigma(y_i a_i)$.
    *   (where $a_i = b + \bar{W}^T\bar{X_i}$).
*   **Gradient of Log-Likelihood $\ell\ell$:**
    *   $\frac{\partial \ell\ell}{\partial b} = \sum_{i=1}^n y_i \cdot \sigma(-y_i a_i)$.
        *   Interpretation: $\sum_{\bar{X_i} \in \mathcal{D}_+} P(y=-1|\bar{X_i}) - \sum_{\bar{X_i} \in \mathcal{D}_-} P(y=+1|\bar{X_i})$. (Sum of probabilities of misclassification for positive instances minus sum for negative instances, scaled by $y_i$).
    *   $\frac{\partial \ell\ell}{\partial w_k} = \sum_{i=1}^n y_i \cdot \sigma(-y_i a_i) \cdot x_k^{(i)}$. (where $x_k^{(i)}$ is the $k$-th feature of $\bar{X_i}$).

#### 20.5. Update Rule (Gradient Ascent for Likelihood / Descent for NLL)
*   **Batch Update:**
    *   $b \leftarrow b + \mu \sum_{i=1}^n y_i \sigma(-y_i a_i)$
    *   $\bar{W} \leftarrow \bar{W} + \mu \sum_{i=1}^n y_i \sigma(-y_i a_i) \bar{X_i}$
    *   Uses whole training set. Popular optimizer: L-BFGS. Can be slow but accurate.
*   **Online (Stochastic Gradient Descent - SGD) Update:** For each training instance $(\bar{X_i}, y_i)$:
    *   $a_i = b + \bar{W}^T\bar{X_i}$
    *   $b \leftarrow b + \mu \cdot y_i \cdot \sigma(-y_i a_i)$
    *   $w_s \leftarrow w_s + \mu \cdot y_i \cdot \sigma(-y_i a_i) \cdot x_s^{(i)}$ for all features $s=1, \dots, d$.
    *   Requires multiple iterations (epochs) over dataset. Frequently used for large-scale tasks.

#### 20.6. Logistic Regression Prediction
`LogisticRegressionTest(b, \bar{W}, \bar{X}_{test})`
1.  $a = b + \bar{W}^T\bar{X}_{test}$.
2.  If $a > 0$:
    *   Predicted label = +1.
    *   Probability (confidence) = $\sigma(a)$.
3.  Else ($a \le 0$):
    *   Predicted label = -1.
    *   Probability (confidence) = $\sigma(-a)$ (which is $1-\sigma(a)$).

#### 20.7. L2 Regularization in Logistic Regression
*   Objective: Minimize $NLL + \lambda ||\bar{W}||_2^2$.
*   SGD Update for $\bar{W}$ (for one instance $(\bar{X}, y)$ with activation $a=b+\bar{W}^T\bar{X}$):
    *   $\bar{W} \leftarrow \bar{W} - \mu \cdot (-\text{gradient of } \ell\ell \text{ for this instance} + 2\lambda\bar{W})$
    *   $\bar{W} \leftarrow \bar{W} - \mu \cdot (-y \sigma(-ya)\bar{X} + 2\lambda\bar{W})$
    *   $\bar{W} \leftarrow \bar{W} + \mu \cdot y \sigma(-ya)\bar{X} - 2\mu\lambda\bar{W}$
    *   $\bar{W} \leftarrow (1 - 2\mu\lambda)\bar{W} + \mu \cdot y \sigma(-ya)\bar{X}$.
    *   Bias $b$ update remains unregularized: $b \leftarrow b + \mu \cdot y \sigma(-ya)$.

## Part 5: Association Pattern Mining

*   Discovering interesting relationships or associations among items in large datasets.

### 21. Applications
*   **Supermarket data:** Which items are frequently bought together (e.g., for target marketing, shelf placement).
*   **Text mining:** Identifying co-occurring terms and keywords.
*   **Dependency-oriented data types:** Web log analysis, software bug detection, spatio-temporal event detection.

### 22. Terminology
*   Borrowed from supermarket analogy.
*   **Transaction:** A single data object (e.g., a customer's shopping basket).
*   **Itemset:** A set of items (e.g., {Milk, Bread}).
*   **Large itemsets (Frequent itemsets/patterns):** Itemsets that appear frequently.

### 23. Usage
*   Frequent itemsets generate **association rules** of the form $X \Rightarrow Y$.
    *   $X, Y$ are itemsets, $X \cap Y = \emptyset$.
    *   Meaning: If a transaction contains $X$, it is "likely" to also contain $Y$.
    *   Example: `{Eggs, Milk} \Rightarrow \{\text{Yogurt}\}`.
        *   Action: Promote yogurt to customers buying eggs and milk. Place yogurt near eggs/milk.

### 24. Frequent Pattern Mining Model
*   **Universe of items $U$:** Set of all possible items, $|U|=d$.
*   **Dataset $\mathcal{D}$:** Consists of $n$ transactions $T_1, \dots, T_n$, each $T_i$ is an itemset (subset of $U$).
*   Each transaction can be represented as a $d$-dimensional binary vector.
*   **Support of an itemset $I$ ($sup(I)$):** Fraction of transactions in $\mathcal{D}$ that contain $I$ as a subset.
    *   $sup(I) = \frac{|\{T_j \in \mathcal{D} | I \subseteq T_j\}|}{|\mathcal{D}|}$.

#### 24.1. Frequent Itemset Mining Problem
*   Given dataset $\mathcal{D}$ and a **frequency threshold $f$** (min_sup).
*   Determine all itemsets $I$ such that $sup(I) \ge f$.
*   Example: If $f=0.65$ (65%).
    *   If {Milk, Butter, Bread} appears in 4 out of 6 transactions, $sup=4/6 \approx 0.667 \ge 0.65 \implies$ frequent.
    *   If {Mushrooms, Onion, Carrot} appears in 1 out of 6, $sup=1/6 \approx 0.167 < 0.65 \implies$ not frequent.

#### 24.2. Monotonicity of Support
*   **Support Monotonicity Property (Anti-monotone):** If an itemset $I$ is frequent, all its subsets $J \subseteq I$ are also frequent. More precisely, $sup(J) \ge sup(I)$ for $J \subseteq I$.
*   **Downward Closure Property:** Every subset of a frequent itemset is also frequent. (Converse: If an itemset is infrequent, all its supersets are also infrequent). This is key for pruning.

#### 24.3. Maximal Frequent Itemsets
*   A frequent itemset $I$ is **maximal** if it is frequent and no superset of $I$ is frequent.
*   Example ($f=0.65$):
    *   {Milk, Butter, Bread} is maximal frequent.
    *   {Butter, Bread} is frequent, but not maximal (because {Milk, Butter, Bread} is also frequent).
*   Maximal frequent itemsets provide a compact representation, but support values of subsets are lost.

### 25. Association Rules
*   Form: $X \Rightarrow Y$, where $X, Y$ are itemsets, $X \cap Y = \emptyset$.
*   **Support of a rule $X \Rightarrow Y$:** $sup(X \Rightarrow Y) = sup(X \cup Y)$.
*   **Confidence of a rule $X \Rightarrow Y$:** $conf(X \Rightarrow Y) = \frac{sup(X \cup Y)}{sup(X)} = P(Y|X)$.
    *   Measures likelihood: conditional probability that a transaction contains $Y$, given it contains $X$.
*   Example: $conf(\{\text{Milk}\} \Rightarrow \{\text{Butter, Bread}\})$
    *   $sup(\{\text{Milk, Butter, Bread}\}) = 4/6 = 2/3$.
    *   $sup(\{\text{Milk}\}) = 5/6$.
    *   $conf = (2/3) / (5/6) = (2/3) \cdot (6/5) = 12/15 = 4/5 = 0.8$.
*   **Definition of an Association Rule:**
    *   Rule $X \Rightarrow Y$ is an association rule at frequency threshold $f$ and confidence threshold $c$ if:
        1.  $sup(X \cup Y) \ge f$.
        2.  $conf(X \Rightarrow Y) \ge c$.

#### 25.1. Association Rule Generation Framework
1.  **Phase 1: Frequent Itemset Generation:** Generate all frequent itemsets for a given $f$.
    *   Brute-force algorithm.
    *   Apriori algorithm.
2.  **Phase 2: Rule Generation:** From frequent itemsets, generate association rules satisfying confidence $c$.
    *   For each frequent itemset $I$:
        *   Partition $I$ into all possible pairs of non-empty subsets $(X, Y)$ such that $Y = I-X$ (so $X \cup Y = I$).
        *   Compute $conf(X \Rightarrow Y) = \frac{sup(I)}{sup(X)}$. If $\ge c$, store the rule.
    *   **Confidence Monotonicity Property (for pruning in Phase 2):** If $X_1 \subset X_2 \subset I$, then $conf(X_2 \Rightarrow I-X_2) \ge conf(X_1 \Rightarrow I-X_1)$. If a rule $X_1 \Rightarrow I-X_1$ has low confidence, then any rule $X_2 \Rightarrow I-X_2$ (where $X_1 \subset X_2$) will also have low or equal confidence (this seems reversed in slide 417, it should be if $X \Rightarrow Y$ is low confidence, then $X' \Rightarrow Y'$ where $X' \subset X$ and $Y' = Y \cup (X \setminus X')$ will have higher or equal confidence. The slide's statement is correct: if $X_1 \Rightarrow I-X_1$ is high confidence, $X_2 \Rightarrow I-X_2$ might be lower, but if $X_2 \Rightarrow I-X_2$ is low, $X_1 \Rightarrow I-X_1$ might be even lower. The key is if $X \Rightarrow Y$ is a rule from $I$, and $X' \subset X$, then $conf(X' \Rightarrow I-X') \le conf(X \Rightarrow I-X)$ because $sup(X') \ge sup(X)$.)
    *   More practically for rule generation from a frequent itemset $I$: if $X \Rightarrow (I-X)$ has low confidence, then for any $X' \supset X$, the rule $X' \Rightarrow (I-X')$ also has low confidence (or doesn't exist if $I-X'$ is empty). This is not what the slide says.
    *   The slide's property: $conf(X_2 \Rightarrow I-X_2) \ge conf(X_1 \Rightarrow I-X_1)$ for $X_1 \subset X_2 \subset I$. This means if $X_1 \Rightarrow I-X_1$ is a high-confidence rule, we don't need to check $X_2 \Rightarrow I-X_2$ if we are looking for rules with *at least* that confidence. More useful: if $X \Rightarrow Y$ is a rule, then for any $Y' \subset Y$, $conf(X \cup Y' \Rightarrow Y \setminus Y')$ will be higher or equal.
    *   Actually, the property is: if $X \rightarrow Y$ is a rule, then $X \rightarrow Y \setminus \{a\}$ has higher or equal confidence. And $X \cup \{a\} \rightarrow Y \setminus \{a\}$ has higher or equal confidence.
    *   The slide's property $conf(X_2 \Rightarrow I-X_2) \ge conf(X_1 \Rightarrow I-X_1)$ for $X_1 \subset X_2 \subset I$ is correct because $sup(X_1) \ge sup(X_2)$. This means if $X_1 \Rightarrow I-X_1$ fails the confidence threshold, $X_2 \Rightarrow I-X_2$ might still pass. If $X_2 \Rightarrow I-X_2$ fails, then $X_1 \Rightarrow I-X_1$ will also fail. This is useful for pruning.

### 26. Frequent Itemset Generation Algorithms

#### 26.1. Brute Force Algorithm
*   Let $U$ be universe of items, $d=|U|$.
*   There are $2^d-1$ distinct non-empty subsets (candidate itemsets).
*   **Algorithm:**
    1.  For every non-empty subset $I$ of $U$:
    2.  Compute $sup(I)$.
    3.  If $sup(I) \ge f$, add $I$ to frequent itemsets.
*   **Issue:** Exponential time complexity (e.g., if $|U|=1000$, $2^{1000} > 10^{300}$ candidates).

#### 26.2. Improved Brute Force Algorithm (Level-wise)
*   Uses Downward Closure: If no $k$-itemset is frequent, then no $(k+1)$-itemset is frequent.
*   **Algorithm:**
    1.  For $k=1$ to $|U|$:
    2.  For every $k$-itemset $I$:
    3.  Compute $sup(I)$.
    4.  If $sup(I) \ge f$, add $I$ to frequent itemsets.
    5.  If no $k$-itemset is frequent, then STOP.
*   Better for sparse datasets (transactions have few items).
*   If $l$ is max items in a transaction, at most $\sum_{i=1}^l \binom{|U|}{i}$ candidates. Still too many if $|U|$ is large (e.g., $|U|=1000, l=10 \implies \approx 10^{23}$ candidates).
*   Most expensive operation: computing support (depends on dataset size).

#### 26.3. Apriori Algorithm
*   Main idea: Ignore candidate $(k+1)$-itemsets that do not satisfy Downward Closure Property (i.e., if any of their $k$-subsets are not frequent).
*   $\mathcal{C}_k$: set of candidate $k$-itemsets.
*   $\mathcal{F}_k$: set of frequent $k$-itemsets.
*   **Algorithm `Apriori(U, D, f)`:**
    1.  $\mathcal{F}_1 = \{\text{frequent 1-itemsets}\}$. $\mathcal{F}_i = \emptyset$ for $i=2, \dots, d$.
    2.  For $k = 2, \dots, d$:
    3.  If $\mathcal{F}_{k-1}$ is empty, break.
    4.  $\mathcal{C}_k = \text{generate-candidates}(\mathcal{F}_{k-1}, k)$.
    5.  For every $I \in \mathcal{C}_k$:
    6.  If $sup(I) \ge f$:
    7.  Add $I$ to $\mathcal{F}_k$.
    8.  Return $\bigcup_{i=1}^d \mathcal{F}_i$.

*   **Assumptions for `generate-candidates`:**
    *   Items in $U$ are ordered (e.g., $U=\{1,2,\dots,d\}$).
    *   Itemsets are ordered lexicographically.
*   **`generate-candidates(F_k-1, k)` function:**
    *   **Join Phase:**
        *   Create candidate $k$-itemsets $\mathcal{C}_k$ by joining pairs of frequent $(k-1)$-itemsets from $\mathcal{F}_{k-1}$.
        *   Two frequent $(k-1)$-itemsets $I_1 = \{j_1, \dots, j_{k-2}, j_{k-1}\}$ and $I_2 = \{j_1, \dots, j_{k-2}, j'_ {k-1}\}$ (i.e., they share first $k-2$ items) are joined if $j_{k-1} < j'_{k-1}$ to form candidate $I = \{j_1, \dots, j_{k-2}, j_{k-1}, j'_{k-1}\}$.
        *   Example from slide 432 (slightly different logic for join):
            For each $I = \{j_1, \dots, j_{k-2}, j_{k-1}\} \in \mathcal{F}_{k-1}$ (ordered):
            For $j = j_{k-1}+1, \dots, d$:
            Let $I' = \{j_1, \dots, j_{k-2}, j\}$. If $I' \in \mathcal{F}_{k-1}$:
            Add $\{j_1, \dots, j_{k-2}, j_{k-1}, j\}$ to $\mathcal{C}_k$.
    *   **Prune Phase:**
        *   For each candidate $I \in \mathcal{C}_k$:
        *   For each $(k-1)$-subset $s$ of $I$:
        *   If $s \notin \mathcal{F}_{k-1}$, remove $I$ from $\mathcal{C}_k$ and break (move to next candidate).
*   **Example (slide 433):** Shows how candidates are generated and pruned.
    *   E.g., if {1,2} and {1,4} are frequent 2-itemsets, {1,2,4} is a candidate 3-itemset.
    *   Then check if all its 2-subsets ({1,2}, {1,4}, {2,4}) are in $\mathcal{F}_2$. If {2,4} is not frequent, prune {1,2,4}.

## Part 6: Graph Mining

*   Networks are everywhere: Air transportation, social networks, cattle movements, email exchange, program flow, chemical reactions, power grids, molecular graphs.

### 27. Graph Mining Topics
*   Graph Classification, Clustering, Pattern Mining, Compression, Dynamics.
*   **Social Network Analysis (SNA).**
*   Graph Visualisation, Link Analysis.

### 28. Possible Settings
*   **Database of many small graphs:** (e.g., chemical/biological data). Tasks: pattern mining, classification, clustering.
*   **A single large graph:** (e.g., web graph, social network). Tasks: community detection, influential nodes, node ranking, link prediction.

### 29. Social Network Analysis (SNA) - Graph Theory Preliminaries

#### 29.1. Graphs
*   A graph $G=(V,E)$ consists of a set of **nodes (vertices)** $V$ and a set of **edges** $E$ (pairs of nodes).
*   Elements of $E$ are edges or **links**.
*   If $(u,v) \in E$, $u$ and $v$ are **adjacent**. Edge $e=(u,v)$ is **incident** with $u$ and $v$. $u,v$ are **endpoints** of $e$.
*   Visual representation: nodes as points/circles, edges as lines.

#### 29.2. Undirected vs. Directed Graphs
*   **Undirected graphs:** Edges are unordered pairs, $(u,v) = (v,u)$.
*   **Directed graphs (Digraphs):** Edges (called **arcs**) are ordered pairs, $(u,v) \ne (v,u)$. $(u,v)$ means an arc from $u$ to $v$.

#### 29.3. Multi-edges and Loops
*   **Multigraph:** Admits multiple edges between a pair of nodes.
*   **Loop:** An edge connecting a vertex to itself, $(u,u)$.
*   (Simple graphs usually don't have multi-edges or loops).

#### 29.4. Weighted Graphs
*   **Edge-weighted:** Each edge has a numerical weight.
*   **Vertex-weighted:** Each vertex has a numerical weight.

#### 29.5. Adjacency Matrix ($\bar{A}$)
*   For an $n$-vertex graph. $\bar{A}$ is an $n \times n$ matrix.
*   **Undirected:** $\bar{A}_{ij}=1$ if $(i,j)$ is an edge, $0$ otherwise. Symmetric.
*   **Directed:** $\bar{A}_{ij}=1$ if $(i,j)$ is an arc (from $i$ to $j$), $0$ otherwise. Not necessarily symmetric.
*   **Weighted:** $\bar{A}_{ij}=w_{ij}$ (weight of edge/arc $(i,j)$) if edge/arc exists, $0$ otherwise.

#### 29.6. Neighbours & Degree
*   **Undirected Graphs:**
    *   Vertex $u$ is a **neighbour** of $v$ if $(u,v)$ is an edge.
    *   **Neighbourhood $N(v)$:** Set of all neighbours of $v$.
    *   **Degree $deg(v)$:** Number of neighbours of $v$, i.e., $|N(v)|$.
*   **Directed Graphs:**
    *   $u$ is an **in-neighbour** of $v$ if $(u,v)$ is an arc.
    *   $u$ is an **out-neighbour** of $v$ if $(v,u)$ is an arc.
    *   **In-degree $deg_+(v)$:** Number of in-neighbours of $v$.
    *   **Out-degree $deg_-(v)$:** Number of out-neighbours of $v$.

#### 29.7. Path & Distance
*   **Undirected Graphs:**
    *   A **path** between $v_0$ and $v_k$ is a sequence of distinct vertices $v_0, v_1, \dots, v_k$ where $(v_{i-1}, v_i)$ is an edge for all $i=1, \dots, k$.
    *   **Length** of path: Number of edges.
    *   **Distance $d(u,v)$:** Length of a shortest path between $u$ and $v$. If no path, $d(u,v)=\infty$.
*   **Directed Graphs:**
    *   A **(directed) path** from $v_0$ to $v_k$ is a sequence $v_0, \dots, v_k$ where $(v_{i-1}, v_i)$ is an arc for all $i$.
    *   Length and distance defined similarly.

#### 29.8. Connected Graphs
*   **Undirected Graphs:**
    *   A graph is **connected** if there is a path between any pair of vertices. Otherwise, **disconnected**.
    *   A **connected component** is a maximal connected subgraph.
*   **Directed Graphs:**
    *   A graph is **strongly connected** if for every ordered pair $(u,v)$, there is a directed path from $u$ to $v$.
    *   A **strongly connected component (SCC)** is a maximal strongly connected subgraph.

### 30. Social Network Analysis (SNA) - Measures of Centrality and Prestige

*   **Actors:** Objects of interest (e.g., people).
*   **Interactions/Relationships:** Represented by edges in a graph $G=(V,E)$.
*   **Goal:** Identify "important", "central", or "influential" nodes/actors.

#### 30.1. Measures of Centrality (for Undirected Graphs)
1.  **Degree Centrality ($C_D(i)$):**
    *   $C_D(i) = \frac{deg(i)}{n-1}$ (where $n$ is total number of nodes).
    *   Motivation: High-degree nodes (hubs) are central, connect distant parts.
    *   Problem: Uses local information only; ignores overall network structure. (Node 1 in slide 461 has high degree but is peripheral).
2.  **Closeness Centrality ($C_C(i)$):**
    *   Defined for connected, undirected graphs.
    *   $AvDist(i) = \frac{\sum_{j=1}^n dist(i,j)}{n-1}$: average shortest path distance from node $i$ to all other nodes.
    *   $C_C(i) = \frac{1}{AvDist(i)}$. Ranges between 0 and 1 (practically, if $n>1$, $AvDist(i) \ge 1$, so $C_C(i) \le 1$).
    *   Node 3 in slide 463 has highest closeness centrality.
    *   Problem: Doesn't account for criticality in terms of paths passing through it.
3.  **Betweenness Centrality ($C_B(i)$):**
    *   $q_{jk}$: number of shortest paths between nodes $j$ and $k$.
    *   $q_{jk}(i)$: number of shortest paths between $j$ and $k$ that pass through node $i$.
    *   $f_{jk}(i) = \frac{q_{jk}(i)}{q_{jk}}$: fraction of shortest paths between $j,k$ that pass through $i$. (Level of control $i$ has over flow between $j,k$).
    *   $C_B(i) = \frac{\sum_{j<k} f_{jk}(i)}{\binom{n}{2}}$. Average value of $f_{jk}(i)$ over all pairs of nodes.
    *   Ranges between 0 and 1. Higher values = better betweenness.
    *   Can be defined for disconnected networks.
    *   Node 4 in slide 467 has highest betweenness.
    *   Can be generalized to edges (edge betweenness). Used in community detection (e.g., Girvan-Newman).

#### 30.2. Measures of Prestige (for Directed Graphs)
1.  **Degree Prestige ($P_D(i)$):**
    *   $P_D(i) = \frac{deg_+(i)}{n-1}$ (uses in-degree).
    *   Motivation: High in-degree means many "votes" for popularity/importance.
    *   Node 1 in slide 471 has highest degree prestige.
2.  **Proximity Prestige ($P_P(i)$):**
    *   $Influence(i)$: Set of nodes that can reach node $i$ with a direct path.
    *   $AvDist(i) = \frac{\sum_{j \in Influence(i)} dist(j,i)}{|Influence(i)|}$: average shortest path distance **to** node $i$ from nodes in its influence set.
    *   Inverse of $AvDist(i)$ is not fair (e.g., node with small influence set might have low $AvDist(i)$).
    *   Penalty factor: $InfluenceFraction(i) = \frac{|Influence(i)|}{n-1}$.
    *   $P_P(i) = \frac{InfluenceFraction(i)}{AvDist(i)}$.
    *   Lies between 0 and 1. Higher values = greater prestige.

### 31. PageRank Algorithm

*   Proposed by Sergey Brin and Larry Page (1998) for Google Search.
*   Ranks web pages based on the web graph structure. Static ranking (offline, query-independent).
*   **Web Graph:** Nodes = webpages, Directed edges = hyperlinks.
*   **Terminology:**
    *   **In-links** of page $a$: Hyperlinks pointing to $a$.
    *   **Out-links** of page $a$: Hyperlinks pointing from $a$.

#### 31.1. PageRank Idea
*   PageRank score of a page is its prestige.
*   Hyperlink from page $x$ to page $a$ is a "vote" from $x$ for $a$.
*   Unlike Degree Prestige, PageRank considers the importance of the voting page.
*   Votes from important pages weigh more.
*   Importance of page $a$ ($P(a)$) is sum of PageRank scores of all pages $x$ pointing to $a$, divided by their out-degree $O_x$.
    *   $P(a) = \sum_{(x,a) \in E} \frac{P(x)}{O_x}$.
*   This forms a system of linear equations.
    *   Let $\bar{P} = (P(1), \dots, P(n))^T$ be the vector of PageRank scores.
    *   Let $\bar{A}$ be a modified adjacency matrix where $\bar{A}_{ij} = 1/O_i$ if $(i,j) \in E$ (arc from $i$ to $j$), and $0$ otherwise.
    *   System: $\bar{P} = \bar{A}^T \bar{P}$.
    *   $\bar{P}$ is an eigenvector of $\bar{A}^T$ with eigenvalue 1.
*   Can be found by **power iteration** if $\bar{A}$ satisfies certain conditions:
    *   Start with $\bar{P}_0$. Iterate $\bar{P}_k = \bar{A}^T \bar{P}_{k-1}$ until $||\bar{P}_k - \bar{P}_{k-1}||_1 \le \epsilon$.
*   Real web graph doesn't satisfy these conditions.

#### 31.2. PageRank: Markov Chain Perspective
*   Nodes = states. Arcs = transitions. At a fixed state, all transitions equally probable.
*   Models a **random Web surfer**.
*   Transition probability from $x$ to a neighbour $y$ is $1/O_x$.
*   Let $\bar{A}$ be the state transition probability matrix: $\bar{A}_{ij}$ is prob. of surfer in state $i$ moving to state $j$.
    *   $\bar{A}_{ij} = 1/O_i$ if $(i,j) \in E$, $0$ otherwise.
*   **Conditions for unique stationary distribution $\bar{\Pi}$:**
    1.  $\bar{A}$ must be **stochastic** (rows sum to 1: $\sum_j \bar{A}_{ij}=1$ for all $i$).
        *   Problem: **Dangling pages** (no out-links) make rows sum to 0.
    2.  $\bar{A}$ must be **irreducible** (graph is strongly connected).
    3.  $\bar{A}$ must be **aperiodic** (no fixed cycles).
*   If these hold, power iteration $\bar{P}_k = \bar{A}^T \bar{P}_{k-1}$ converges to $\bar{\Pi}$, where $\bar{\Pi} = \bar{A}^T \bar{\Pi}$. $\bar{\Pi}$ is the principal eigenvector.

#### 31.3. Modifications to $\bar{A}$ for Real Web Graph
1.  **Make $\bar{A}$ stochastic (Handle dangling pages):**
    *   For every dangling page $i$, add outgoing links to *all* pages in the graph, each with probability $1/n$.
    *   Row $i$ of $\bar{A}$ becomes $(1/n, 1/n, \dots, 1/n)$.
2.  **Make $\bar{A}$ irreducible and aperiodic (Teleportation):**
    *   Add a link from each page to every other page with a small transition probability.
    *   Modified transition matrix $\bar{M}$: $\bar{M}^T = (1-d)\bar{A}^T + d \frac{\bar{E}}{n}$.
        *   $\bar{A}^T$ is the (stochastic-modified) transpose adjacency matrix.
        *   $d$ is the damping factor (e.g., 0.15, so $1-d=0.85$). Probability of random jump.
        *   $\bar{E}$ is an $n \times n$ matrix of all 1s. $\frac{\bar{E}}{n}$ represents uniform jump probability.
    *   The surfer follows existing links with probability $1-d$, and "teleports" to a random page with probability $d$.
    *   This ensures $\bar{M}$ is stochastic, irreducible, and aperiodic.

#### 31.4. PageRank Algorithm (Final Form)
`PageRank(Graph G=(V,E), damping factor d, tolerance \epsilon)`
(Assumes G has no dangling vertices, or they've been handled by making $\bar{A}$ stochastic)
1.  Initialize $P_0(i) = 1/n$ for all $i=1, \dots, n$.
2.  $k=1$.
3.  **repeat**
4.  For each page $i=1, \dots, n$:
    $P_k(i) = \frac{1-d}{n} + d \sum_{(x,i) \in E} \frac{P_{k-1}(x)}{O_x}$.
5.  $k = k+1$.
6.  **until** $|| \bar{P}_k - \bar{P}_{k-1} ||_1 \le \epsilon$. (Sum of absolute differences).
7.  Return $\bar{P}_k$.

*   **Remarks:**
    *   Typical $d=0.85$.
    *   Can be applied to any graph for ranking vertices.
    *   Convergence may not be strictly necessary if only ranking is needed; can terminate after max iterations.
    *   Reported to converge in ~52 iterations for 322 million links.

## Part 7: Data Visualization

### 32. Basic Components of Visualisation

*   **Aesthetics:** Visual properties used to represent data.
    *   **Position:** Where the element is located (e.g., X,Y axes).
    *   **Shape:** Form of the element (e.g., circle, square, triangle).
    *   **Size:** Magnitude of the element.
    *   **Color:** Hue, saturation, brightness.
*   **Types of Data (for mapping to aesthetics):**
    *   Quantitative/numerical continuous (e.g., [1.5, 2.6]).
    *   Numerical discrete, categorical (ordered/unordered) (e.g., [dog, cat], [good, fair, poor]).
    *   Date/time.
    *   Text.

### 33. Types of Positions
*   **Cartesian coordinates:** Standard X-Y plots.
*   **Nonlinear axes:** Logarithmic scales (e.g., `log_10(x)`).
*   **Coordinate systems with curved axes:** Polar coordinates (e.g., radar charts).

### 34. Color Scales
*   Many qualitative color scales available.
*   **ColorBrewer project:** Provides good selections (e.g., `Okabe Ito`, `ColorBrewer Dark2`, `ggplot2 hue`).

### 35. Basic Visualisation Principles
*   Mapping between data values and aesthetics values via **scales**.
*   A scale must be **one-to-one**.
*   Sizes of shaded areas in a visualisation must be **proportional** to the data values they represent (e.g., bar charts should start at 0).
*   Use larger axis labels.
*   Avoid unnecessary line drawings (e.g., empty bars in histograms can be filled).
*   Choose the right visualization software/library.

### 36. Types of Visualisation (by Purpose)

*   **Amount:**
    *   **Bar Chart:** Good for comparing amounts across categories. Horizontal for long labels.
    *   **Heat Map:** Uses color intensity to show magnitude in a 2D matrix/grid.
*   **Distribution:**
    *   **Density Plot:** Smoothed version of a histogram, shows distribution shape. Can be scaled by count.
    *   **Histogram:** Shows frequency distribution by binning data. Can overlap for comparison.
*   **Proportions:**
    *   **Pie Chart:** Shows parts of a whole. Best for few categories.
    *   **Stacked Bar Chart / Side-by-Side Bars:** Shows proportions within categories, or how proportions change.
*   **X-Y Relationships:**
    *   **Scatter Plot:** Shows relationship between two continuous variables. Each dot is an observation.
    *   **Correlograms:** Matrix of scatter plots showing correlations between multiple pairs of variables.
*   **Geospatial Data:**
    *   **Projections:** Transforming spherical Earth data to a flat map (e.g., Mercator). Introduces distortions.
    *   **Layers:** Overlaying different types of data on a map (e.g., wind turbines on a map of SF Bay Area).
*   **Uncertainty:**
    *   **Error Bars:** Show range of uncertainty (e.g., standard deviation, confidence interval) around a point estimate.
    *   **Confidence Strips/Bands:** Show uncertainty range along a line or curve.
*   **Trends:**
    *   **Smoothing (Moving Averages):** Shows underlying trend in time series data by averaging over windows (e.g., 20-day, 50-day moving average for stock prices).
    *   **Time Series Decomposition:** Break down a time series into trend, seasonal fluctuations, and remainder (noise) (e.g., Keeling curve).

### 37. Common Pitfalls for Color Coding
1.  **Encoding too much or irrelevant information:**
    *   Qualitative color scales work best for 3-5 categories. More than 8-10 becomes burdensome to match colors to legend.
    *   Consider direct labeling for many categories instead of relying solely on color.
2.  **Using non-monotonic color scales to encode data values:**
    *   Rainbow color scale is highly non-monotonic in lightness. Converting to grayscale reveals this. Lightest part (yellow/cyan) can take up large portion of scale, while darkest (blue) is concentrated. Can mislead.
3.  **Not designing for color-vision deficiency:**
    *   ~8% of men have some form of color blindness. Use colorblind-safe palettes. Rainbow scale is particularly bad for this.

### 38. Python Visualisation Libraries
*   Matplotlib
*   Seaborn (built on Matplotlib, higher-level interface)
*   Plotting (likely refers to pandas .plot() or similar)
*   Bokeh (interactive web visualisations)
*   Pygal (SVG charts)
*   Geoplotlib (geographical data)

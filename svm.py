import numpy as np
from typing import Optional

class SVM:
    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_passes: int = 5,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Simplified SVM classifier using SMO for linear kernel.
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state

        self.alpha = None  # Lagrange multipliers
        self.b = 0.0       # Bias term
        self.w = None      # Weight vector

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit the SVM model on training data using simplified SMO.
        """
        self.X = X
        self.y = y.copy()
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        passes = 0
        it = 0

        # Main SMO loop
        while passes < self.max_passes and it < self.max_iter:
            num_changed = 0

            for i in range(n_samples):
                xi, yi = X[i], y[i]

                # Compute f(x_i) and error E_i
                fxi = np.sum(self.alpha * self.y * (self.X @ xi)) + self.b
                Ei = fxi - yi

                # Check if KKT conditions are violated
                if (yi * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (yi * Ei > self.tol and self.alpha[i] > 0):

                    # Select a random j ≠ i
                    j = np.random.choice([_ for _ in range(n_samples) if _ != i])
                    xj, yj = X[j], y[j]
                    fxj = np.sum(self.alpha * self.y * (self.X @ xj)) + self.b
                    Ej = fxj - yj

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # Compute bounds L and H for alpha[j]
                    if yi != yj:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue

                    # Compute eta (second derivative of objective function)
                    eta = 2 * xi @ xj - xi @ xi - xj @ xj
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alpha[j] -= yj * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # Check if change in alpha[j] is significant
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i] accordingly
                    self.alpha[i] += yi * yj * (alpha_j_old - self.alpha[j])

                    # Compute new thresholds b1 and b2
                    b1 = self.b - Ei \
                        - yi * (self.alpha[i] - alpha_i_old) * (xi @ xi) \
                        - yj * (self.alpha[j] - alpha_j_old) * (xi @ xj)
                    b2 = self.b - Ej \
                        - yi * (self.alpha[i] - alpha_i_old) * (xi @ xj) \
                        - yj * (self.alpha[j] - alpha_j_old) * (xj @ xj)

                    # Select updated bias based on constraints
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed += 1

            # Check for convergence
            passes = passes + 1 if num_changed == 0 else 0
            it += 1

        # Compute final weight vector from support vectors
        self.w = ((self.alpha * self.y)[:, None] * self.X).sum(axis=0)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for given inputs.
        """
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {-1, 1} using the sign of the decision function.
        """
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy of the model on given test data.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
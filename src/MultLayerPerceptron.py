import numpy as np
from typing import List, Tuple, Optional

class MultiLayerPerceptron:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        output_activation: str = 'softmax',
        learning_rate: float = 0.01,
        random_state: Optional[int] = None  # Added random_state as an optional parameter
    ):
        """
        Initialize the Multi-Layer Perceptron (MLP).

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_sizes : list of int
            Sizes of hidden layers.
        output_size : int
            Number of output neurons (usually number of classes).
        activation : str
            Activation function for hidden layers ('relu' or 'sigmoid').
        output_activation : str
            Activation function for output layer ('softmax' or 'sigmoid').
        learning_rate : float
            Learning rate for weight updates.
        random_state : int, optional
            Seed for reproducibility.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Set the random seed if random_state is provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        # Define full architecture: input → hidden layers → output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros((1, fan_out)))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        if activation == 'sigmoid':
            f = self._sigmoid(x)
            return f * (1 - f)
        elif activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        pre_activations = []
        a = X

        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            if i == len(self.weights) - 1:
                if self.output_activation == 'softmax':
                    a = self._softmax(z)
                elif self.output_activation == 'sigmoid':
                    a = self._sigmoid(z)
                else:
                    raise ValueError(f"Unsupported output activation: {self.output_activation}")
            else:
                if self.activation == 'relu':
                    a = self._relu(z)
                elif self.activation == 'sigmoid':
                    a = self._sigmoid(z)
                else:
                    raise ValueError(f"Unsupported hidden activation: {self.activation}")

            activations.append(a)

        return activations, pre_activations

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        weight_grads = []
        bias_grads = []

        delta = activations[-1] - y

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            z = pre_activations[i]

            dw = a_prev.T @ delta / X.shape[0]
            db = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

            weight_grads.insert(0, dw)
            bias_grads.insert(0, db)

            if i > 0:
                da = delta @ self.weights[i].T
                dz = da * self._activation_derivative(pre_activations[i - 1], self.activation)
                delta = dz

        return weight_grads, bias_grads

    def update_parameters(self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray]) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_grads[i]
            self.biases[i] -= self.learning_rate * bias_grads[i]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> List[float]:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            step = batch_size or n_samples
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n_samples, step):
                end = start + step
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                activations, pre_activations = self.forward(X_batch)
                y_pred = np.clip(activations[-1], 1e-8, 1 - 1e-8)

                if self.output_size == 1:
                    loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                else:
                    loss = -np.mean(np.sum(y_batch * np.log(y_pred), axis=1))

                dW, dB = self.backward(X_batch, y_batch, activations, pre_activations)
                self.update_parameters(dW, dB)

                epoch_loss += loss
                n_batches += 1

            losses.append(epoch_loss / n_batches)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} — loss: {losses[-1]:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        y_pred = activations[-1]

        if self.output_activation == 'sigmoid':
            return (y_pred > 0.5).astype(int)
        elif self.output_activation == 'softmax':
            return np.argmax(y_pred, axis=1)
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]
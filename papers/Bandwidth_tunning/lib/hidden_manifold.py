import numpy as np
from sklearn.preprocessing import StandardScaler


### TWO CURVES
def fourier_series(t, coeffs, degree=5, noise=0.00):
    """Fourier series of input t.

    Args:
        t (float): scalar input
        coeffs (ndarray): coefficient tensor of dimension ()
        degree (int): maximum degree of Fourier series
        noise (flaot): standard deviation of Gaussian noise added to output
    """
    scaling = 0.5 * 2 * np.pi
    res = coeffs[0, 0] + coeffs[0, 1]
    for frequency in range(1, degree + 1):
        res += coeffs[frequency, 0] * np.cos(frequency * scaling * t) + coeffs[
            frequency, 1
        ] * np.sin(frequency * scaling * t)
    return res + np.random.normal(loc=0, scale=noise)


def generate_two_curves(n_samples, n_features, degree = 5, offset = 0.1, noise = 0.1, seed = 1):
    """Data generation procedure for 'two curves'.

    
    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        degree (int): maximum degree of Fourier series
        offset (float): distance between two curves
        noise (float): standard deviation of Gaussian noise added to curves
    """
    np.random.seed(seed)
    fourier_coeffs = np.random.uniform(size=(n_features, degree + 1, 2))
    fourier_coeffs = fourier_coeffs / np.linalg.norm(fourier_coeffs)

    # first manifold
    A = np.zeros(shape=(n_samples // 2, n_features))
    for s in range(n_samples // 2):
        # sample a point on the curve
        t = np.random.rand()
        # embed this point
        # every component is computed by another Fourier series
        for i in range(n_features):
            A[s, i] = fourier_series(t, fourier_coeffs[i], degree=degree, noise=noise)

    # second manifold: use same fourier series, plus offset
    B = np.zeros(shape=(n_samples // 2, n_features))
    for s in range(n_samples // 2):
        t = np.random.rand()
        for i in range(n_features):
            B[s, i] = fourier_series(t, fourier_coeffs[i], degree=degree, noise=noise)
    B = np.add(B, offset)

    X = np.r_[A, B]
    y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))

    s = StandardScaler()
    X = s.fit_transform(X)

    return X, y

def neural_net(x, W, v):
    """Transforms inputs via a single-layer neural network.
    Args:
        x (ndarray): input of shape (manifold_dimension,)
        W (ndarray): input-to-hidden weight matrix of shape (manifold_dimension, manifold_dimension)
        v (ndarray): hidden-to-output weight matrix of shape (manifold_dimension,)
    """
    return np.dot(v, np.tanh(W @ x) / np.sqrt(W.shape[0]))


def nonlinearity(X, biases):
    """Element-wise nonlinearity.

    Args:
        X (ndarray): inputs of shape (n_samples, n_features)
        biases (ndarray): biases of shape (n_features,)
    """
    return np.tanh(X - biases)


def generate_hidden_manifold_model(n_samples, n_features=10, manifold_dimension = 6, seed = 1):
    """Data generation procedure for the 'hidden manifold model'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        manifold_dimension (int): dimension of hidden maniforls
    """

    np.random.seed(seed)
    # feature matrix F controls the embedding of the manifold
    F = np.random.normal(size=(manifold_dimension, n_features))

    # Gaussian matrix samples original inputs from the lower-dimensional manifold
    C = np.random.normal(size=(n_samples, manifold_dimension), loc=0, scale=1)

    # embed data, adding an element-wise nonlinearity
    biases = 2 * np.random.uniform(size=(n_features,)) - 1
    X = nonlinearity(C @ F / np.sqrt(manifold_dimension), biases)

    # define labels via a neural network
    W = np.random.normal(size=(manifold_dimension, manifold_dimension))
    v = np.random.normal(size=(manifold_dimension,))
    y = np.array([neural_net(c, W, v) for c in C])

    # post-process the labels to get balanced classes
    y = y - np.median(y)
    y = np.array([-1 if y_ < 0 else 1 for y_ in y])
    assert len(X[y == 1]) == n_samples // 2
    assert len(X[y == -1]) == n_samples // 2

    return X, y
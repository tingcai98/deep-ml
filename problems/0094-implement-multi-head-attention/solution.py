import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    # pass
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return tuple([Q, K, V])

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    # pass
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    scores -= np.max(scores, axis = 1, keepdims = True)

    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

    output = weights @ V

    return output
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # pass
    seq_len, d_model = Q.shape

    assert d_model % n_heads == 0

    head_dim = d_model // n_heads
    outputs = []

    for i in range(n_heads):
        start = i*head_dim
        end = (i+1)*head_dim

        q = Q[:, start:end]
        k = K[:, start:end]
        v = V[:, start:end]

        head_output = self_attention(q, k, v)
        outputs.append(head_output)

    return np.concatenate(outputs, axis = -1)





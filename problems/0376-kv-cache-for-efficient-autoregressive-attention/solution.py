import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    # pass
    q = x_new @ W_Q # (d_k, )
    k = x_new @ W_K
    v = x_new @ W_V

    if cache is None:
        K_new = k.reshape(1, -1)
        V_new = v.reshape(1, -1)
    else:
        K_cache, V_cache = cache

        K_new = np.concatenate([K_cache, k.reshape(1,-1)], axis = 0)
        V_new = np.concatenate([V_cache, v.reshape(1,-1)], axis = 0)

    d_k = q.shape[0]
    scores = q @ K_new.T / np.sqrt(d_k)
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    weight = exp_scores / np.sum(exp_scores)

    output = weight @ V_new

    return output, (K_new, V_new)






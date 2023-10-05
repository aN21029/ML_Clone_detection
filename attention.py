import numpy as np


def _dot_attention_score(dec_hidden_state, enc_hidden_state):
    # return the product of dec_hidden_state transpose and enc_hidden_states
    return np.matmul(np.transpose(dec_hidden_state), enc_hidden_state)


def _softmax(x):
    x = np.array(x, dtype=np.float64)
    normalized_x = x / np.sqrt(np.sum(x**2))
    e_x = np.exp(normalized_x)
    return e_x / e_x.sum(axis=0)


def _apply_attention_scores(attention_weights, enc_hidden_state):
    # Multiply the encoder hidden states by their weights
    return attention_weights * enc_hidden_state


def _calculate_attention_vector(applied_attention):
    return np.sum(applied_attention, axis=1)


def apply_attention(enc_hidden_state, dec_hidden_state):
    annotations = np.transpose(enc_hidden_state)
    attention_weights_raw = _dot_attention_score(dec_hidden_state, annotations)
    attention_weights = _softmax(attention_weights_raw)
    applied_attention = _apply_attention_scores(attention_weights, annotations)
    attention_vector = _calculate_attention_vector(applied_attention)
    return attention_vector
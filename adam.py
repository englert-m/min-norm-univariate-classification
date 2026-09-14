import numpy as np
from scipy.special import expit

def compute_grads(model, x, y, actual_weight_decay=0.0, include_biases=True):
    """Return gradients of the regularized objective.

    The regularizer is actual_weight_decay * 0.5 * ||theta||^2, applied to
    W1/W2 and optionally b1.  Skip parameters are never regularized.
    """
    y = np.asarray(y, dtype=DTYPE).reshape(-1)
    logits, (x_flat, z, h) = model.forward(x, return_cache=True)
    margin = float(np.min(y * logits))
    
    n = y.size
    # d/dlogit mean softplus(-y*logit)
    dlogits = (-y * expit(-y * logits)) / n

    grads = {}
    grads['W2'] = h.T @ dlogits

    z[...] = z > 0.0
    z *= dlogits[:, None]
    z *= model.W2[None, :]
    grads["W1"] = z.T @ x_flat
    grads["b1"] = z.sum(axis=0)


    if model.use_skip:
        grads['skip_w'] = np.array([np.sum(dlogits * x_flat)], dtype=DTYPE)
        grads['skip_b'] = np.array([np.sum(dlogits)], dtype=DTYPE)

    if actual_weight_decay != 0.0:
        grads['W1'] = grads['W1'] + actual_weight_decay * model.W1
        grads['W2'] = grads['W2'] + actual_weight_decay * model.W2
        if include_biases:
            grads['b1'] = grads['b1'] + actual_weight_decay * model.b1

    return grads, margin



class AdamNumpy:
    """Minimal Adam implementation matching the needs of this script.

    Defaults mirror torch.optim.Adam defaults:
    betas=(0.9, 0.999), eps=1e-8, no AMSGrad. Weight decay is deliberately not
    implemented here; L2 regularization is added to the objective/gradient in
    compute_grads().
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.step_count = 0
        self.interpolation_adjusted = False
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}

    def step(self, grads):
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        bias_correction1 = 1.0 - b1 ** self.step_count
        bias_correction2 = 1.0 - b2 ** self.step_count
        step_size = self.lr / bias_correction1

        for name, p in self.params.items():
            g = grads[name]
            m = self.m[name]
            v = self.v[name]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            denom = np.sqrt(self.v[name]) / np.sqrt(bias_correction2) + self.eps
            p -= step_size * self.m[name] / denom


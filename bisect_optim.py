import jax

from jax.experimental import optimizers

import numpy
def bisect(dx, eps=2**-16):
    def init(x0):
        return jax.tree_util.tree_map(lambda x: x - dx, x0), jax.tree_util.tree_map(lambda x: x + dx, x0)
    def update(grad, bounds):
        mid = get_params(bounds)
        flat_data, trees = zip(*map(jax.tree_util.tree_flatten,
            bounds + tuple(map(grad, bounds)) + (mid, grad(mid))))

        def f(low, high, grad_low, grad_high, mid, grad_mid):
            sign_grad_low = jax.numpy.sign(grad_low)
            sign_grad_high = jax.numpy.sign(grad_high)
            sign_grad_mid = jax.numpy.sign(grad_mid)
            bounding = sign_grad_low != sign_grad_high
            replace_low = sign_grad_low == sign_grad_mid
            idx = bounding * replace_low
            low += mid * idx - low * idx
            idx = bounding * ~replace_low
            high += mid * idx - high * idx
            abs_grad_low = abs(grad_low)
            abs_grad_high = abs(grad_high)
            low_closer = abs_grad_low < abs_grad_high
            idx = ~bounding * low_closer * (abs_grad_low > eps)
            high += 3 * idx * (low - high)
            idx = ~bounding * ~low_closer * (abs_grad_high > eps)
            low += 3 * idx * (high - low)
            return low, high
        return tuple(map(lambda tree: trees[0].unflatten(tree),
                zip(*map(lambda x: f(*x),
                    zip(*map(jax.tree_util.tree_leaves,
                        flat_data))))))

    def get_params(bounds):
        return jax.tree_util.tree_multimap(lambda x, y: (x + y) / 2, *bounds)

    return init, update, get_params

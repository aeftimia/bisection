import jax

from scipy.optimize import fmin
from bisect_optim import bisect

def loss(x):
    x, y = x
    return (x + 8) ** 2 + (y - 0.0) ** 2 #+ x * y + 1

opt_init, opt_update, get_params = bisect(1)
x0 = jax.numpy.zeros((2,))
opt_state = opt_init(x0)
print('init')
print(opt_state)
for _ in range(100):
    opt_state = opt_update(lambda params: jax.grad(loss)(params), opt_state)
    print(opt_state)
print('final:', get_params(opt_state))
print(fmin(lambda params: (jax.grad(loss)(params) ** 2).sum(), x0))

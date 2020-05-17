import jax

from scipy.optimize import fmin
from bisect_optim import bisect

def loss(x):
    #x, y = x
    z = ((x - 4) ** 2).sum()
    return jax.numpy.cos(z)

opt_init, opt_update, get_params = bisect(1)
x0 = jax.numpy.zeros((2,), dtype='float')
opt_state = opt_init(x0)
print('init')
print(opt_state)
for _ in range(50):
    opt_state = opt_update(lambda params: jax.grad(loss)(params), opt_state)
    print(opt_state)
pred =  get_params(opt_state)
print('final:', pred, loss(pred))
print(fmin(loss, x0))

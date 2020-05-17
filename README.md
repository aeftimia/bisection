# What Is This?

When you spend too much time thinking about gradient descent, you realize it's actually a bad rootfinding algorithm for the gradient. Well almost. It also will generally be repelled from maxima and attracted to minima--so it finds roots of the derivatives such that these derivatives are increasing when they reach zero. You could imagine a root finding algorithm that looks like this:

$$x_{n+1} = x_n - g(x_n) \mathrm{step}$$

The idea is you move up if the function is increasing, and down when the function is decreasing, and decrease your stepsize according to the absolute value of the function itself. This generalizes to multidimensional functions by just applying gradient descent to each coordinate simultaneously (i.e. via vector addition). That made me wonder why we don't use bisection method for this? As a rule, bisection method does not apply to multidimensional functions, but if you can simultaneously take steps accross multiple coordinates during gradient descent, why not try the same thing with bisection method?

In this hypothetical generalization to the bisection method, you use a midpoint defined by the coordinate-wise midpoint at the center of a hypercube bounded by a _high_ coordinate (representing the largest value of each coordinate) and a _low_ coordinate (representing the smallest value of each coordinate). Hence mid = (high + low) / 2. We can then apply bisection method to each of these coordinates independently using their respective element of a vector valued gradient (just like how we apply gradient descent to each coordinates independently using a vector addition between coordinates and the gradient at that point). With an added check to ensure low coordinates are only moved higher if they are negative (and otherwise moved lower) when the sign of the function is negative (and similarly for high coordinates), we can provide similar assurance of convergence on a local minima as opposed to either a minimum or a maximum.

This seems to work for traditional optimization, however it does not seem to work for optimizing a neural network. I wonder whether gradient descent is more amenable to stochastic generalizations with minibatches because the expected result of a gradient update is the same as the infinite sample limit (i.e. has zero bias). It isn't clear that this would be the case for a bisection based approach.

However, this might still be useful for hyperparameter optimization.

import functools
from collections import namedtuple

import jax.numpy as jnp
from jax import vmap

from trajax import integrators, optimizers

m1 = 1.4
m2 = 1.1
l1 = 0.3
l2 = 0.33
s1 = 0.11
s2 = 0.16
I1 = 0.025
I2 = 0.045

# eqn 3.5
d1 = I1 + I2 + m2 * l1 ** 2
d2 = m2 * l1 * s2
d3 = I2

bii = 0.05
bij = 0.025
T = 50

ReachingParams = namedtuple('ReachingParams', ['action_cost', 'velocity_cost', 'target'])


def e(x):
    return jnp.array([l1 * jnp.cos(x[0]) + l2 * jnp.cos(x[0] + x[1]),
                      l1 * jnp.sin(x[0]) + l2 * jnp.sin(x[0] + x[1])])


def gamma(x):
    return jnp.array([[-l1 * jnp.sin(x[0]) - l2 * jnp.sin(x[0] + x[1]), -l2 * jnp.sin(x[0] + x[1])],
                      [l1 * jnp.cos(x[0]) + l2 * jnp.cos(x[0] + x[1]), l2 * jnp.cos(x[0] + x[1])]])


def edot(x):
    return gamma(x) @ x[2:]


def arm(x, u, t):
    det = d1 * d3 - d3 ** 2 - (d2 * jnp.cos(x[1])) ** 2
    dx = jnp.array([x[2],
                    x[3],
                    1 / det * (-d2 * d3 * (x[2] + x[3]) ** 2 * jnp.sin(x[1]) - d2 ** 2 *
                               x[2] ** 2 * jnp.sin(
                                x[1]) * jnp.cos(x[1]) - d2 * (
                                       bij * x[2] + bii * x[3]) * jnp.cos(x[1]) + (
                                       d3 * bii - d3 * bij) * x[2] + (
                                       d3 * bij - d3 * bii) * x[3]),
                    1 / det * (d2 * d3 * x[3] * (2 * x[2] + x[3]) * jnp.sin(
                        x[1]) + d1 * d2 * x[
                                   2] ** 2 * jnp.sin(x[1]) + d2 ** 2 * (
                                       x[2] + x[3]) ** 2 * jnp.sin(x[1]) * jnp.cos(
                        x[1]) + d2 * (
                                       (2 * bij - bii) * x[2] + (2 * bii - bij) * x[
                                   3]) * jnp.cos(x[1]) + (
                                       d1 * bij - d3 * bii) * x[2] + (
                                       d1 * bii - d3 * bij) * x[3])])
    G = 1 / det * jnp.array([[0., 0.],
                             [0., 0.],
                             [d3, -(d3 + d2 * jnp.cos(x[1]))],
                             [-(d3 + d2 * jnp.cos(x[1])), d1 + 2 * d2 * jnp.cos(x[1])]])

    du = G @ u
    return dx + du


dynamics = integrators.euler(arm, 0.01)


def cost(params, x, u, t):
    stagewise_cost = 0.5 * params.action_cost * jnp.sum(u ** 2)
    terminal_cost = jnp.sum((e(x) - params.target) ** 2) + params.velocity_cost * jnp.sum(edot(x) ** 2)
    return jnp.where(t == T, terminal_cost, stagewise_cost)


x0 = jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.])
u0 = jnp.zeros((T, 2))


def solve(params):
    xs, us, c, *solver_outputs = optimizers.ilqr(
        functools.partial(cost, params), dynamics, x0, u0,
        grad_norm_threshold=0)

    return xs, us


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer

    from jax import grad, jit

    params = ReachingParams(action_cost=1e-2, velocity_cost=5e-3, target=jnp.array([0.05, 0.5]))
    xs, us = solve(params=params)

    es = vmap(e)(xs)

    # Plot simulations
    plt.figure()
    plt.plot(es[:, 0], es[:, 1], marker='x')
    plt.show()


    def loss(params, x):
        xs, us = solve(params=params)
        return jnp.sum((xs - x) ** 2)


    grad_solve = jit(grad(loss))

    start = timer()
    for i in range(100):
        grad_solve(params, xs)
    end = timer()
    print(end - start)

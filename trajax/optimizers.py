# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
"""Building Blocks for Gradient-based Trajectory Optimizers.

Notation:

- x denotes state, a 1D numpy array of shape [n]
- u denotes control, a 1D numpy array of shape [m]
- t denotes time, an scalar integer time index.

A Trajectory optimization problem is specified via three components:

(1) A numpy scalar-valued cost function with signature,
              c = cost(x, u, t, *args)

(2) A numpy vector-valued dynamics function with signature,
              xdot = dynamics(x, u, t, *args)

    where xdot is state time derivative of shape [n].

(3) The initial state x0, a 1D numpy array of shape [n].

The problem is to minimize over a sequence u[0], u[1]...u[T-1],

    sum_{t=0}^{T-1} cost(x[t], u[t], t) + cost(x[T], np.zeros(m), T)

    subject to:

      x[t+1] = dynamics(x[t], u[t], t)
      x[0] = x0 is given.
"""

from functools import partial  # pylint: disable=g-importing-member

import jax
from jax import custom_derivatives
from jax import hessian
from jax import jacobian
from jax import jit
from jax import lax
from jax import vmap
import jax.numpy as jnp
from trajax.tvlqr import rollout as tvlqr_rollout
from trajax.tvlqr import tvlqr

# Convenience routine to pad zeros for vectorization purposes.
pad = lambda A: jnp.vstack((A, jnp.zeros((1,) + A.shape[1:])))


def vectorize(fun, argnums=3):
    """Returns a jitted and vectorized version of the input function.

    See https://jax.readthedocs.io/en/latest/jax.html#jax.vmap

    Args:
      fun: a numpy function f(*args) to be mapped over.
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      Vectorized/Batched function with arguments corresponding to fun, but extra
      batch dimension in axis 0 for first argnums arguments (x, u, t typically).
      Remaining arguments are not batched.
    """

    def vfun(*args):
        _fun = lambda tup, *margs: fun(*(margs + tup))
        return vmap(
            _fun, in_axes=(None,) + (0,) * argnums)(args[argnums:], *args[:argnums])

    return vfun


def linearize(fun, argnums=3):
    """Vectorized gradient or jacobian operator.

    Args:
      fun: numpy scalar or vector function with signature fun(x, u, t, *args).
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      A function that evaluates Gradients or Jacobians with respect to states and
      controls along a trajectory, e.g.,

          dynamics_jacobians = linearize(dynamics)
          cost_gradients = linearize(cost)
          A, B = dynamics_jacobians(X, pad(U), timesteps)
          q, r = cost_gradients(X, pad(U), timesteps)

          where,
            X is [T+1, n] state trajectory,
            U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
            timesteps is typically np.arange(T+1)

            and A, B are Dynamics Jacobians wrt state (x) and control (u) of
            shape [T+1, n, n] and [T+1, n, m] respectively;

            and q, r are Cost Gradients wrt state (x) and control (u) of
            shape [T+1, n] and [T+1, m] respectively.

            Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jacobian(fun)
    jacobian_u = jacobian(fun, argnums=1)

    def linearizer(*args):
        return jacobian_x(*args), jacobian_u(*args)

    return vectorize(linearizer, argnums)


def quadratize(fun, argnums=3):
    """Vectorized Hessian operator for a scalar function.

    Args:
      fun: numpy scalar with signature fun(x, u, t, *args).
      argnums: number of leading arguments of fun to vectorize.

    Returns:
      A function that evaluates Hessians with respect to state and controls along
      a trajectory, e.g.,

        Q, R, M = quadratize(cost)(X, pad(U), timesteps)

       where,
            X is [T+1, n] state trajectory,
            U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
            timesteps is typically np.arange(T+1)

      and,
            Q is [T+1, n, n] Hessian wrt state: partial^2 fun/ partial^2 x,
            R is [T+1, m, m] Hessian wrt control: partial^2 fun/ partial^2 u,
            M is [T+1, n, m] mixed derivatives: partial^2 fun/partial_x partial_u
    """
    hessian_x = hessian(fun)
    hessian_u = hessian(fun, argnums=1)
    hessian_x_u = jacobian(jax.grad(fun), argnums=1)

    def quadratizer(*args):
        return hessian_x(*args), hessian_u(*args), hessian_x_u(*args)

    return vectorize(quadratizer, argnums)


def rollout(dynamics, U, x0):
    """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0.

    Args:
      dynamics: a function f(x, u, t) to rollout.
      U: (T, m) np array for control sequence.
      x0: (n, ) np array for initial state.

    Returns:
       X: (T+1, n) state trajectory.
    """
    return _rollout(dynamics, U, x0)


def _rollout(dynamics, U, x0, *args):
    def dynamics_for_scan(x, ut):
        u, t = ut
        x_next = dynamics(x, u, t, *args)
        return x_next, x_next

    return jnp.vstack(
        (x0, lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))[1]))


def evaluate(cost, X, U, *args):
    """Evaluates cost(x, u, t) along a trajectory.

    Args:
      cost: cost_fn with signature cost(x, u, t, *args)
      X: (T, n) state trajectory.
      U: (T, m) control sequence.
      *args: args for cost_fn

    Returns:
      objectives: (T, ) array of objectives.
    """
    timesteps = jnp.arange(X.shape[0])
    return vectorize(cost)(X, U, timesteps, *args)


def objective(cost, dynamics, U, x0):
    """Evaluates total cost for a control sequence.

    Args:
      cost: cost_fn with signature cost(x, u, t)
      dynamics: dynamics_fn with signature dynamics(x, u, t)
      U: (T, m) control sequence.
      x0: (n, ) initial state.

    Returns:
      objectives: total objective summed across time.
    """
    cost_converted, cost_consts = custom_derivatives.closure_convert(
        cost, x0, U[0], 0)
    dynamics_converted, dynamics_consts = custom_derivatives.closure_convert(
        dynamics, x0, U[0], 0)
    return _objective(cost_converted, dynamics_converted, U, x0, cost_consts,
                      dynamics_consts)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _objective(cost, dynamics, U, x0, cost_args, dynamics_args):
    return jnp.sum(
        evaluate(cost, _rollout(dynamics, U, x0, *dynamics_args), pad(U),
                 *cost_args))


def _objective_fwd(cost, dynamics, U, x0, cost_args, dynamics_args):
    obj = _objective(cost, dynamics, U, x0, cost_args, dynamics_args)
    return (obj, (U, x0, cost_args, dynamics_args))


def _objective_bwd(cost, dynamics, res, g):
    return (g * grad_wrt_controls(cost, dynamics, *res),) + (None,) * 3


_objective.defvjp(_objective_fwd, _objective_bwd)


def adjoint(A, B, q, r):
    """Solve adjoint equations.

    Args:
        A: dynamics Jacobians with respect to state.
        B: dynamics Jacobians with respect to control.
        q: cost gradients with respect to state.
        r: cost gradients with respect to control.

    Returns:
        gradient, adjoints, final adjoint variable.

    Usage:
      q, r = linearize(cost)(X, pad(U), timesteps)
      A, B = linearize(dynamics)(X, pad(U), np.arange(T + 1))
      gradient, adjoints, _ = adjoint(A, B, q, r)
    """

    n = q.shape[1]
    T = q.shape[0] - 1
    m = r.shape[1]
    P = jnp.zeros((T, n))
    g = jnp.zeros((T, m))

    def body(p, t):  # backward recursion of Adjoint equations.
        g = r[t] + jnp.matmul(B[t].T, p)
        p = jnp.matmul(A[t].T, p) + q[t]
        return p, (p, g)

    p, (P, g) = lax.scan(body, q[T], jnp.arange(T - 1, -1, -1))
    return jnp.flipud(g), jnp.vstack((jnp.flipud(P[:T - 1]), q[T])), p


def grad_wrt_controls(cost, dynamics, U, x0, cost_args, dynamics_args):
    """Evaluates gradient at a control sequence.

    Args:
      cost: cost_fn
      dynamics: dynamics_fn
      U: (T, m) control sequence.
      x0: (n, ) initial state.
      cost_args: args passed to cost
      dynamics_args: args passed to dynamics.

    Returns:
      gradient (T, m) of total cost with respect to controls.
    """
    jacobians = linearize(dynamics)
    grad_cost = linearize(cost)

    X = _rollout(dynamics, U, x0, *dynamics_args)
    timesteps = jnp.arange(X.shape[0])
    A, B = jacobians(X, pad(U), timesteps, *dynamics_args)
    q, r = grad_cost(X, pad(U), timesteps, *cost_args)
    gradient, _, _ = adjoint(A, B, q, r)
    return gradient


def hvp(cost, dynamics, U, x0, V, cost_args, dynamics_args):
    """Evaluates hvp at a control sequence.

    Args:
      cost: cost_fn
      dynamics: dynamics_fn
      U: (T, m) control sequence.
      x0: (n, ) initial state.
      V: (T, m) vector in Hessian-vector product.
      cost_args: args passed to cost
      dynamics_args: args passed to dynamics.

    Returns:
      gradient (T, m) of total cost with respect to controls.
    """
    grad_fn = partial(grad_wrt_controls, cost, dynamics)
    return jax.jvp(lambda U1: grad_fn(U1, x0, cost_args, dynamics_args), (U,),
                   (V,))


@partial(jit, static_argnums=(0,))
def ddp_rollout(dynamics, X, U, K, k, alpha, *args):
    """Rollouts used in Differential Dynamic Programming.

    Args:
      dynamics: function with signature dynamics(x, u, t, *args).
      X: [T+1, n] current state trajectory.
      U: [T, m] current control sequence.
      K: [T, m, n] state feedback gains.
      k: [T, m] affine terms in state feedback.
      alpha: line search parameter.
      *args: passed to dynamics.

    Returns:
      Xnew, Unew: updated state trajectory and control sequence, via:

        del_u = alpha * k[t] + np.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        x = dynamics(Xnew[t], u, t)
    """
    n = X.shape[1]
    T, m = U.shape
    Xnew = jnp.zeros((T + 1, n))
    Unew = jnp.zeros((T, m))
    Xnew = Xnew.at[0].set(X[0])

    def body(t, inputs):
        Xnew, Unew = inputs
        del_u = alpha * k[t] + jnp.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        x = dynamics(Xnew[t], u, t, *args)
        Unew = Unew.at[t].set(u)
        Xnew = Xnew.at[t + 1].set(x)
        return Xnew, Unew

    return lax.fori_loop(0, T, body, (Xnew, Unew))


@partial(jit, static_argnums=(0, 1))
def line_search_ddp(cost,
                    dynamics,
                    X,
                    U,
                    K,
                    k,
                    obj,
                    cost_args=(),
                    dynamics_args=(),
                    alpha_0=1.0,
                    alpha_min=0.00005):
    """Performs line search with respect to DDP rollouts."""

    obj = jnp.where(jnp.isnan(obj), jnp.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, *margs: jnp.sum(costs(X, pad(U), *margs))

    def line_search(inputs):
        """Line search to find improved control sequence."""
        _, _, _, alpha = inputs
        Xnew, Unew = ddp_rollout(dynamics, X, U, K, k, alpha, *dynamics_args)
        obj_new = total_cost(Xnew, Unew, *cost_args)
        alpha = 0.5 * alpha
        obj_new = jnp.where(jnp.isnan(obj_new), obj, obj_new)

        # Only return new trajs if leads to a strict cost decrease
        X_return = jnp.where(obj_new < obj, Xnew, X)
        U_return = jnp.where(obj_new < obj, Unew, U)

        return X_return, U_return, jnp.minimum(obj_new, obj), alpha

    return lax.while_loop(
        lambda inputs: jnp.logical_and(inputs[2] >= obj, inputs[3] > alpha_min),
        line_search, (X, U, obj, alpha_0))


@jit
def project_psd_cone(Q, delta=0.0):
    """Projects to the cone of positive semi-definite matrices.

    Args:
      Q: [n, n] symmetric matrix.
      delta: minimum eigenvalue of the projection.

    Returns:
      [n, n] symmetric matrix projection of the input.
    """
    S, V = jnp.linalg.eigh(Q)
    S = jnp.maximum(S, delta)
    Q_plus = jnp.matmul(V, jnp.matmul(jnp.diag(S), V.T))
    return 0.5 * (Q_plus + Q_plus.T)


def ilqr(cost,
         dynamics,
         x0,
         U,
         maxiter=100,
         grad_norm_threshold=1e-4,
         make_psd=False,
         psd_delta=0.0,
         alpha_0=1.0,
         alpha_min=0.00005):
    """Iterative Linear Quadratic Regulator.

    Args:
      cost:      cost(x, u, t) returns scalar.
      dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
      x0: initial_state - 1D np array of shape (n, ).
      U: initial_controls - 2D np array of shape (T, m).
      maxiter: maximum iterations.
      grad_norm_threshold: tolerance for stopping optimization.
      make_psd: whether to zero negative eigenvalues after quadratization.
      psd_delta: The delta value to make the problem PSD. Specifically, it will
        ensure that d^2c/dx^2 and d^2c/du^2, i.e. the hessian of cost function
        with respect to state and control are always positive definite.
      alpha_0: initial line search value.
      alpha_min: minimum line search value.

    Returns:
      X: optimal state trajectory - nd array of shape (T+1, n).
      U: optimal control trajectory - nd array of shape (T, m).
      obj: final objective achieved.
      gradient: gradient at the solution returned.
      adjoints: associated adjoint variables.
      lqr: inputs to the final LQR solve.
      iteration: number of iterations upon convergence.
    """
    cost_fn, cost_args = custom_derivatives.closure_convert(cost, x0, U[0], 0)
    dynamics_fn, dynamics_args = custom_derivatives.closure_convert(
        dynamics, x0, U[0], 0)
    return ilqr_base(cost_fn, dynamics_fn, x0, U, tuple(cost_args),
                     tuple(dynamics_args), maxiter, grad_norm_threshold, make_psd,
                     psd_delta, alpha_0, alpha_min)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
@partial(jit, static_argnums=(0, 1))
def ilqr_base(cost, dynamics, x0, U, cost_args, dynamics_args, maxiter,
              grad_norm_threshold, make_psd, psd_delta, alpha_0, alpha_min):
    """ilqr implementation."""

    T, m = U.shape
    n = x0.shape[0]

    roll = partial(_rollout, dynamics)
    quadratizer = quadratize(cost)
    dynamics_jacobians = linearize(dynamics)
    cost_gradients = linearize(cost)
    evaluator = partial(evaluate, cost)
    psd = vmap(partial(project_psd_cone, delta=psd_delta))

    X = roll(U, x0, *dynamics_args)
    timesteps = jnp.arange(X.shape[0])
    obj = jnp.sum(evaluator(X, pad(U), *cost_args))

    def get_lqr_params(X, U):
        Q, R, M = quadratizer(X, pad(U), timesteps, *cost_args)

        Q = lax.cond(make_psd, Q, psd, Q, lambda x: x)
        R = lax.cond(make_psd, R, psd, R, lambda x: x)

        q, r = cost_gradients(X, pad(U), timesteps, *cost_args)
        A, B = dynamics_jacobians(X, pad(U), jnp.arange(T + 1), *dynamics_args)

        return (Q, q, R, r, M, A, B)

    c = jnp.zeros((T, n))  # assumes trajectory is always dynamically feasible.

    gradient = jnp.full((T, m), jnp.inf)
    adjoints = jnp.zeros((T, n))

    def body(inputs):
        """Solves LQR subproblem and returns updated trajectory."""
        X, U, obj, alpha, gradient, adjoints, lqr, iteration = inputs

        Q, q, R, r, M, A, B = lqr

        K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)
        X, U, obj, alpha = line_search_ddp(cost, dynamics, X, U, K, k, obj,
                                           cost_args, dynamics_args, alpha_0,
                                           alpha_min)
        gradient, adjoints, _ = adjoint(A, B, q, r)
        # print("Iteration=%d, Objective=%f, Alpha=%f, Grad-norm=%f\n" %
        #      (device_get(iteration), device_get(obj), device_get(alpha),
        #       device_get(np.linalg.norm(gradient))))

        lqr = get_lqr_params(X, U)
        iteration = iteration + 1
        return X, U, obj, alpha, gradient, adjoints, lqr, iteration

    def continuation_criterion(inputs):
        _, _, _, alpha, gradient, _, _, iteration = inputs
        grad_norm = jnp.linalg.norm(gradient)
        grad_norm = jnp.where(jnp.isnan(grad_norm), jnp.inf, grad_norm)

        return jnp.logical_and(
            iteration < maxiter,
            jnp.logical_and(grad_norm > grad_norm_threshold, alpha > alpha_min))

    lqr = get_lqr_params(X, U)
    X, U, obj, _, gradient, adjoints, lqr, it = lax.while_loop(
        continuation_criterion, body,
        (X, U, obj, alpha_0, gradient, adjoints, lqr, 0))

    return X, U, obj, gradient, adjoints, lqr, it


def _ilqr_fwd(cost, dynamics, *args):
    """Forward pass of custom vector-Jacobian product implementation."""
    ilqr_output = ilqr_base(cost, dynamics, *args)  # pylint: disable=no-value-for-parameter
    X, U, _, _, adjoints, lqr, _ = ilqr_output
    return ilqr_output, (args, X, U, adjoints, lqr)


def _ilqr_bwd(cost, dynamics, fwd_residuals, gX_gU_gNonDifferentiableOutputs):
    """Backward pass of custom vector-Jacobian product implementation."""
    # TODO(schmrlng): Add gradient of `obj` with respect to inputs.
    args, X, U, adjoints, lqr = fwd_residuals
    x0, _, cost_args, dynamics_args = args[:4]
    gX, gU = gX_gU_gNonDifferentiableOutputs[:2]

    _, _, _, _, _, A, B = lqr
    timesteps = jnp.arange(X.shape[0])

    quadratizer = quadratize(hamiltonian(cost, dynamics), argnums=4)
    Q, R, M = quadratizer(X, pad(U), timesteps, pad(adjoints), cost_args,
                          dynamics_args)

    c = jnp.zeros(A.shape[:2])
    K, k, _, _ = tvlqr(Q, gX, R, gU, M, A, B, c)
    _, dU = tvlqr_rollout(K, k, jnp.zeros_like(x0), A, B, c)

    vhp = vhp_params(cost)
    gradients = vhp(pad(dU), X, pad(U), A, B, *cost_args)[1]
    zeros_like_args = jax.tree_map(jnp.zeros_like, args)
    # TODO(schmrlng): Add gradients with respect to `cost_args` other than the
    # first, `x0`, and `dynamics_args`.
    return (zeros_like_args[:2] + ((gradients, *zeros_like_args[2][1:]),) +
            zeros_like_args[3:])


ilqr_base.defvjp(_ilqr_fwd, _ilqr_bwd)


def hamiltonian(cost, dynamics):
    """Returns function to evaluate associated Hamiltonian."""

    def fun(x, u, t, p, cost_args=(), dynamics_args=()):
        return cost(x, u, t, *cost_args) + jnp.dot(p,
                                                   dynamics(x, u, t, *dynamics_args))

    return fun


def vhp_params(cost):
    """Returns a function that evaluates vector hessian products.

    Args:
      cost: function with signature cost(x, u, t, *args).
    """
    hessian_u_params = jacobian(jax.grad(cost, argnums=1), argnums=3)
    hessian_x_params = jacobian(jax.grad(cost, argnums=0), argnums=3)

    def vhp(vector, X, U, A, B, *args):
        """Evaluates vector hessian products.

        Args:
          vector: input vector to compute vector hessian products.
          X: [T+1, n] state trajectory.
          U: [T, m] control trajectory.
          A: dynamics Jacobians wrt states.
          B: dynamics Jacobians wrt controls.
          *args: additional arguments passed to cost.

        Returns:
          Tuple
        """
        T = X.shape[0] - 1
        params = args[0]
        gradient = jax.tree_map(jnp.zeros_like, params)
        Cx = hessian_x_params(X[T], U[T], T, *args)
        contract = lambda x, y: jnp.tensordot(x, y, (-1, 0))

        def body(tt, inputs):
            """Accumulates vector hessian product over all time steps."""
            P, g = inputs
            t = T - 1 - tt
            Cx = hessian_x_params(X[t], U[t], t, *args)
            Cu = hessian_u_params(X[t], U[t], t, *args)
            w = jnp.matmul(B[t], vector[t])
            g = jax.tree_map(
                lambda P_, g_, Cu_: g_ + contract(vector[t], Cu_) + contract(w, P_),
                P, g, Cu)
            P = jax.tree_map(lambda P_, Cx_: contract(A[t].T, P_) + Cx_, P, Cx)
            return P, g

        return lax.fori_loop(0, T, body, (Cx, gradient))

    return vhp


# Constrained Iterative LQR
@partial(jit, static_argnums=(0, 1, 4, 5, 12,))
def constrained_ilqr(
        cost,
        dynamics,
        x0,
        U,
        equality_constraint=lambda x, u, t: jnp.empty(1),
        inequality_constraint=lambda x, u, t: jnp.empty(1),
        maxiter_al=5,
        maxiter_ilqr=100,
        grad_norm_threshold=1.0e-4,
        constraints_threshold=1.0e-2,
        penalty_init=1.0,
        penalty_update_rate=10.0,
        make_psd=True,
        psd_delta=0.0,
        alpha_0=1.0,
        alpha_min=0.00005):
    """Constrained Iterative Linear Quadratic Regulator.

    Args:
      cost:      cost(x, u, t) returns scalar.
      dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
      x0: initial_state - 1D np array of shape (n, ); should satisfy constraints
        at t == 0.

      U: initial_controls - 2D np array of shape (T, m); this input does not have
        to be initially feasible.
      equality_constraint: equality_constraint(x, u, t) == 0 returns
        (num_equality, ) nd array.
      inequality_constraint: inequality_constraint(x, u, t) <= 0 returns
        (num_inequality, ) nd array.
      maxiter_al: maximum number of outer-loop augmented Lagrangian dual and
        penalty updates.
      maxiter_ilqr: maximum iterations for iLQR.
      grad_norm_threshold: tolerance for stopping iLQR optimization
        before augmented Lagrangian update.
      constraints_threshold: tolerance for constraint violation (infinity norm).
      penalty_init: initial penalty value.
      penalty_update_rate: update rate for increasing penalty.
      make_psd: whether to zero negative eigenvalues after quadratization.
      psd_delta: The delta value to make the problem PSD. Specifically, it will
        ensure that d^2c/dx^2 and d^2c/du^2, i.e. the hessian of cost function
        with respect to state and control are always positive definite.
      alpha_0: initial line search value.
      alpha_min: minimum line search value.

    Returns:
      X: optimal state trajectory - nd array of shape (T+1, n).
      U: optimal control trajectory - nd array of shape (T, m).
      dual_equality: approximate dual (equality) trajectory - nd array of shape
        (T+1, num_equality).
      dual_inequality: approximate dual (inequality) trajectory nd array of shape
        (T+1, num_inequality).
      penalty: final penalty value.
      equality_constraints: final constraint (equality) violation trajectory - nd
        array of shape (T+1, num_equality).
      inequality_constraints: final constraint (inequality) violation trajectory -
        nd array of shape (T+1, num_inequality).
      max_constraint_violation: maximum equality or inequality violation.
      obj: final augmented Lagrangian objective achieved.
      gradient: gradient at the solution returned.
      iteration_ilqr: cumulative number of iLQR iterations for entire constrained
        solve upon convergence.
      iteration_al: number of augmented Lagrangian outer-loop iterations upon
        convergence.

    """

    # horizon
    horizon = len(U) + 1
    t_range = jnp.arange(horizon)

    # rollout
    X = rollout(dynamics, U, x0)

    # augmented Lagrangian methods
    def augmented_lagrangian(x, u, t, dual_equality, dual_inequality, penalty):
        # stage cost
        J = cost(x, u, t)

        # stage equality constraint
        equality = equality_constraint(x, u, t)

        # stage inequality constraint
        inequality = inequality_constraint(x, u, t)

        # active set
        active_set = jnp.invert(
            jnp.isclose(dual_inequality[t], 0.0) & (inequality < 0.0))

        # update cost
        # TODO(taylorhowell): Gauss-Newton approximation for constraints,
        # specifically in the Hessian of the objective
        J += dual_equality[t].T @ equality + 0.5 * penalty * equality.T @ equality
        J += dual_inequality[t].T @ inequality + 0.5 * penalty * inequality.T @ (
                active_set * inequality)

        return J

    def dual_update(constraint, dual, penalty):
        return dual + penalty * constraint

    def inequality_projection(dual):
        return jnp.maximum(dual, 0.0)

    # vectorize
    equality_constraint_mapped = vectorize(equality_constraint)
    inequality_constraint_mapped = vectorize(inequality_constraint)
    dual_update_mapped = vmap(dual_update, in_axes=(0, 0, None))

    # evaluate constraints
    U_pad = pad(U)
    equality_constraints = equality_constraint_mapped(X, U_pad, t_range)
    inequality_constraints = inequality_constraint_mapped(X, U_pad, t_range)

    # initialize dual variables
    dual_equality = jnp.zeros_like(equality_constraints)
    dual_inequality = jnp.zeros_like(inequality_constraints)

    # initialize penalty
    penalty = penalty_init

    def body(inputs):
        # unpack
        _, U, dual_equality, dual_inequality, penalty, equality_constraints, inequality_constraints, _, _, _, iteration_ilqr, iteration_al = inputs

        # augmented Lagrangian parameters
        al_args = {
            'dual_equality': dual_equality,
            'dual_inequality': dual_inequality,
            'penalty': penalty,
        }

        # solve iLQR problem
        X, U, obj, gradient, _, _, iteration = ilqr(
            partial(augmented_lagrangian, **al_args),
            dynamics,
            x0,
            U,
            grad_norm_threshold=grad_norm_threshold,
            make_psd=make_psd,
            psd_delta=psd_delta,
            alpha_0=alpha_0,
            alpha_min=alpha_min,
            maxiter=maxiter_ilqr)

        # evalute constraints
        U_pad = pad(U)

        equality_constraints = equality_constraint_mapped(X, U_pad, t_range)

        inequality_constraints = inequality_constraint_mapped(X, U_pad, t_range)
        inequality_constraints_projected = inequality_projection(
            inequality_constraints)

        max_constraint_violation = jnp.maximum(
            jnp.max(jnp.abs(equality_constraints)),
            jnp.max(inequality_constraints_projected))

        # augmented Lagrangian update
        dual_equality = dual_update_mapped(equality_constraints, dual_equality,
                                           penalty)

        dual_inequality = dual_update_mapped(inequality_constraints,
                                             dual_inequality, penalty)
        dual_inequality = inequality_projection(dual_inequality)

        penalty *= penalty_update_rate

        # increment
        iteration_ilqr += iteration
        iteration_al += 1

        return X, U, dual_equality, dual_inequality, penalty, equality_constraints, inequality_constraints, max_constraint_violation, obj, gradient, iteration_ilqr, iteration_al

    def continuation_criteria(inputs):
        # unpack
        dual_inequality = inputs[3]
        inequality_constraints = inputs[6]
        max_constraint_violation = inputs[7]
        iteration_al = inputs[11]
        max_complementary_slack = jnp.max(
            jnp.abs(inequality_constraints * dual_inequality))
        # check maximum constraint violation and augmented Lagrangian iterations
        return jnp.logical_and(iteration_al < maxiter_al,
                               jnp.logical_or(
                                   max_constraint_violation > constraints_threshold,
                                   max_complementary_slack > constraints_threshold))

    return lax.while_loop(
        continuation_criteria, body,
        (X, U, dual_equality, dual_inequality, penalty,
         equality_constraints, inequality_constraints, jnp.inf, jnp.inf,
         jnp.full(U.shape, jnp.inf), 0, 0))

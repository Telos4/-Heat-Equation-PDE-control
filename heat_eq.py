from __future__ import absolute_import, division, print_function
from fenics import *
from os.path import abspath, basename, dirname, join
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize

from collections import OrderedDict

# define a mesh
mesh = UnitIntervalMesh(100)

# Compile sub domains for boundaries
left = CompiledSubDomain("near(x[0], 0.)")
right = CompiledSubDomain("near(x[0], 1.)")

# Label boundaries, required for the objective
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
left.mark(boundary_parts, 0)  # boundary part for outside temperature
right.mark(boundary_parts, 1)  # boundary part where control is applied
ds = Measure("ds", subdomain_data=boundary_parts)

# Choose a time step size
delta_t = 5.0e-3

# define constants of the PDE
alpha = Constant(1.0)
beta = Constant(1.0)
gamma = Constant(1.0e6)

U = FunctionSpace(mesh, "Lagrange", 1)

# data from objective:
# min_(y,u)  \sigma_Q/2 \int_{0,T} \int_{\Omega} |y - y_Q|_L2 dx dt + \sigma_T/2 \int_{\Omega} |y - y_T| dx
#   + sigma_u/2 \int_{0,T} |u - u_ref|^2 dt
y_T = Function(U)
y_T.interpolate(Expression("0.0", degree=1))
y_Q = Function(U)
y_Q.interpolate(Expression("0.0", degree=1))
u_ref = 0.0

# weights for objective
sigma_T = 1.0
sigma_Q = 1.0
sigma_u = 0.1

# fenics output level
set_log_level(WARNING)


# set_log_active(False)

def solve_forward_split(y0, us, y_outs):
    """ Solve forward equation of split system """
    phi = TestFunction(U)
    # y_hat
    y_hat_k1 = TrialFunction(U)  # function for state at time k+1 (this is what we solve for)
    y_hat_k0 = Function(U)  # function for state at time k  (initial value)
    y_hat_k0.assign(y0)

    # y_tilde
    y_tilde_k1 = TrialFunction(U)  #
    y_tilde_k0 = Function(U)  #
    y_tilde_k0.interpolate(Expression("0.0", degree=1))  # initial value for forward solve

    y_out = Constant(1.0)
    u = Constant(1.0)

    # variational formulations
    lhs_hat = (y_hat_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(
        y_hat_k1)) * dx + gamma * phi * y_hat_k1 * ds
    rhs_hat = (y_hat_k0 / Constant(delta_t) * phi) * dx + gamma * y_out * phi * ds(0)

    lhs_tilde = (y_tilde_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(
        y_tilde_k1)) * dx + gamma * phi * y_tilde_k1 * ds
    rhs_tilde = (y_tilde_k0 / Constant(delta_t) * phi) * dx + gamma * u * phi * ds(1)

    # functions for storing the solution
    y_hat = Function(U, name="y_hat")
    y_tilde = Function(U, name="y_tilde")
    y = Function(U, name="y")

    i = 0
    N = len(us)

    # lists for storing the open loop
    y_hats = [Function(U, name="y_hat_" + str(j)) for j in xrange(0, N + 1)]
    y_tildes = [Function(U, name="y_tilde_" + str(j)) for j in xrange(0, N + 1)]
    ys = [Function(U, name="ys_" + str(j)) for j in xrange(0, N + 1)]

    y_hats[0].assign(y_hat_k0)
    y_tildes[0].assign(y_tilde_k0)
    ys[0].assign(y_hat_k0 + y_tilde_k0)

    while i < N:
        y_out.assign(y_outs[i])
        u.assign(us[i])

        solve(lhs_hat == rhs_hat, y_hat)
        solve(lhs_tilde == rhs_tilde, y_tilde)

        y_hat_k0.assign(y_hat)
        y_tilde_k0.assign(y_tilde)
        y.assign(y_hat + y_tilde)

        i += 1

        y_hats[i].assign(y_hat)
        y_tildes[i].assign(y_tilde)
        ys[i].assign(y)

    return ys, y_hats, y_tildes


def solve_adjoint_split(y_hats, y_tildes):
    y_hat_T = Function(U)
    y_hat_T.assign(y_T - y_hats[-1])
    y_hat_Q = Function(U)

    y_tilde_T = Function(U)
    y_tilde_T.assign(y_tildes[-1])
    y_tilde_Q = Function(U)

    phi = TestFunction(U)
    q_hat_k0 = Function(U)  # function for state at time k+1 (initial value)
    q_hat_k1 = TrialFunction(U)  # function for state at time k   (this is what we solve for)
    q_hat_k0.assign(Constant(sigma_T) * y_hat_T)  # initial value for adjoint

    q_tilde_k0 = Function(U)
    q_tilde_k1 = TrialFunction(U)
    q_tilde_k0.assign(Constant(-sigma_T) * y_tilde_T)  # initial value for adjoint

    # variational formulations
    lhs_hat = (q_hat_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(
        q_hat_k1)) * dx + gamma * phi * q_hat_k1 * ds
    rhs_hat = (q_hat_k0 / Constant(delta_t) * phi) * dx + Constant(sigma_Q) * y_hat_Q * phi * dx

    lhs_tilde = (q_tilde_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(
        q_tilde_k1)) * dx + gamma * phi * q_tilde_k1 * ds
    rhs_tilde = (q_tilde_k0 / Constant(delta_t) * phi) * dx - Constant(sigma_Q) * y_tilde_Q * phi * dx

    # functions for storing the solution
    q_hat = Function(U, name="q_hat")
    q_tilde = Function(U, name="q_tilde")
    q = Function(U, name="q")

    i = 0
    N = len(y_hats) - 1

    # lists for storing the open loop
    q_hats = [Function(U, name="q_hat_" + str(j)) for j in xrange(0, N + 1)]
    q_tildes = [Function(U, name="q_tilde_" + str(j)) for j in xrange(0, N + 1)]

    q_hats[0].assign(q_hat_k0)
    q_tildes[0].assign(q_tilde_k0)

    while i < N:
        # plot(q_hat)
        # plot(q_tilde)
        # plot(q)

        y_hat_Q.assign(y_Q - y_hats[-(1 + i)])  # take i-th value from behind
        y_tilde_Q.assign(y_tildes[-(1 + i)])

        solve(lhs_hat == rhs_hat, q_hat)
        solve(lhs_tilde == rhs_tilde, q_tilde)
        q.assign(q_hat + q_tilde)

        q_hat_k0.assign(q_hat)
        q_tilde_k0.assign(q_tilde)

        i += 1

        q_hats[i].assign(q_hat)
        q_tildes[i].assign(q_tilde)

    q_hats.reverse()
    q_tildes.reverse()

    return q_hats, q_tildes


def solve_forward(us, y_outs, record=False):
    """ The forward problem """
    ofile = File("results/y.pvd")

    # Define function space
    U = FunctionSpace(mesh, "Lagrange", 1)

    # Set up initial values
    y0 = Function(U, name="y0")
    y0 = interpolate(Expression("0.0", degree=1), U)

    # Define test and trial functions
    v = TestFunction(U)
    y = TrialFunction(U)
    u = Constant(1.0)
    y_out = Constant(1.0)

    # Define variational formulation
    # On domain:
    # part depending on solution at current time step
    a = (y / Constant(delta_t) * v + alpha * inner(grad(y), grad(v))) * dx + alpha * gamma / beta * y * v * ds
    # part depending on solution at previous time step
    f_y = (y0 / Constant(delta_t) * v) * dx

    # On boundary:
    # forcing due to control
    f_u = alpha * gamma / beta * u * v * ds(1)
    # forcing due to outside data
    f_y_out = alpha * gamma / beta * y_out * v * ds(0)

    # Prepare solution
    y = Function(U, name="y")

    i = 0

    ys = OrderedDict()
    y_omegas = OrderedDict()
    y_omegas[i] = Function(U, name="y_omega[0]")

    L = min(len(us), len(y_outs))

    while i < L:
        plot(y0)
        u.assign(us[i])
        y_out.assign(y_outs[i])

        solve(a == f_u + f_y + f_y_out, y)
        y0.assign(y)

        i += 1

    return y, ys, y_omegas


def compute_gradient_fd(y0, u_n, y_outs):
    # numerical approximation of the gradient using finite differences
    eps = 1.0e-7
    L = len(u_n)
    grad_f = np.zeros(L)

    for i in range(0, L):
        u_minus = np.copy(u_n)
        u_plus = np.copy(u_n)

        u_minus[i] -= eps
        u_plus[i] += eps

        ys, _, _ = solve_forward_split(y0, u_minus, y_outs)
        J_minus = eval_J(u_minus, ys)

        ys, _, _ = solve_forward_split(y0, u_plus, y_outs)
        J_plus = eval_J(u_plus, ys)

        # central difference quotient
        grad_f[i] = (J_plus - J_minus) / (2.0 * eps)

    return grad_f


def eval_J(u_n, ys):
    norm_y = 0.0
    norm_u = 0.0

    L = len(u_n)

    y_temp = Function(U)

    for i in range(0, L+1):
        y_temp.assign(ys[i] - y_Q)
        y_sum_temp = delta_t * norm(y_temp) ** 2
        norm_y += y_sum_temp
    # for i in range(0, L):
    #     y_temp.assign(ys[i] - y_Q)
    #     y_sum_temp = norm(y_temp) ** 2
    #     y_temp.assign(ys[i + 1] - y_Q)
    #     y_sum_temp += norm(y_temp) ** 2
    #     y_sum_temp *= delta_t * 0.5
    #     norm_y += y_sum_temp

    for i in range(0, L):
        u_sum_temp = (u_n[i] - u_ref) ** 2 * delta_t
        norm_u += u_sum_temp

    # final value
    y_temp.assign(ys[L] - y_T)
    norm_y_final = norm(y_temp)**2

    J = 0.5 * sigma_Q * norm_y + 0.5 * sigma_u * norm_u + 0.5 * sigma_T*norm_y_final

    return J


def compute_gradient_adj(p_hats, p_tildes, u):
    N = len(p_hats) - 1
    grad_adj = np.zeros(N)
    p = Function(U)

    for i in xrange(0, N):
        p.assign(p_hats[i] + p_tildes[i])
        grad_adj[i] = delta_t * (sigma_u * (u[i] - u_ref) - assemble(gamma * p * ds(1)))
#        grad_adj[i] = (sigma_u * (u[i] - u_ref) - assemble(gamma * p * ds(1)))

    return grad_adj


def func_J(u, y0, y_out):
    # forward solve
    ys, y_hats, y_tildes = solve_forward_split(y0, u, y_out)

    J = eval_J(u, ys)
    print("J = {}".format(J))

    return J


def grad_J(u, y0, y_out):
    grad_fd_approximation = False

    # forward solve
    ys, y_hats, y_tildes = solve_forward_split(y0, u, y_out)

    # adjoint solve
    p_hats, p_tildes = solve_adjoint_split(y_hats, y_tildes)

    # compute gradient
    grad_adj = compute_gradient_adj(p_hats, p_tildes, u)

    print("grad_adj = {}".format(grad_adj))
    print("|grad_adj| = {}".format(np.linalg.norm(grad_adj)))

    if grad_fd_approximation:
        grad_fd = compute_gradient_fd(y0, u, y_out)
        print("grad_fd  = {}".format(grad_fd))
        grad_error = np.linalg.norm(grad_adj - grad_fd) / np.linalg.norm(grad_fd)
        if grad_error > 1.0e-2:
            print("WARNING: gradient error = {}".format(grad_error))

    return grad_adj


def optimization(y0, u, y_out):
    u_opt = scipy.optimize.fmin_bfgs(func_J, u, grad_J, args=(y0, y_out))

    print("u = {}".format(u_opt))

    return u_opt


if __name__ == "__main__":
    L = 200
    N = 3

    print("time interval: {}".format(N * delta_t))

    y_outs = np.array([0.00 * sin(i) for i in range(0, L + N)])

    y0 = Function(U)
    y0.interpolate(Expression("0.1", degree=1))  # initial value

    u_guess = np.array([1.0 for i in range(0, N)])
    #u_guess = np.array([ 0.88017965,  0.73220114,  0.64458604,  0.57328225,  0.49232443])

    y = solve_forward_split(y0, u_guess, y_outs)

    for i in range(0, L):
        print("\n\ntime step {}".format(i))
        plot(y0)
        plt.show()

        # solve optimal control problem
        u_opt = optimization(y0, u_guess, y_outs[i:i + N])

        print("u_opt = {}".format(u_opt))

        # simulate next time step
        ys, y_hats, y_tildes = solve_forward_split(y0, u_opt[0:1], y_outs[i:i + 1])

        # update initial value for next time step
        y0.assign(ys[1])

        u_guess = np.roll(u_opt, -1)
        u_guess[-1] = u_guess[-2]


    pass

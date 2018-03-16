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
y_T.interpolate(Expression("1.0", degree=1))
y_Q = Function(U)
y_Q.interpolate(Expression("0.0", degree=1))
u_ref = 0.0

# weights for objective
sigma_T = 1.0
sigma_Q = 1.0
sigma_u = 1.0

# fenics output level
set_log_level(WARNING)

def solve_forward(y0, us, y_outs):
    """ Solve forward equation of split system """
    phi = TestFunction(U)
    y_k1 = TrialFunction(U)
    y_k0 = Function(U)
    y_k0.assign(y0)

    y_out = Constant(1.0)
    u = Constant(1.0)

    # variational formulation
    lhs = (y_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(
        y_k1)) * dx + gamma * phi * y_k1 * ds
    rhs = (y_k0 / Constant(delta_t) * phi) * dx + gamma * u * phi * ds(1) + gamma * y_out * phi * ds(0)

    # function for storing the solution
    y = Function(U, name="y")

    i = 0
    N = len(us)

    # list for storing the open loop
    ys = [Function(U, name="ys_" + str(j)) for j in xrange(0, N + 1)]
    ys[0].assign(y_k0)

    while i < N:
        y_out.assign(y_outs[i+1])
        u.assign(us[i])
        solve(lhs == rhs, y)
        y_k0.assign(y)

        i += 1

        ys[i].assign(y)

    return ys

def solve_adjoint(ys):
    ybar = Function(U)          # solution of forward equation
    phi = TestFunction(U)       # test function
    q_k0 = Function(U)                              # function for state at time k+1 (initial value)
    q_k1 = TrialFunction(U)                         # function for state at time k   (this is what we solve for)
    q_k0.assign(Constant(sigma_T) * (y_T - ys[-1])) # initial value for adjoint

    # variational formulation
    lhs = (q_k1 / Constant(delta_t) * phi) * dx + alpha * inner(grad(phi), grad(q_k1)) * dx + gamma * phi * q_k1 * ds
    rhs = (q_k0 / Constant(delta_t) * phi) * dx + Constant(sigma_Q) * (y_Q - ybar) * phi * dx

    # function for storing the solution
    q = Function(U, name="q")

    i = 0
    N = len(ys) - 1

    # list for storing the open loop
    qs = [Function(U, name="q_" + str(j)) for j in xrange(0, N+1)]
    qs[0].assign(q_k0)

    while i < N:
        ybar.assign(ys[-(1+i)])     # take i-th value from behind
        solve(lhs == rhs, q)
        q_k0.assign(q)
        i += 1
        qs[i].assign(q)

    qs.reverse()

    return qs


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

        ys = solve_forward(y0, u_minus, y_outs)
        J_minus = eval_J(u_minus, ys)

        ys = solve_forward(y0, u_plus, y_outs)
        J_plus = eval_J(u_plus, ys)

        # central difference quotient
        grad_f[i] = (J_plus - J_minus) / (2.0 * eps)

    return grad_f


def eval_J(u_n, ys):
    norm_y = 0.0
    norm_u = 0.0

    L = len(u_n)

    y_temp = Function(U)

    for i in range(0, L):
        y_temp.assign(ys[i + 1] - y_Q)
        norm_y += norm(y_temp)**2 * delta_t

    for i in range(0, L):
        norm_u += (u_n[i] - u_ref)**2 * delta_t

    # final value
    y_temp.assign(ys[L] - y_T)
    norm_y_final = norm(y_temp)**2

    J = 0.5 * sigma_Q * norm_y + 0.5 * sigma_u * norm_u + 0.5 * sigma_T*norm_y_final

    return J


def compute_gradient_adj(ps, u):
    N = len(ps) - 1
    grad_adj = np.zeros(N)
    p = Function(U)

    for i in xrange(0, N):
        p.assign(ps[i])
        grad_adj[i] = delta_t * (sigma_u * (u[i] - u_ref) - assemble(gamma * p * ds(1)))

    return grad_adj


def func_J(u, y0, y_out):
    # forward solve
    ys = solve_forward(y0, u, y_out)
    J = eval_J(u, ys)

    return J


def grad_J(u, y0, y_out):
    grad_fd_approximation = True

    # forward solve
    ys = solve_forward(y0, u, y_out)

    # adjoint solve
    ps = solve_adjoint(ys)

    # compute gradient
    grad_adj = compute_gradient_adj(ps, u)

    if grad_fd_approximation:
        print("grad_adj = {}".format(grad_adj))
        grad_fd = compute_gradient_fd(y0, u, y_out)
        print("grad_fd  = {}".format(grad_fd))
        grad_error = np.linalg.norm(grad_adj - grad_fd) / np.linalg.norm(grad_fd)
        if grad_error > 1.0e-2:
            print("WARNING: gradient error = {}".format(grad_error))
        print("|grad_adj| = {}".format(np.linalg.norm(grad_adj)))

    return grad_adj


def optimization(y0, u, y_out):
    res = scipy.optimize.minimize(func_J, u, method='L-BFGS-B', jac=grad_J, args=(y0, y_out))
    u_opt = res.x
    J = res.fun
    print("J     = {}\nu_opt = {}".format(J, u_opt))
    #u_opt = scipy.optimize.fmin_bfgs(func_J, u, grad_J, args=(y0, y_out))
    return u_opt


if __name__ == "__main__":
    L = 10
    N = 5

    print("time interval: {}".format(N * delta_t))

    y_outs = np.array([1.0 for i in range(0, L + N)])

    y0 = Function(U)
    y0.interpolate(Expression("1.0", degree=1))  # initial value

    u_guess = np.array([0.1 * (i) for i in range(0, N)])

    t0 = time.clock()
    for i in range(0, L):
        print("\n\ntime step {}".format(i))
        plot(y0)
        #plt.show()

        # solve optimal control problem
        u_opt = optimization(y0, u_guess, y_outs[i:i + N + 1])

        #print("u_opt = {}".format(u_opt))

        # simulate next time step
        ys = solve_forward(y0, u_opt[0:1], y_outs[i:i + 2])

        # update initial value for next time step
        y0.assign(ys[1])

        u_guess = np.roll(u_opt, -1)
        u_guess[-1] = u_guess[-2]

        pass
    t = time.clock() - t0
    print("elapsed time: {}".format(t))

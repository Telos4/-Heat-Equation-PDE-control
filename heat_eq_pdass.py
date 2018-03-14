from __future__ import absolute_import, division, print_function
from fenics import *
from os.path import abspath, basename, dirname, join
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from timeit import default_timer as timer

from collections import OrderedDict

# define a mesh
mesh = UnitIntervalMesh(50)
#mesh = UnitSquareMesh(10,10)

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
sigma_Q = 0.01
sigma_u = 0.01

# fenics output level
set_log_level(WARNING)

def optimality_system(y0, y_outs, ys_opt, ps_opt, u_opt):

    t0 = timer()
    N = len(y_outs)
    phi = TestFunction(U)
    ys = [TrialFunction(U) for i in range(0,N)]
    ps = [TrialFunction(U) for i in range(0,N)]

    a = {}
    b = {}
    c = {}
    d = {}
    e = {}

    f = {}
    g = {}
    h = {}
    j = {}
    for i in range(0,N-1):
        a[i] = ys[i+1]/Constant(delta_t) * phi * dx + alpha * inner(grad(ys[i+1]), grad(phi)) * dx \
               + gamma * phi * ys[i+1] * ds
        b[i] = -ys[i]/Constant(delta_t) * phi * dx
        c[i] = -gamma * Constant(1)/Constant(sigma_u) * ps[i+1] * ds(1)
        d[i] = gamma * phi * ds(1)
        e[i] = gamma * Constant(y_outs[i+1]) * phi * ds(0)

        f[i] = ps[i]/Constant(delta_t) * phi * dx + alpha * inner(grad(phi), grad(ps[i])) * dx \
               + gamma * phi * ps[i] * ds
        g[i] = -ps[i+1]/Constant(delta_t) * phi * dx
        h[i] = sigma_Q * ys[i] * phi * dx
        j[i] = sigma_Q * y_Q * phi * dx

    A = {}
    B = {}
    C = {}
    D = {}
    E = {}

    F = {}
    G = {}
    H = {}
    J = {}

    DC = {}
    for i in range(0,N-1):
        A[i] = assemble(a[i]).array()
        B[i] = assemble(b[i]).array()
        C[i] = assemble(c[i]).get_local()
        C[i].shape = (1,C[i].shape[0])
        D[i] = assemble(d[i]).get_local()
        D[i].shape = (D[i].shape[0],1)
        DC[i] = np.dot(D[i], C[i])
        E[i] = assemble(e[i]).get_local()

        F[i] = assemble(f[i]).array()
        G[i] = assemble(g[i]).array()
        H[i] = assemble(h[i]).array()
        J[i] = assemble(j[i]).get_local()


    n = A[0].shape[0]
    M = np.zeros((2*N*n, 2*N*n))
    rhs = np.zeros(2*N*n)

    # set matrix blocks for initial values of state equation
    M[0:n,0:n] = np.eye(n,n)
    rhs[0:n] = y0.vector().get_local()

    # set matrix blocks for initial values of adjoint equation
    M[N*n:(N+1)*n, (N-1)*n:N*n] = sigma_T * np.eye(n,n)
    M[N*n:(N+1)*n, N*n+(N-1)*n:2*N*n] = np.eye(n,n)
    rhs[N*n:(N+1)*n] = sigma_T * y_T.vector().get_local()

    for i in range(0,N-1):
        # matrix blocks for state equation
        M[n+i*n:n+(i+1)*n, i*n:(i+1)*n] = B[i]
        M[n+i*n:n+(i+1)*n, n+i*n:n+(i+1)*n] = A[i]
        M[n+i*n:n+(i+1)*n, N*n+i*n:N*n+(i+1)*n] = DC[i]

        # matrix blocks for adjoint equation
        M[(N+1)*n+i*n:(N+1)*n+(i+1)*n, n+i*n:n+(i+1)*n] = H[i]
        M[(N+1)*n+i*n:(N+1)*n+(i+1)*n, N*n+i*n:N*n+(i+1)*n] = F[i]
        M[(N+1)*n+i*n:(N+1)*n+(i+1)*n, (N+1)*n+i*n:(N+1)*n+(i+1)*n] = G[i]

        # right hand side of state equation
        rhs[n+i*n:n+(i+1)*n] = E[i]

        # right hand side of adjoint equation
        rhs[(N+1)*n+i*n:(N+1)*n+(i+1)*n] = J[i]
    t1 = timer()
    print("time for assembly in PDASS: {}".format(t1 - t0))

    # solve the linear system
    t0 = timer()
    v = np.linalg.solve(M, rhs)
    t1 = timer()
    print("time for linear solve in PDASS: {}".format(t1 - t0))
    #v_opt = np.concatenate([ys_opt[0].vector().get_local(), ys_opt[1].vector().get_local(), ys_opt[2].vector().get_local(), \
    #        ps_opt[0].vector().get_local(), ps_opt[1].vector().get_local(), ps_opt[2].vector().get_local()])

    ys = [Function(U) for i in range(0,N)]
    ps = [Function(U) for i in range(0,N)]
    u = np.zeros(N-1)

    for i in range(0,N):
        ys[i].vector()[:] = v[i*n:(i+1)*n]
        ps[i].vector()[:] = v[N*n+i*n:N*n+(i+1)*n]

    for i in range(0,N-1):
        u[i] = assemble(gamma/Constant(sigma_u) * ps[i]*ds(1))

    print("|u - u_opt| = {}".format(np.linalg.norm(u-u_opt)))
    return M, rhs, v
    pass


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
    grad_fd_approximation = False

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
    t0 = timer()
    res = scipy.optimize.minimize(func_J, u, method='L-BFGS-B', jac=grad_J, args=(y0, y_out), options={'gtol':1e-5})
    t1 = timer()
    print("time for gradient method: {}".format(t1 - t0))
    #res = scipy.optimize.minimize(func_J, u, method='CG', jac=grad_J, args=(y0, y_out))
    u_opt = res.x
    J = res.fun
    print("J     = {}\nu_opt = {}".format(J, u_opt))
    #u_opt = scipy.optimize.fmin_bfgs(func_J, u, grad_J, args=(y0, y_out))
    return u_opt


if __name__ == "__main__":

    L = 1
    N = 50

    print("time interval: {}".format(N * delta_t))

    y_outs = np.array([0.0 for i in range(0, L + N)])

    y0 = Function(U)
    y0.interpolate(Expression("1.0", degree=1))  # initial value



    u_guess = np.array([0.0 for i in range(0, N)])
    #u_guess = np.array([0.03720524, 0.0494554 ])
    #u_guess = np.array([0.49124591, 0.98660247])

    for i in range(0, L):
        print("\n\ntime step {}".format(i))
        plot(y0)
        #plt.show()

        # solve optimal control problem
        u_opt = optimization(y0, u_guess, y_outs[i:i + N + 1])

        print("u_opt = {}".format(u_opt))

        # simulate next time step
        #ys = solve_forward(y0, u_opt[0:1], y_outs[i:i + 2])
        ys_opt = solve_forward(y0, u_opt, y_outs)
        ps_opt = solve_adjoint(ys_opt)

        M, rhs, v = optimality_system(y0, y_outs, ys_opt, ps_opt, u_opt)
        # update initial value for next time step
        #y0.assign(ys[1])

        u_guess = np.roll(u_opt, -1)
        u_guess[-1] = u_guess[-2]

        pass

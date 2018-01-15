from __future__ import absolute_import, division, print_function
from fenics import *
from os.path import abspath, basename, dirname, join
import numpy as np

from collections import OrderedDict

# define a mesh
mesh = UnitIntervalMesh(100)

# Compile sub domains for boundaries
left = CompiledSubDomain("near(x[0], 0.)")
right = CompiledSubDomain("near(x[0], 1.)")

# Label boundaries, required for the objective
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
left.mark(boundary_parts, 0)    # boundary part where control is applied
right.mark(boundary_parts, 1)   # boundary part for outside temperature
ds = Measure("ds", subdomain_data=boundary_parts)

# Choose a time step size
k = Constant(1e-2)

# MPC horizon length
N = 10

# define constants
alpha = Constant(0.1)
beta = Constant(1.0)
gamma = Constant(1.0e6)

U = FunctionSpace(mesh, "Lagrange", 1)

def solve_forward_split(us,y_outs):
    """ Solve forward equation of split system """

    phi = TestFunction(U)
    # y_hat
    y_hat_k1 = TrialFunction(U)     # function for state at time k+1 (this is what we solve for)
    y_hat_k0 = Function(U)          # function for state at time k  (initial value)
    y_hat_k0.interpolate(Expression("0.5", degree=1))  # initial value for forward solve

    # y_tilde
    y_tilde_k1 = TrialFunction(U)   #
    y_tilde_k0 = Function(U)        #
    y_tilde_k0.interpolate(Expression("0.0", degree=1))  # initial value for forward solve

    y_out = Constant(1.0)
    u = Constant(1.0)

    # variational formulations
    lhs_hat = (y_hat_k1 / k * phi) * dx + alpha * inner(grad(phi), grad(y_hat_k1)) * dx + gamma * phi * y_hat_k1 * ds
    rhs_hat = (y_hat_k0 / k * phi) * dx + gamma * y_out * phi * ds(0)

    lhs_tilde = (y_tilde_k1 / k * phi) * dx + alpha * inner(grad(phi), grad(y_tilde_k1)) * dx + gamma * phi * y_tilde_k1 * ds
    rhs_tilde = (y_tilde_k0 / k * phi) * dx + gamma * u * phi * ds(1)

    # functions for storing the solution
    y_hat   = Function(U, name="y_hat")
    y_tilde = Function(U, name="y_tilde")
    y = Function(U, name="y")

    i = 0

    # lists for storing the open loop
    y_hats = [Function(U, name="y_hat_" + str(j)) for j in xrange(0,L+1)]
    y_tildes = [Function(U, name="y_tilde_" + str(j)) for j in xrange(0,L+1)]

    y_hats[0].assign(y_hat_k0)
    y_tildes[0].assign(y_tilde_k0)

    while i < L:
        plot(y_hat)
        plot(y_tilde)

        y_out.assign(y_outs[i])
        u.assign(us[i])

        solve(lhs_hat == rhs_hat, y_hat)
        solve(lhs_tilde == rhs_tilde, y_tilde)
        y.assign(y_hat + y_tilde)

        y_hat_k0.assign(y_hat)
        y_tilde_k0.assign(y_tilde)

        i += 1

        y_hats[i].assign(y_hat)
        y_tildes[i].assign(y_tilde)

    return y, y_hats, y_tildes

def solve_adjoint_split(y_hats, y_tildes):

    y_T = Function(U)
    y_T.interpolate(Expression("0.5", degree=1))
    sigma_T = Constant(1.0)
    y_Q = Function(U)
    y_Q.interpolate(Expression("0.5", degree=1))
    sigma_Q = Constant(1.0)

    y_hat_T = Function(U)
    y_hat_T.assign(y_T - y_hats[-1])
    y_hat_Q = Function(U)

    phi = TestFunction(U)
    q_hat_k0 = Function(U)          # function for state at time k+1 (initial value)
    q_hat_k1 = TrialFunction(U)     # function for state at time k   (this is what we solve for)
    q_hat_k0.assign(sigma_T * y_hat_T)     # initial value for adjoint

    # y_tilde
    #y_tilde_k1 = TrialFunction(U)  #
    #y_tilde_k0 = Function(U)  #
    #y_tilde_k0.interpolate(Expression("0.0", degree=1))  # initial value for forward solve

    #y_out = Constant(1.0)
    #u = Constant(1.0)

    # variational formulations
    lhs_hat = (q_hat_k1 / k * phi) * dx + alpha * inner(grad(phi), grad(q_hat_k1)) * dx + gamma * phi * q_hat_k1 * ds
    rhs_hat = (q_hat_k0 / k * phi) * dx + sigma_Q * y_hat_Q * phi * dx

    #lhs_tilde = (y_tilde_k1 / k * phi) * dx + alpha * inner(grad(phi),
    #                                                        grad(y_tilde_k1)) * dx + gamma * phi * y_tilde_k1 * ds
    #rhs_tilde = (y_tilde_k0 / k * phi) * dx + gamma * u * phi * ds(1)

    # functions for storing the solution
    q_hat = Function(U, name="q_hat")
    #y_tilde = Function(U, name="y_tilde")
    #y = Function(U, name="y")

    i = 0

    # lists for storing the open loop
    q_hats = [Function(U, name="q_hat_" + str(j)) for j in xrange(0, L + 1)]
    #y_tildes = [Function(U, name="y_tilde_" + str(j)) for j in xrange(0, L + 1)]

    q_hats[0].assign(q_hat_k0)
    #y_tildes[0].assign(y_tilde_k0)

    while i < L:
        plot(q_hat)
        #plot(y_tilde)

        #y_out.assign(y_outs[i])
        #u.assign(us[i])
        y_hat_Q.assign(y_Q - y_hats[-(1+i)]) # take i-th value from behind

        solve(lhs_hat == rhs_hat, q_hat)
        #solve(lhs_tilde == rhs_tilde, y_tilde)
        #y.assign(y_hat + y_tilde)

        q_hat_k0.assign(q_hat)
        #y_tilde_k0.assign(y_tilde)

        i += 1

        q_hats[i].assign(q_hat)
        #y_tildes[i].assign(y_tilde)

    return q_hats #y, y_hats, y_tildes


def solve_forward(us, y_outs, record=False):
    """ The forward problem """
    ofile = File("results/y.pvd")

    # Define function space
    U = FunctionSpace(mesh, "Lagrange", 1)

    # Set up initial values
    y0 = Function(U, name="y0")
    y0 = interpolate(Expression("0.5", degree=1), U)

    # Define test and trial functions
    v = TestFunction(U)
    y = TrialFunction(U)
    u = Constant(1.0)
    y_out = Constant(1.0)

    # Define variational formulation
    # On domain:
    # part depending on solution at current time step
    a = (y / k * v + alpha * inner(grad(y), grad(v))) * dx + alpha * gamma / beta * y * v * ds
    # part depending on solution at previous time step
    f_y = (y0 / k * v) * dx

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

    #J = 0
    #for i in range(1, len(Jlist)):
    #    J += 0.5 * (Jlist[i - 1] + Jlist[i]) * float(k) #+ us[i-1]**2

    #dJdu = compute_gradient(J, control)

    return y, ys, y_omegas

    # def open_loop_solve_adjoint(self, pT):
    #     # solve the adjoint PDE
    #     N = self.N
    #     h = Constant(self.dt)
    #     gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
    #     alpha = Constant(self.alpha)
    #
    #     # set initial value
    #     self.p_ol[0].assign(pT)
    #
    #     for k in range(0, N):
    #         #print("k = %i" % k)
    #         v = TestFunction(self.S)
    #         a = ((self.p_ol[k+1] - self.p_ol[k] - h * (self.y_ol[N-k] - self.y_omega)) * v) * dx
    #         a += h * alpha * inner(grad(self.p_ol[k+1]), grad(v)) * dx
    #         for i in range(1, 5):
    #             a += h * alpha * Constant(gamma[i - 1]) * self.p_ol[k+1] * v * ds(i)
    #
    #         heat_eq_adj_problem = NonlinearVariationalProblem(a, self.p_ol[k+1])
    #
    #         heat_eq_adj_solver = NonlinearVariationalSolver(
    #         heat_eq_adj_problem,
    #             solver_parameters=self.heat_eq_solver_parameters)
    #
    #         heat_eq_adj_solver.solve()
    #
    #     self.p_ol.reverse()
    #
    #     for k in range(0,N+1):
    #         self.outfile_p.write(self.p_ol[k])
    #
    # def compute_gradient(self, u_n):
    #     r_n = np.array([0.0 for i in range(0,self.N)])
    #     for i in range(0, self.N):
    #         p = self.p_ol[i].at([0.5, 0.0])
    #         #print("p[{}] = {}".format(i,p))
    #         #for j in range(0,10):
    #         #    print("p[{}] = {}".format(0.1*j,self.p_ol[i].at([0.1*j, 0.0])))
    #         r_n[i] = - (self.lmda * u_n[i] + self.gamma_c * self.alpha * p) # use integral here?
    #
    #     return r_n

def compute_gradient_fd(self, y0, u_n):
    # numerical approximation of the gradient using finite differences
    eps = 1.0e-3
    grad_f = np.zeros(self.N)

    for i in range(0,self.N):
        u_minus = np.copy(u_n)
        u_plus  = np.copy(u_n)

        u_minus[i] -= eps
        u_plus[i]  += eps

        self.open_loop_solve(y0, u_minus)
        J_minus = self.eval_J(u_minus)

        self.open_loop_solve(y0, u_plus)
        J_plus = self.eval_J(u_plus)

        # central difference quotient
        grad_f[i] = (J_plus - J_minus)/(2.0*eps)

    return grad_f


def eval_J(self, u_n):
    norm_y = 0.0
    norm_u = 0.0

    for i in range(0,self.N):
        y_temp = norm(self.y_ol[i] - self.y_omega)**2
        u_temp = u_n[i]**2

        norm_y += y_temp
        norm_u += u_temp

    # final value
    y_temp = norm(self.y_ol[self.N] - self.y_omega) ** 2
    norm_y += y_temp

    J = 0.5*norm_y + self.lmda*0.5*norm_u

    return J



if __name__ == "__main__":
    L = 10

    #us = np.array([0.5 - 1.0/3.0 * sin(i/10.0) for i in range(0,L)])
    #y_outs = np.array([0.5 + 1.0/3.0 * sin(i/10.0) for i in range(0,L)])
    us = np.array([0.5 for i in range(0, L)])
    y_outs = np.array([0.5 for i in range(0, L)])

    y_split, y_hats, y_tildes = solve_forward_split(us, y_outs)

    solve_adjoint_split(y_hats, y_tildes)

    #y = solve_forward(us, y_outs)


    #t = y_split.vector().array() - y.vector().array()

    print("error = {}".format(np.linalg.norm(t)))

    pass

    # cwd = abspath(dirname(__file__))
    # data_dir = join(cwd, "data")
    #
    # mesh = UnitSquareMesh(10,10)
    #
    #
    # heateq.dt = 0.005
    #
    # S = heateq.S
    #
    # us = []
    # ue = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 25.0, 25.0, 20.0, 20.0, 20.0]
    # for i in range(0,heateq.N):
    #     if i < len(ue):
    #         us.append(ue[i])
    #     else:
    #         us.append(0.0)
    # y0 = Function(S)
    # y0.assign(0.0)
    #
    # u_n = np.array([0.0 for i in range(0, heateq.N)])
    # u_n[0] = 1.63
    # u_n[1] = 1.12
    #
    # for i in range(0,300):
    #     # 1. solve PDE
    #     heateq.open_loop_solve(y0, u_n)
    #
    #     J_n = heateq.eval_J(u_n)
    #     print("J(y,u) = {}".format(J_n))
    #
    #     # 2. solve Adjoint
    #     pT = Function(S)
    #     pT.interpolate(heateq.y_ol[heateq.N] - heateq.y_omega)
    #
    #     heateq.open_loop_solve_adjoint(pT)
    #
    #     # 3. compute descent direction
    #     r_n = heateq.compute_gradient(u_n)
    #
    #     # 3.1 compute descrent direction using finite differences
    #     grad_f = heateq.compute_gradient_fd(y0, u_n)
    #
    #     print("grad_f = {}".format(grad_f))
    #     print("r_n    = {}".format(r_n))
    #
    #     #u_n = u_n + 0.1 * r_n
    #     u_n = u_n - 0.1 * grad_f
    #
    #     print("u_n" + str(u_n))
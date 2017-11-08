from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join
import numpy as np


class HeatEq(object):
    def __init__(self, mesh):
        self.verbose = False
        self.mesh = mesh
        self.dt = 0.1
        #self.alpha = 0.257
        self.alpha = 1.0
        self.u = 25.0
        self.gamma_c = 1.0e3
        self.gamma_i = 0.0
        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.y_omega = Function(self.S)
        self.y_omega.assign(1.0)
        self.lmda = 1.0e-3

        self.heat_eq_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "hypre",
        }

        self.heat_eq_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "ksp_atol": 1e-10,
            "pc_type": "lu",
        }

        if self.verbose:
            self.heat_eq_solver_parameters["snes_monitor"] = True
            self.heat_eq_solver_parameters["ksp_converged_reason"] = True

        self.v = TestFunction(self.S)

        self.N = 2

        self.y_ol = []
        for i in range(0, self.N + 1):
            self.y_ol.append(Function(self.S))
            self.y_ol[i].rename("y_ol")

        self.p_ol = []
        for i in range(0, self.N + 1):
            self.p_ol.append(Function(self.S))
            self.p_ol[i].rename("p_ol")

        self.outfile_y = File(join(data_dir, "../", "results/", "y_ol.pvd"))
        self.outfile_p = File(join(data_dir, "../", "results/", "p_ol.pvd"))

    def open_loop_solve(self, y0, us):
        # given y(0), u(0), ..., u(N-1) solve the PDE and return the sequence of y(0), ..., y(N)
        N = self.N
        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        alpha = Constant(self.alpha)

        # set initial value
        self.y_ol[0].assign(y0)

        y_out = 20.0

        for k in range(0, N):
            v = TestFunction(self.S)
            z = [y_out, y_out, us[k], y_out]

            a = (self.y_ol[k+1] * v + h * alpha * inner(grad(self.y_ol[k+1]), grad(v))) * dx
            for i in range(1, 5):
                a += h * alpha * Constant(gamma[i - 1]) * self.y_ol[k+1] * v * ds(i)
            a -= inner(self.y_ol[k], v) * dx
            for i in range(1, 5):
                a -= h * alpha * Constant(gamma[i - 1]) * Constant(z[i - 1]) * v * ds(i)

            heat_eq_problem = NonlinearVariationalProblem(a, self.y_ol[k+1])

            heat_eq_solver = NonlinearVariationalSolver(
                heat_eq_problem,
                solver_parameters=self.heat_eq_solver_parameters)

            heat_eq_solver.solve()

        for k in range(0,N+1):
            self.outfile_y.write(self.y_ol[k])

    def open_loop_solve_adjoint(self, pT):
        # solve the adjoint PDE
        N = self.N
        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        alpha = Constant(self.alpha)

        # set initial value
        self.p_ol[0].assign(pT)

        for k in range(0, N):
            #print("k = %i" % k)
            v = TestFunction(self.S)
            a = ((self.p_ol[k+1] - self.p_ol[k] - h * (self.y_ol[N-k] - self.y_omega)) * v) * dx
            a += h * alpha * inner(grad(self.p_ol[k+1]), grad(v)) * dx
            for i in range(1, 5):
                a += h * alpha * Constant(gamma[i - 1]) * self.p_ol[k+1] * v * ds(i)

            heat_eq_adj_problem = NonlinearVariationalProblem(a, self.p_ol[k+1])

            heat_eq_adj_solver = NonlinearVariationalSolver(
            heat_eq_adj_problem,
                solver_parameters=self.heat_eq_solver_parameters)

            heat_eq_adj_solver.solve()

        self.p_ol.reverse()

        for k in range(0,N+1):
            self.outfile_p.write(self.p_ol[k])

    def compute_gradient(self, u_n):
        r_n = np.array([0.0 for i in range(0,self.N)])
        for i in range(0, self.N):
            p = self.p_ol[i].at([0.5, 0.0])
            #print("p[{}] = {}".format(i,p))
            #for j in range(0,10):
            #    print("p[{}] = {}".format(0.1*j,self.p_ol[i].at([0.1*j, 0.0])))
            r_n[i] = - (self.lmda * u_n[i] + self.gamma_c * self.alpha * p) # use integral here?

        return r_n

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
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(10,10)

    heateq = HeatEq(mesh)

    heateq.dt = 0.005

    S = heateq.S

    us = []
    ue = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 25.0, 25.0, 20.0, 20.0, 20.0]
    for i in range(0,heateq.N):
        if i < len(ue):
            us.append(ue[i])
        else:
            us.append(0.0)
    y0 = Function(S)
    y0.assign(0.0)

    u_n = np.array([0.0 for i in range(0, heateq.N)])
    u_n[0] = 1.63
    u_n[1] = 1.12

    for i in range(0,300):
        # 1. solve PDE
        heateq.open_loop_solve(y0, u_n)

        J_n = heateq.eval_J(u_n)
        print("J(y,u) = {}".format(J_n))

        # 2. solve Adjoint
        pT = Function(S)
        pT.interpolate(heateq.y_ol[heateq.N] - heateq.y_omega)

        heateq.open_loop_solve_adjoint(pT)

        # 3. compute descent direction
        r_n = heateq.compute_gradient(u_n)

        # 3.1 compute descrent direction using finite differences
        grad_f = heateq.compute_gradient_fd(y0, u_n)

        print("grad_f = {}".format(grad_f))
        print("r_n    = {}".format(r_n))

        #u_n = u_n + 0.1 * r_n
        u_n = u_n - 0.1 * grad_f

        print("u_n" + str(u_n))
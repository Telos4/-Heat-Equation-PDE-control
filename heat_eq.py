from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join
import numpy as np

import time

from firedrake_adjoint import *

class HeatEq(object):
    def __init__(self, mesh):
        self.verbose = False
        self.mesh = mesh
        self.dt = 0.1

        self.alpha = 1.0e3
        self.beta = 1.0e3

        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.y_omega = Function(self.S)
        self.y_omega.assign(0.0)
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
        self.outfile_u = File(join(data_dir, "../", "results/", "u_ol.pvd"))


    def open_loop_solve(self, y0, us,gradient=False, output=False):
        # given y(0), u(0), ..., u(N-1) solve the PDE and return the sequence of y(0), ..., y(N)
        N = 2           #self.N
        h = 0.1         #Constant(self.dt)
        alpha = 10.0e3  #Constant(self.alpha)
        beta = 10.0e3   #Constant(self.beta)

        # set initial value
        S = self.S
        y = Function(S)
        y_next = Function(S)
        v = TestFunction(S)
        ufs = [Function(S) for k in range(0,N)]

        y = y0.copy(deepcopy=True)

        #uss = [Constant(u) for u in us]


        for k in range(0, N):
            ufs[k].interpolate(Expression(us[k]))
            a = ((y_next - y)/h * v + inner(grad(y_next), grad(v))) * dx
            for i in range(1, 5):
                a -= (beta * ufs[k] - alpha * y_next) * v * ds(i)
            heat_eq_problem = NonlinearVariationalProblem(a, y_next)

            heat_eq_solver_parameters = {
                "mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "ksp_atol": 1e-10,
                "pc_type": "lu",
            }

            heat_eq_solver = NonlinearVariationalSolver(
                heat_eq_problem,
                solver_parameters=heat_eq_solver_parameters)

            heat_eq_solver.solve()

            if output == True:
                self.outfile_y.write(y)

            y.assign(y_next)

        if output == True:
            self.outfile_y.write(y)

        gradJ = np.zeros(N)
        if gradient==True:
            uc = [FunctionControl(ufs[k]) for k in range(0,N)]
            J = Functional(inner(y, y) * dx * dt[FINISH_TIME])
            Jhat = ReducedFunctional(J, uc)

            u_opt = minimize(Jhat, options = {'disp': True})

            print("u_opt = {}".format(u_opt))
            #dJdic = compute_gradient(J, Control(y))

            # param = Control(uss[1])
            # for (solution, variable) in compute_adjoint(J, forget=False):
            #     #outfile_adjoint.write(variable)
            #     #outfile = File(join(data_dir, "../", "results/", "test.pvd"))
            #     print("var: " + str(variable) + ", sol: " + str(solution))
            #     outfile = File(join(data_dir, "../", "results/", str(solution)+".pvd"))
            #
            #     outfile.write(solution)
            #
            #     print("norm: " + str(norm(solution-y_next)))


            # for k in range(0, N):
            #     forget = not (k < N-1)
            #     dJ = compute_gradient(J, Control(uss[k]), forget=forget)
            #     gradJ[k] = dJ._ufl_evaluate_scalar_()

        #for k in range(0,N+1):
        #    self.outfile_y.write(self.y_ol[k])
        return y, gradJ

    def open_loop_solve_adjoint(self, pT):
        # solve the adjoint PDE
        N = self.N
        h = Constant(self.dt)

        # set initial value
        self.p_ol[0].assign(pT)

        for k in range(0, N):
            v = TestFunction(self.S)
            a = ((self.p_ol[k+1] - self.p_ol[k])/h * v + inner(grad(self.p_ol[k+1]), grad(v))) * dx
            for i in range(1, 5):
                a += Constant(self.alpha) * self.p_ol[k+1] * v * ds(i)

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
            p1 = self.p_ol[i].at([0.5, 0.001])
            p2 = self.p_ol[i].at([0.001, 0.5])
            p3 = self.p_ol[i].at([0.5, 0.999])
            p4 = self.p_ol[i].at([0.999, 0.5])
            pm = max(np.abs(p1-p2),np.abs(p1-p3), np.abs(p1-p4), np.abs(p2-p3), np.abs(p2-p4), np.abs(p3-p4))
            #print("ps: ({},{},{},{}), max = {}".format(p1,p2,p3,p4,pm))
            r_n[i] = - (self.beta * p1 + self.lmda * u_n[i])

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

            y, _ = self.open_loop_solve(y0, u_minus)
            J_minus = self.eval_J(y,u_minus)

            y, _ = self.open_loop_solve(y0, u_plus)
            J_plus = self.eval_J(y,u_plus)

            # central difference quotient
            grad_f[i] = (J_plus - J_minus)/(2.0*eps)

        return grad_f


    def eval_J(self,y, u_n):
        norm_y = 0.0
        norm_u = 0.0

        for i in range(0,self.N):
            u_temp = u_n[i]**2
            norm_u += u_temp

        # final value
        y_temp = norm(self.y_ol[self.N] - self.y_omega) ** 2
        norm_y += y_temp

        J = 0.5*norm_y #+ self.lmda*0.5*norm_u

        J = norm(y-self.y_omega) ** 2

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
    y0.assign(1.0)

    u_n = np.array([0.0 for i in range(0, heateq.N)])

    parameters["adjoint"]["test_derivative"] = True

    for i in range(0,300):

        start = time.time()
        parameters["adjoint"]["stop_annotating"] = False
        # 1. solve PDE
        y, gradJ = heateq.open_loop_solve(y0, u_n, gradient=True, output=False)
        end = time.time()
        print("elapsed time: {}".format(end-start))

        J = heateq.eval_J(y, u_n)
        print("J(y,u) = {}".format(J))
        print("dJ        = {}".format(gradJ))

        parameters["adjoint"]["stop_annotating"] = True

        # 2. solve Adjoint
        #pT = Function(S)
        #pT.interpolate(heateq.y_ol[heateq.N] - heateq.y_omega)

        #heateq.open_loop_solve_adjoint(pT)

        # 3. compute descent direction
        #r_n = heateq.compute_gradient(u_n)

        # 3.1 compute descent direction using finite differences
        grad_f = heateq.compute_gradient_fd(y0, u_n)

        print("dJ (num.) = {}".format(grad_f))
        #print("r_n    = {}".format(r_n))

        #u_n = u_n + 0.1 * r_n
        u_n = u_n - 0.1 * grad_f

        print("u_n" + str(u_n))
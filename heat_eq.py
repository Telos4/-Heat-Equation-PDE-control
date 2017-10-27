from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join


class HeatEq(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = 0.257
        self.u = 25.0
        self.gamma_c = 1.0e3
        self.gamma_i = 0.0
        self.S = FunctionSpace(self.mesh, "CG", 1)

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
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "lu",
        }

        if self.verbose:
            self.heat_eq_solver_parameters["snes_monitor"] = True
            self.heat_eq_solver_parameters["ksp_converged_reason"] = True

        self.v = TestFunction(self.S)

        self.N = 15

        self.y_ol = []
        for i in range(0, self.N + 1):
            self.y_ol.append(Function(self.S))
            self.y_ol[i].rename("y_ol")

        self.outfile_y = File(join(data_dir, "../", "results/", "y_ol.pvd"))

    def open_loop_solve(self, y0, us):
        # given y(0), u(0), ..., u(N-1) solve the PDE and return the sequence of y(0), ..., y(N)
        N = self.N
        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        ac = Constant(self.k)

        # set initial value
        self.y_ol[0].assign(y0)

        self.outfile_y.write(self.y_ol[0])

        for k in range(0, N):
            v = TestFunction(self.S)
            u = [0.0, 0.0, us[k], 0.0]

            a = (self.y_ol[k+1] * v + h * ac * inner(grad(self.y_ol[k+1]), grad(v))) * dx
            for i in range(1, 5):
                a += h * ac * Constant(gamma[i - 1]) * self.y_ol[k+1] * v * ds(i)
            F = inner(self.y_ol[k], v) * dx
            for i in range(1, 5):
                F += h * ac * Constant(gamma[i - 1]) * Constant(u[i - 1]) * v * ds(i)

            heat_eq_problem = NonlinearVariationalProblem(a - F, self.y_ol[k+1])

            heat_eq_solver = NonlinearVariationalSolver(
                heat_eq_problem,
                solver_parameters=self.heat_eq_solver_parameters)

            heat_eq_solver.solve()

            self.outfile_y.write(self.y_ol[k+1])

    def open_loop_solve_adjoint(self):
        # solve the adjoint PDE
        N = self.N
        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        ac = Constant(self.k)

        # set initial value
        self.y_ol[0].assign(y0)

        self.outfile_y.write(self.y_ol[0])

        for k in range(0, N):
            v = TestFunction(self.S)
            u = [0.0, 0.0, us[k], 0.0]

            a = (self.y_ol[k+1] * v + h * ac * inner(grad(self.y_ol[k+1]), grad(v))) * dx
            for i in range(1, 5):
                a += h * ac * Constant(gamma[i - 1]) * self.y_ol[k+1] * v * ds(i)
            F = inner(self.y_ol[k], v) * dx
            for i in range(1, 5):
                F += h * ac * Constant(gamma[i - 1]) * Constant(u[i - 1]) * v * ds(i)

            heat_eq_problem = NonlinearVariationalProblem(a - F, self.y_ol[k+1])

            heat_eq_solver = NonlinearVariationalSolver(
                heat_eq_problem,
                solver_parameters=self.heat_eq_solver_parameters)

            heat_eq_solver.solve()

            self.outfile_y.write(self.y_ol[k+1])

    def setup_solver(self):
        v = TestFunction(self.S)
        self.y1 = Function(self.S)
        self.y1.rename("temperature")

        self.y0 = Function(self.S)

        #self.y1.assign(self.y0)

        self.outfile_y.write(self.y1)

        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        ac = self.k
        u = [Constant(0.0), Constant(0.0), self.u, Constant(0.0)]

        # ENERGY EQUATION

        # BACKWAD EULER

        self.a = (self.y1 * v + h * ac * inner(grad(self.y1), grad(v))) * dx
        for i in range(1,5):
            self.a += h * ac * Constant(gamma[i-1]) * self.y1 * v * ds(i)
        self.F = inner(self.y0, v) * dx
        for i in range(1, 5):
            self.F += h * ac * Constant(gamma[i-1]) * Constant(u[i-1]) * v * ds(i)

        self.heat_eq_problem = NonlinearVariationalProblem(self.a - self.F, self.y1)


        self.heat_eq_solver = NonlinearVariationalSolver(
            self.heat_eq_problem,
            solver_parameters=self.heat_eq_solver_parameters)

    def update_solver(self):
        v = TestFunction(self.S)
        h = Constant(self.dt)
        gamma = [self.gamma_i, self.gamma_i, self.gamma_c, self.gamma_i]
        ac = self.k
        u = [0.0, 0.0, self.u, 0.0]

        self.a = (inner(self.y1, v) + h * ac * inner(grad(self.y1), grad(v))) * dx
        for i in range(1, 5):
            self.a += h * Constant(gamma[i - 1]) * self.y1 * v * ds(i)
        self.F = inner(self.y0, v) * dx
        for i in range(1, 5):
            self.F += h * Constant(gamma[i - 1]) * Constant(u[i - 1]) * v * ds(i)

        self.heat_eq_problem = NonlinearVariationalProblem(self.a - self.F, self.y1)

        self.heat_eq_solver = NonlinearVariationalSolver(
            self.heat_eq_problem,
            solver_parameters=self.heat_eq_solver_parameters)

    def get_fs(self):
        return self.S

    def get_mass_matrix(self):
        s = TestFunction(self.S)
        t = TrialFunction(self.S)
        M = inner(t, s) * dx

        return M

    def get_jacobian_matrix(self):
        return self.heat_eq_problem.J

    def set_bcs(self, T_bcs):
        self.T_bcs = T_bcs

    def step(self):
        if self.verbose:
            print("HeatEq")

        self.heat_eq_solver.solve()

        self.outfile_y.write(self.y1)

        self.y0.assign(self.y1)

        return self.y1

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(10,10)

    heateq = HeatEq(mesh)

    heateq.dt = 0.005

    S = heateq.get_fs()

    us = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 25.0, 25.0, 20.0, 20.0, 20.0]
    y0 = Function(S)

    heateq.open_loop_solve(y0, us)


    pass

    # bcT = [DirichletBC(S, Constant(300.0), (1,3)),
    #        DirichletBC(S, Constant(250.0), (2, 4))]
    #
    # heateq.setup_solver()
    # outfile = File(join(data_dir, "../", "results/", "heateq.pvd"))
    #
    # step = 0
    # t = 0.0
    # t_end = 2.0
    # num_timesteps = int(t_end / heateq.dt)
    # output_frequency = 1
    #
    # print("Number of timesteps: {}".format(num_timesteps))
    # print("Output frequency: {}".format(output_frequency))
    # print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    # print("HeatEq DOFs: {}".format(heateq.y0.vector().size()))
    #
    # while (t <= t_end):
    #     t += heateq.dt
    #
    #     print("***********************")
    #     print("Timestep {}".format(t))
    #
    #     y1 = heateq.step()
    #
    #     print("")
    #
    #     #if step % output_frequency == 0:
    #     #    outfile.write(y1)
    #
    #     step += 1
    #
    #     if step % 100 == 0:
    #             heateq.u = -25.0
    #             #heateq.update_solver()
    #
    #     print("step: " + str(step))
    #     print("u = " + str(heateq.u))
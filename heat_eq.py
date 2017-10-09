from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join


class HeatEq(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = 0.0257
        self.gamma = 0.1
        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.heat_eq_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "hypre",
        }

        if self.verbose:
            self.heat_eq_solver_parameters["snes_monitor"] = True
            self.heat_eq_solver_parameters["ksp_converged_reason"] = True

    def _weak_form(self, T, s, k):
        F = k * inner(grad(T), grad(s)) * dx
        return F

    def setup_solver(self):
        v = TestFunction(self.S)
        self.y1 = Function(self.S)
        self.y1.rename("temperature")

        self.y0 = Function(self.S)

        self.y1.assign(self.y0)

        h = Constant(self.dt)
        gamma_c = self.gamma
        ac = self.k
        u = 0.1

        # ENERGY EQUATION

        # BACKWAD EULER

        #F = inner(self.y1 - self.y0, v) * dx + h * ac * inner(grad(self.y1), grad(v)) * dx + h * gamma_c * (self.y1 - u) * v * ds(1) \
            #- h * gamma_c * u * v * ds(1)
            #- inner(self.y0, v) * dx

        self.a = (inner(self.y1, v) + h * ac * inner(grad(self.y1), grad(v))) * dx
        for i in range(1,5):
            self.a -= h * gamma_c * self.y1 * v * ds(i)
        self.F = inner(self.y0, v) * dx
        for i in range(1, 5):
            self.F -= h * gamma_c * u * v * ds(i)
        #F1 = inner((self.T1 - self.T0), s) * dx + self.idt * self._weak_form(self.T1, s, self.k) \
        #        + self.idt * self.gamma_c * self.T1 * s * ds - self.idt * self.gamma_c * self.u * s * ds

        #self.energy_eq_problem = NonlinearVariationalProblem(
        #    F1, self.T1, self.T_bcs)

        #self.energy_eq_solver = NonlinearVariationalSolver(
        #    self.energy_eq_problem,
        #    solver_parameters=self.energy_eq_solver_parameters)



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
        #solve(self.a == self.F, self.y1)

        #self.y1.assign(self.y0)
        self.y0.assign(self.y1)

        return self.y1

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(25,25)

    heateq = HeatEq(mesh)

    heateq.dt = 0.005

    S = heateq.get_fs()

    bcT = [DirichletBC(S, Constant(300.0), (1,3)),
           DirichletBC(S, Constant(250.0), (2, 4))]

    #heateq.set_bcs(bcT)

    heateq.setup_solver()
    outfile = File(join(data_dir, "../", "results/", "heateq.pvd"))

    step = 0
    t = 0.0
    t_end = 10.0
    num_timesteps = int(t_end / heateq.dt)
    output_frequency = 1

    print("Number of timesteps: {}".format(num_timesteps))
    print("Output frequency: {}".format(output_frequency))
    print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    print("HeatEq DOFs: {}".format(heateq.y0.vector().size()))

    while (t <= t_end):
        t += heateq.dt

        print("***********************")
        print("Timestep {}".format(t))

        y1 = heateq.step()

        print("")

        if step % output_frequency == 0:
            outfile.write(y1)

        step += 1
from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join


class HeatEq(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = 0.0257
        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.energy_eq_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "hypre",
        }

        if self.verbose:
            self.energy_eq_solver_parameters["snes_monitor"] = True
            self.energy_eq_solver_parameters["ksp_converged_reason"] = True

    def _weak_form(self, T, s, k):
        F = k * inner(grad(T), grad(s)) * dx
        return F

    def setup_solver(self):
        s = TestFunction(self.S)
        self.T_1 = Function(self.S)
        self.T0 = Function(self.S)
        self.T1 = Function(self.S)
        self.T1.rename("temperature")

        self.idt = Constant(self.dt)
        self.T_1.assign(self.T0)

        # ENERGY EQUATION

        # BACKWAD EULER

        F1 = inner((self.T1 - self.T0), s) * dx + self.idt * self._weak_form(self.T1, s, self.k) \
                + self.idt * self.gamma_c * self.T1 * s * ds - self.idt * self.gamma_c * self.u * s * ds

        self.energy_eq_problem = NonlinearVariationalProblem(
            F1, self.T1, self.T_bcs)

        self.energy_eq_solver = NonlinearVariationalSolver(
            self.energy_eq_problem,
            solver_parameters=self.energy_eq_solver_parameters)

    def get_fs(self):
        return self.S

    def get_mass_matrix(self):
        s = TestFunction(self.S)
        t = TrialFunction(self.S)
        M = inner(t, s) * dx

        return M

    def get_jacobian_matrix(self):
        return self.energy_eq_problem.J

    def set_bcs(self, T_bcs):
        self.T_bcs = T_bcs

    def step(self):
        if self.verbose:
            print("HeatEq")

        self.energy_eq_solver.solve()
        self.T_1.assign(self.T0)
        self.T0.assign(self.T1)

        return self.T1

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = Mesh(data_dir + "/cyl.e")

    dm = mesh._plex
    from firedrake.mg.impl import filter_exterior_facet_labels
    for _ in range(2):
        dm.setRefinementUniform(True)
        dm = dm.refine()
        dm.removeLabel("interior_facets")
        dm.removeLabel("op2_core")
        dm.removeLabel("op2_non_core")
        dm.removeLabel("op2_exec_halo")
        dm.removeLabel("op2_non_exec_halo")
        filter_exterior_facet_labels(dm)

    mesh = Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(), distribute=False,
                reorder=True)

    heateq = HeatEq(mesh)

    heateq.dt = 0.005

    S = heateq.get_fs()

    bcT = [DirichletBC(S, Constant(300.0), (1,)),
           DirichletBC(S, Constant(250.0), (2, 4))]

    heateq.set_bcs(bcT)

    heateq.setup_solver(S)
    outfile = File(join(data_dir, "../", "results/", "heateq.pvd"))

    step = 0
    t = 0.0
    t_end = 3.0
    num_timesteps = int(t_end / heateq.dt)
    output_frequency = 1

    print("Number of timesteps: {}".format(num_timesteps))
    print("Output frequency: {}".format(output_frequency))
    print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    print("HeatEq DOFs: {}".format(heateq.T0.vector().size()))

    while (t <= t_end):
        t += heateq.dt

        print("***********************")
        print("Timestep {}".format(t))

        T1 = heateq.step()

        print("")

        if step % output_frequency == 0:
            outfile.write(T1)

        step += 1
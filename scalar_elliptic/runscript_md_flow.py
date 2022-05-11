""" Set up and solve mixed-dimensional linear problem.

To change linear solver, search for 'hazmath', look for comments.

To modify the problem setup, see comments towards the end of this file.

"""
from typing import Dict
import time
import scipy.sparse as sps
import numpy as np
import porepy as pp

import haznics

class EllipticProblem(pp.IncompressibleFlow):
    def __init__(self, params: Dict) -> None:

        super().__init__(params)
        self.fracture_perm = params.get("fracture_perm", 1)
        self.matrix_fracture_perm = params.get("matrix_fracture_perm", 1)

        self.solver_options = params.get("solver_options")

        self.use_mpfa = params.get("use_mpfa", True)

    def prepare_simulation(self) -> None:
        tic = time.time()
        print("Create grid")
        self.create_grid()
        print(f"Grid finished. Elapsed time: {time.time() - tic}")

        self._set_parameters()
        self._assign_variables()
        self._assign_discretizations()
        self._initial_condition()

        self._export()
        print("Discretize")
        tic = time.time()
        self._discretize()
        print(f"Discretization finished. Elapsed time: {time.time() - tic}")

    def create_grid(self) -> pp.GridBucket:

        grid_type = self.params["grid_type"]
        mesh_size = self.params["mesh_size"]

        if grid_type == "no_fracture":

            # Domain is unit square
            domain = pp.utils.default_domains.SquareDomain([1, 1])
            # Make an empty fracture network out of it to be compatible with more
            # advanced cases to come
            network = pp.FractureNetwork2d(domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_args)
            gb.compute_geometry()
            self.gb = gb
            self.box = domain
        elif grid_type == "single_fracture":
            # Domain is unit square
            domain = pp.utils.default_domains.SquareDomain([1, 1])

            p = np.array([[0.75, 0.25], [0.25, 0.75]])
            edge = np.array([[0], [1]])
            network = pp.FractureNetwork2d(pts=p, edges=edge, domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_args)
            gb.compute_geometry()
            self.gb = gb
            self.box = domain
        elif grid_type == "2d_benchmark_complex":

            network = pp.fracture_importer.network_2d_from_csv(
                "2d_benchmark_fracture_data.csv"
            )
            gb = network.mesh(mesh_size)

            self.gb = gb
            self.box = network.domain
        elif grid_type == "3d_regular":
            network = pp.fracture_importer.network_3d_from_csv(
                "3d_regular_fracture_data.csv"
            )
            gb = network.mesh(mesh_size)

            self.gb = gb
            self.box = network.domain
        elif grid_type == "3d_field":
            network = pp.fracture_importer.network_3d_from_csv(
                "3d_field_fracture_data.csv"
            )
            gb = network.mesh(mesh_size)

            self.gb = gb
            self.box = network.domain
        else:
            raise ValueError(f"Unknown grid type {grid_type}")

    def _set_parameters(self):
        super()._set_parameters()

        for g, d in self.gb:
            param = d[pp.PARAMETERS][self.parameter_key]
            param["max_memory"] = 5e7
        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            mg = data_edge["mortar_grid"]
            data_edge[pp.PARAMETERS][self.parameter_key][
                "normal_diffusivity"
            ] = self.matrix_fracture_perm * np.ones(mg.num_cells)

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Boundary conditions: Something non-boring.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        is_boundary = g.get_boundary_faces()
        bc = np.zeros(g.num_faces)
        bc_val = np.prod(g.face_centers[: self.gb.dim_max()], axis=0)
        bc[is_boundary] = bc_val[is_boundary]
        return bc

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Unitary permeability.

        Units: m^2
        """
        if g.dim == self.gb.dim_max():
            return np.ones(g.num_cells)
        else:
            return self.params["fracture_perm"] * np.ones(g.num_cells)

    def _assign_discretizations(self) -> None:
        """Define equations through discretizations.

        Assigns a Laplace/Darcy problem discretized using Mpfa on all subdomains with
        Neumann conditions on all internal boundaries. On edges of co-dimension one,
        interface fluxes are related to higher- and lower-dimensional pressures using
        the RobinCoupling.

        Gravity is included, but may be set to 0 through assignment of the vector_source
        parameter.
        """

        gb = self.gb
        dof_manager = pp.DofManager(gb)
        self.dof_manager = dof_manager
        self.assembler = pp.Assembler(self.gb, self.dof_manager)
        self._eq_manager = pp.ad.EquationManager(gb, dof_manager)

        grid_list = [g for g, _ in gb.nodes()]
        self.grid_list = grid_list
        if len(self.gb.grids_of_dimension(self.gb.dim_max())) != 1:
            raise NotImplementedError("This will require further work")
        edge_list = [e for e, d in gb.edges() if d["mortar_grid"].codim < 2]

        # Ad representation of discretizations
        if self.use_mpfa:
            flow_ad = pp.ad.MpfaAd(self.parameter_key, grid_list)
        else:
            flow_ad = pp.ad.TpfaAd(self.parameter_key, grid_list)
        div = pp.ad.Divergence(grids=grid_list)

        # Ad variables
        p = self._eq_manager.merge_variables([(g, self.variable) for g in grid_list])

        has_edge = len(edge_list) > 0

        mortar_proj = pp.ad.MortarProjections(
            edges=edge_list, grids=grid_list, gb=self.gb, nd=1
        )

        robin_ad = pp.ad.RobinCouplingAd(self.parameter_key, edge_list)
        if has_edge:
            mortar_flux = self._eq_manager.merge_variables(
                [(e, self.mortar_variable) for e in edge_list]
            )
        bc_val = pp.ad.BoundaryCondition(self.parameter_key, grid_list)

        # Ad equations
        flux = flow_ad.flux * p + flow_ad.bound_flux * bc_val
        if has_edge:
            flux += flow_ad.bound_flux * mortar_proj.mortar_to_primary_int * mortar_flux
        subdomain_flow_eq = div * flux
        if has_edge:
            subdomain_flow_eq -= mortar_proj.mortar_to_secondary_int * mortar_flux
        # Interface equation: \lambda = -\kappa (p_l - p_h)
        subdomain_flow_eq.set_name("flow on nodes")
        # Robin_ad.mortar_discr represents -\kappa. The involved term is
        # reconstruction of p_h on internal boundary, which has contributions
        # from cell center pressure, external boundary and interface flux
        # on internal boundaries (including those corresponding to "other"
        # fractures).
        if has_edge:
            p_primary = (
                flow_ad.bound_pressure_cell * p
                + flow_ad.bound_pressure_face
                * mortar_proj.mortar_to_primary_int
                * mortar_flux
                + flow_ad.bound_pressure_face * bc_val
            )

            # Project the two pressures to the interface and equate with \lambda
            interface_flow_eq = (
                robin_ad.mortar_discr
                * (
                    mortar_proj.primary_to_mortar_avg * p_primary
                    - mortar_proj.secondary_to_mortar_avg * p
                )
                + mortar_flux
            )
            interface_flow_eq.set_name("flow on interfaces")
        # Add to the equation list:
        self._eq_manager.equations.update(
            {
                "subdomain_flow": subdomain_flow_eq,
            }
        )
        if has_edge:
            self._eq_manager.equations.update(
                {
                    "interface_flow": interface_flow_eq,
                }
            )

    def _export(self):
        pass

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Use a direct solver for the linear system."""

        if self.solver_options["solver"] == "direct":
            A, b = self._eq_manager.assemble()

            tic = time.time()
            print(f"System size: {b.size}")
            x = sps.linalg.spsolve(A, b)
            print("Solved linear system in {} seconds".format(time.time() - tic))
            return x
        elif self.solver_options["solver"] == "hazmath":

            # The following lines give you the Dofs relating to pressure and
            # mortar variables.
            # Dofs relating to individual grids and interfaces can also be found
            # quite easily, let me know.

            dof_manager = self.dof_manager
            pressure_dofs = np.hstack(
                [
                    dof_manager.grid_and_variable_to_dofs(g, self.variable)
                    for g, _ in self.gb
                ]
            )
            mortar_dofs = np.hstack(
                [
                    dof_manager.grid_and_variable_to_dofs(e, self.mortar_variable)
                    for e, _ in self.gb.edges()
                ]
            )

            # This is where the hazmath magic should enter.
            #raise NotImplementedError("This is where the magic is missing")

            # assemble
            tic = time.time()
            A, b = self._eq_manager.assemble()
            print("Assemble linear system in {} seconds".format(time.time() - tic))

            # cast to hazmath
            A_haz = haznics.create_matrix(A.data, A.indices, A.indptr,  A.shape[1])
            b_haz = haznics.create_dvector(b)
            u_haz = haznics.dvector()
            haznics.dvec_alloc(b.shape[0],u_haz)

            pressure_dofs_haz = haznics.ivector()
            pressure_dofs32 = pressure_dofs.astype(np.int32)
            pressure_dofs_haz = haznics.create_ivector(pressure_dofs32)
            mortar_dofs_haz = haznics.ivector()
            mortar_dofs32 = mortar_dofs.astype(np.int32)
            mortar_dofs_haz = haznics.create_ivector(mortar_dofs32)

            # if needed, output matrix, right hand side, and dofs
            #haznics.dcsr_write_dcoo('A.dat', A_haz)
            #haznics.dvector_write('b.dat', b_haz)
            #haznics.iarray_print(pressure_dofs_haz.val, pressure_dofs_haz.row);
            #haznics.ivector_write('pressure_dofs.dat', pressure_dofs_haz)
            #haznics.ivector_write('mortar_dofs.dat', mortar_dofs_haz)

            # initialize parameters
            amgparam = haznics.AMG_param()
            itsparam = haznics.linear_itsolver_param()

            # set parameters
            params_amg = {
                'print_level': 3,
                'AMG_type': haznics.SA_AMG,       #haznics.UA_AMG
                'aggregation_type': haznics.VMB,  #haznics.VMB  haznics.HEC
                'cycle_type': haznics.V_CYCLE,
                #'coarse_dof': 100,
                'coarse_solver':haznics.SOLVER_UMFPACK,
                #'Schwarz_levels': 0,
            }
            haznics.param_amg_set_dict(params_amg, amgparam)

            params_its = {
                'linear_print_level': 3,
                'linear_itsolver_type': haznics.SOLVER_VFGMRES,
                'linear_precond_type': 21,
                'linear_maxit': 200,
                'linear_restart': 200,
                'linear_tol': 1e-6,
            }
            haznics.param_linear_solver_set_dict(params_its, itsparam)

            # if needed, print parameters
            haznics.param_amg_print(amgparam)
            haznics.param_linear_solver_print(itsparam)


            # solve
            tic = time.time()
            haznics.linear_solver_dcsr_krylov_md_scalar_elliptic(A_haz, b_haz, u_haz, itsparam, amgparam, pressure_dofs_haz, mortar_dofs_haz)
            print("Solved linear system in {} seconds".format(time.time() - tic))

            return u_haz.to_ndarray()


##################################
### DO MODIFICATIONS TO CHANGE THE PROBLEM
##################################
#
# The following updates are the most relevant:
#   1) Change permeability in fractures and on the fracture-matrix interface
#   2) Change geometry, including fracture network and mesh size
#   3) Change linear solver, currently set to


## CHANGE PERMEABILITY HERE
# Tuning of permeabilities, to check parameter robustness. The matrix permeability
# is always 1.
# Change this to alter the permeability in the fractures
fracture_permeability = 1.e0
#print(fracture_permeability)
# Change this to alter permeability between fractures and matrix
matrix_fracture_permeability = 1.e0
#print(matrix_fracture_permeability)

## CHANGE GRID HERE
# Implemented values for grid type:
#   1) 'no_fracture' - unit square in 2d
#   2) 'single_fracture' - single immersed fracture in 2d
#   3) '2d_benchmark_complex' - complex 2d case, 60-odd fractures.
#   4) '3d_regular' - 3d problem, Rubik's cube type geometry
#
# See below for switching
grid_type = "single_fracture"
#grid_type = "2d_benchmark_complex"
#grid_type = "3d_regular"
#grid_type = "3d_field"

# SET MESH SIZE
# If you tweak mesh_size_bound, it will adjust the mesh size at the
# domain boundaries. Changing mesh_size_frac will change the mesh size around
# the fractures.
# The if-else sets reasonable (?), values for each of the implemented cases
if grid_type == "no_fracture":
    # In this case, mesh_size_frac will not be used, but it is needed for
    # consistency
    mesh_args = {"mesh_size_bound": 0.1, "mesh_size_frac": 0.1}
elif grid_type == "single_fracture":
    mesh_args = {"mesh_size_bound": 0.5, "mesh_size_frac": 0.5}
elif grid_type == "2d_benchmark_complex":
    # Use 40 to get a rough mesh (similar to the coarse case set up previously)
    # 20 gives a mesh with reasonable cell geometries (in the eye norm)
    # 10 is a refined version again, this time with more standard refinement
    mesh_args = {"mesh_size_bound": 5, "mesh_size_frac": 5}
elif grid_type == "3d_regular":
    mesh_args = {"mesh_size_bound": 0.05, "mesh_size_frac": 0.05, "mesh_size_min": 0.05}
elif grid_type == "3d_field":
    mesh_args = {"mesh_size_bound": 400, "mesh_size_frac": 400, "mesh_size_min": 400}
#
## CHANGE SOLVER HERE
#solver = "direct"
# Uncomment the next line to activate hazmath. This will make the code crash
# at the place where the hazmath interface should be implemented.
solver = 'hazmath'
solver_options = {"solver": solver}


model_params = {
    "fracture_perm": fracture_permeability,
    "matrix_fracture_perm": matrix_fracture_permeability,
    "solver_options": solver_options,
    "grid_type": grid_type,
    "mesh_size": mesh_args,
}

model = EllipticProblem(params=model_params)

# The machinery for running simulations based on a model is rather general in scope
# (though not in functionality). There is a possibility to pass parameters to
# the runner, but we don't need that.
params_runner = {}

pp.run_stationary_model(model, params_runner)

# The lines below plots the pressure solution and get hold of some pressure values,
# should you be interested
gb = model.gb

if False:
    # Plot soluting using matplotlib. Will take some time for grids with many cells
    g = gb.grids_of_dimension(gb.dim_max())[0]
    p = gb.node_props(g, pp.STATE)[model.variable]
    pp.plot_grid(g, p)
if False:
    # Write solution to vtu, ready to be visualized with Paraview
    viz = pp.Exporter(gb, "solution.vtu")
    viz.write_vtu(model.variable)

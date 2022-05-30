"""Set up a simple 
"""
from typing import Dict, Optional
import time
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import numpy as np
import porepy as pp

# import haznics


class MechanicsProblem(pp.ContactMechanics):
    def __init__(self, params: Optional[Dict] = None):

        super().__init__(params)

        self._use_ad = True

    def create_grid(self) -> None:
        grid_type = self.params["grid_type"]
        mesh_size = self.params["mesh_size"]

        if grid_type == "2d_no_fracture":

            # Domain is unit square
            domain = pp.utils.default_domains.SquareDomain([1, 1])
            # Make an empty fracture network out of it to be compatible with more
            # advanced cases to come
            network = pp.FractureNetwork2d(domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_size)
            gb.compute_geometry()
            self.box = domain
        elif grid_type == "2d_single_fracture":
            # Domain is unit square
            domain = pp.utils.default_domains.SquareDomain([1, 1])

            p = np.array([[0.75, 0.25], [0.25, 0.75]])
            edge = np.array([[0], [1]])
            network = pp.FractureNetwork2d(pts=p, edges=edge, domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_size)
            gb.compute_geometry()
            self.box = domain

        if grid_type == "3d_no_fracture":

            # Domain is unit square
            domain = pp.utils.default_domains.CubeDomain([1, 1, 1])
            # Make an empty fracture network out of it to be compatible with more
            # advanced cases to come
            network = pp.FractureNetwork3d(domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_size)
            gb.compute_geometry()
            self.box = domain

        elif grid_type == "3d_single_fracture":
            # Domain is unit square
            domain = pp.utils.default_domains.CubeDomain([1, 1, 1])

            p = np.array(
                [
                    [0.25, 0.75, 0.75, 0.25],
                    [0.25, 0.25, 0.75, 0.75],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            fracs = [pp.Fracture(p)]
            network = pp.FractureNetwork3d(fracs, domain=domain)

            # Construct mixed-dimensional grid
            gb = network.mesh(mesh_args=mesh_size)
            gb.compute_geometry()
            self.box = domain

        self.faces_split = len(gb.grids_of_dimension(gb.dim_max() - 1)) > 0

        g = gb.grids_of_dimension(gb.dim_max())
        gb_reduced = pp.GridBucket()
        gb_reduced.add_nodes(g)
        self.gb = gb_reduced

        pp.contact_conditions.set_projections(self.gb)

    #    def after_newton_convergence(
    #        self, solution: np.ndarray, errors: float, iteration_counter: int
    #    ) -> None:
    #        # Distribute to pp.STATE
    #        self.dof_manager.distribute_variable(values=solution, additive=False)
    #        self.convergence_status = True

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions: Dirichlet on all global boundaries.
        Neumann on fracture faces.
        """
        all_bf = g.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(g, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = True
        bc.is_dir[:, frac_face] = False
        return bc

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces,
        but non-zero on fracture faces"""
        # Values for all Nd components, facewise
        values = np.zeros((self._Nd, g.num_faces))
        all_bf = g.get_boundary_faces()
        if self.faces_split:
            frac_face = g.tags["fracture_faces"]
            values[: g.dim, frac_face] = 0.1
        else:
            values[0, all_bf] = g.face_centers[0, all_bf] * g.face_centers[1, all_bf]

        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def _assign_variables(self) -> None:
        """
        Assign variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {self.displacement_variable: {"cells": self._Nd}}

    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        """
        gb = self.gb
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.gb)

        eq_manager = pp.ad.EquationManager(self.gb, self.dof_manager)
        Nd = self._Nd
        g_primary: pp.Grid = gb.grids_of_dimension(Nd)[0]

        mpsa_ad = mpsa_ad = pp.ad.MpsaAd(self.mechanics_parameter_key, [g_primary])
        bc_ad = pp.ad.BoundaryCondition(self.mechanics_parameter_key, grids=[g_primary])
        div = pp.ad.Divergence(grids=[g_primary], dim=g_primary.dim)

        # Primary variables on Ad form
        u = eq_manager.variable(g_primary, self.displacement_variable)

        stress = mpsa_ad.stress * u + mpsa_ad.bound_stress * bc_ad

        # momentum balance equation in g_h
        momentum_eq = div * stress

        eq_manager.equations.update(
            {
                "momentum": momentum_eq,
            }
        )

        self._eq_manager = eq_manager

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        if self._use_ad:
            A, b = self._eq_manager.assemble()
        else:
            A, b = self.assembler.assemble_matrix_rhs()

        if self.linear_solver == "direct":
            return spla.spsolve(A, b)
        else:
            # hazmath
            raise NotImplementedError("Not that far yet")


mesh_args = {"mesh_size_bound": 0.2, "mesh_size_frac": 0.2, "mesh_size_min": 0.05}


mesh_type = "2d_no_fracture"
# mesh_type = "2d_single_fracture"
# mesh_type = "3d_no_fracture"
# mesh_type = "3d_single_fracture"

model_params = {"grid_type": mesh_type, "mesh_size": mesh_args}


model = MechanicsProblem(model_params)

pp.run_stationary_model(model, {})

gb = model.gb
g = gb.grids_of_dimension(gb.dim_max())[0]

state = gb.node_props(g, pp.STATE)
u = state[model.displacement_variable]
exp = pp.Exporter(g, "mechanics")
exp.write_vtu({"ux": u[:: g.dim], "uy": u[1 :: g.dim]})

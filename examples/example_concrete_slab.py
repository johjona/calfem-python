import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.utils as cfu

def build_slab_mesh():
    """Create the marked 8 by 5 m quadrilateral mesh from the MATLAB example."""
    geometry = cfg.Geometry()
    geometry.point([0.0, 0.0])
    geometry.point([8.0, 0.0])
    geometry.point([8.0, 5.0])
    geometry.point([0.0, 5.0])

    # MATLAB segment markers: bottom=1, right=2, top=3, left=4.
    geometry.line([0, 1], marker=1, el_on_curve=16)
    geometry.line([1, 2], marker=2, el_on_curve=12)
    geometry.line([2, 3], marker=3, el_on_curve=16)
    geometry.line([3, 0], marker=4, el_on_curve=12)
    geometry.structured_surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(geometry)
    mesh.el_type = 3  # Four-node quadrilateral.
    mesh.dofs_per_node = 3
    mesh.el_size_factor = 1
    coords, edof, dofs, bdofs, _ = mesh.create()
    return np.asarray(coords)[:, :2], edof, dofs, bdofs


def solve_slab():
    """Assemble and solve the concrete slab and return element results."""
    coords, edof, dofs, bdofs = build_slab_mesh()
    ex, ey = cfc.coordxtr(edof, coords, dofs)

    thickness = 0.15
    youngs_modulus = 30e9
    poisson_ratio = 0.2
    pressure = 0.91*(-4.9e3*1.2 - 2e3*1.5)
    D = cfc.hooke(1, youngs_modulus, poisson_ratio)
    ep = [thickness]

    ndof = int(np.max(edof))
    K = np.zeros((ndof, ndof))
    f = np.zeros((ndof, 1))
    for i in range(edof.shape[0]):
        Ke, fe = cfc.platre(ex[i], ey[i], ep, D, pressure)
        K, f = cfc.assem(edof[i], K, Ke, f, fe)

    # Clamp all three plate DOFs on segment 1 and transverse displacement
    # (the first DOF) on segment 2, matching the MATLAB example.
    bc = np.array([], dtype=int)
    bc_val = np.array([], dtype=float)
    # bc, bc_val = cfu.applybc(bdofs, bc, bc_val, [1, 2, 3], 0.0)
    bc, bc_val = cfu.apply_bc_3d(bdofs, bc, bc_val, 1, dimension = 1) 
    bc, bc_val = cfu.apply_bc_3d(bdofs, bc, bc_val, 2, dimension = 1) 
    bc, bc_val = cfu.apply_bc_3d(bdofs, bc, bc_val, 3, dimension = 1) 
    bc, bc_val = cfu.apply_bc_3d(bdofs, bc, bc_val, 1, dimension = 2)
    a, r = cfc.solveq(K, f, bc, bc_val)
    ed = cfc.extract_eldisp(edof, a)

    es = np.zeros((edof.shape[0], 5))
    et = np.zeros((edof.shape[0], 3))
    for i in range(edof.shape[0]):
        es[i], et[i] = cfc.platrs(ex[i], ey[i], ep, D, ed[i])

    return coords, edof, dofs, bdofs, a, r, es, et


def fill3(coords, edof, dofs, a):
    """Plot the undeformed mesh and the transverse displacement surface."""
    w = np.asarray(a).reshape(-1)[dofs[:, 0] - 1]
    element_nodes = (edof[:, ::3] - 1)//3

    figure = plt.figure(figsize=(12, 5))
    mesh_ax = figure.add_subplot(1, 2, 1)
    for nodes in element_nodes:
        loop = np.r_[nodes, nodes[0]]
        mesh_ax.plot(coords[loop, 0], coords[loop, 1], color="black", linewidth=0.45)
    mesh_ax.set_aspect("equal")
    mesh_ax.set_title("Concrete slab mesh")
    mesh_ax.set_xlabel("x [m]")
    mesh_ax.set_ylabel("y [m]")

    def_ax = figure.add_subplot(1, 2, 2, projection="3d")
    vertices = [
        np.column_stack((coords[nodes], w[nodes]))
        for nodes in element_nodes
    ]
    surface = Poly3DCollection(
        vertices,
        array=np.mean(w[element_nodes], axis=1),
        cmap="viridis",
        edgecolors="black",
        linewidths=0.2,
    )
    def_ax.add_collection3d(surface)
    def_ax.set_xlim(0, 8)
    def_ax.set_ylim(0, 5)
    def_ax.set_zlim(float(np.min(w)), float(np.max(w)))
    def_ax.set_title("Transverse displacement")
    def_ax.set_xlabel("x [m]")
    def_ax.set_ylabel("y [m]")
    def_ax.set_zlabel("w [m]")
    figure.colorbar(surface, ax=def_ax, shrink=0.65, label="w [m]")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    coords, edof, dofs, bdofs, a, reactions, es, et = solve_slab()
    print(f"Elements: {edof.shape[0]}")
    print(f"Maximum transverse displacement: {np.max(np.abs(a[dofs[:, 0] - 1])):.6g} m")
    print(f"Maximum absolute bending moment: {np.max(np.abs(es[:, :3])):.6g} N m/m")
    fill3(coords, edof, dofs, a)


## Algorithmic Structure (High-Level Overview)

The ESFEM implementation follows a clear **algorithmic pipeline**:

```text
Parametric surface / mesh
        ↓
Geometric quantities
(mean curvature, normals, metric terms)
        ↓
Weak formulation of surface PDE
        ↓
ESFEM spatial discretization
        ↓
Semi-discrete ODE system
        ↓
Structure-aware time integration
        ↓
Updated surface and mesh
```

This structure is reflected directly in the code layout:

- **Geometry definition** → `geometry/`
- **Variational formulation & ESFEM operators** → `esfem/`
- **Time integration and ALE motion** → `esfem/ale.py`
- **Application-level flows** → `esfem/_applications/`

---

## Repository Structure and Main Components

### 1. Core ESFEM Framework (`esfem/`)

This directory contains the **core solvers for PDEs on evolving surfaces**, built on top of **NGSolve**.

Key modules include:

- `willmore_mdr.py`  
  Willmore flow using a mean-curvature-driven reformulation.

- `mcf_mdr.py`  
  Mean curvature flow in a related formulation.

- `dziuk.py`  
  Classical ESFEM formulation following Dziuk.

- `bgn.py`  
  BGN-type schemes for geometric evolution.

- `ale.py`  
  ALE motion and mesh handling.

- `ode.py`  
  Time discretization of semi-discrete systems.

High-level application drivers:
- `esfem/_applications/willmore_flow.py`
- `esfem/_applications/mcf_flow.py`

---

### 2. Geometry and Mesh Utilities (`geometry/`)

This directory provides **parametric and discrete geometry tools**:

- Parametric curves and surfaces:  
  `param_curve.py`, `param_surface.py`

- Mesh data structures:  
  `mesh_1d.py`, `mesh_2d.py`, `discrete_mesh.py`

- Special geometries:  
  `angenent.py` (Angenent torus)

Example notebooks for geometry and mesh generation:

- `geometry/AngenentTorus.ipynb`
- `geometry/Mirror-symmetry-mesh.ipynb`
- `geometry/MGen/PFEM-ParamSurface-class.ipynb`
- `geometry/MGen/PFEM-DMesh-Perturbed-Torus-2.ipynb`

These are useful starting points for creating **new initial surfaces**.

---

### 3. Numerical Experiments (Main Entry Points)

The main Willmore flow simulations are provided as Jupyter notebooks:

- `PFEM-KLL-Willmore-PerturbedTorus.ipynb`
- `PFEM-KLL-Willmore-RBC1.ipynb`
- `PFEM-KLL-Willmore-RBC2.ipynb`
- `PFEM-KLL-Willmore-Clifford1.ipynb`

These notebooks:
- assemble the ESFEM discretization,
- perform time evolution,
- visualize the evolving surfaces.

---

### 4. Assembly, Utilities, and Visualization (details)

- `assemble/artificial_interface.py`  
  Assembly utilities.

- `es_utils.py`, `global_utils.py`  
  Shared helper functions.

- `viz/vtk_out.py`  
  Export results to VTK format for ParaView.

# ngs-esfem  
**A General ESFEM Framework and Reference Implementation for Geometric Flows**

This repository provides a **general-purpose evolving surface finite element method (ESFEM / PFEM) framework** together with **reference implementations of geometric flows**, most notably the **Willmore flow of closed surfaces**.

The code base is designed both as:
- a **reproducible research implementation** of state-of-the-art ESFEM algorithms, and  
- a **reusable toolbox** for developing and testing new PDEs on evolving curves and surfaces.

The implementation follows the papers

> **J. Hu, B. Li (2022)**  
> *Evolving finite element methods with an artificial tangential velocity for mean curvature flow and Willmore flow*,  
> Numerische Mathematik, 152(1), pp. 127–181.

> **B. Kovács, B. Li, C. Lubich (2021)**  
> *A convergent evolving finite element algorithm for Willmore flow of closed surfaces*,  
> Numerische Mathematik, 149(3), pp. 595–643.

If you use this code, please consider citing the papers above.

---

## Scope and Positioning

**ngs-esfem** is intended as a

> **general ESFEM framework and reference implementation for geometric evolution equations**.

It supports:
- parametric evolving meshes,
- weak formulations of surface PDEs,
- structure-preserving time discretizations,
- and reproducible numerical experiments.

While the repository includes **high-quality Willmore flow simulations**, its modular design allows straightforward extension to:
- mean curvature flow,
- surface diffusion,
- and other geometric or coupled surface PDEs.

---

## Quick Start  
### Generate a Perturbed Torus in 3 Steps

### Step 1: Clone the repository

```bash
git clone --recurse-submodules https://github.com/jiashhu/ngs-esfem.git
cd ngs-esfem
```

### Step 2: Build and start the Docker environment (recommended)

```bash
docker build -t my-ngsxfem .
./run-image.sh
```

This starts a Jupyter Notebook server with **NGSolve and all required dependencies preinstalled**.

---

### Step 3: Run a numerical example

Open the following notebook as a recommended starting point:

- [`geometry/MGen/PFEM-DMesh-Perturbed-Torus-2.ipynb`](geometry/MGen/PFEM-DMesh-Perturbed-Torus-2.ipynb)  
  → Generation of a **high-order discrete surface** for a perturbed torus.

Run all cells to generate the surface and visualize the geometry.

---

## Geometry Gallery and Mesh Generation Examples

The folder **`geometry/MGen/`** contains a small **gallery of geometry and mesh generation examples**, illustrating how complex initial surfaces can be constructed and discretized.

Notable examples include:

- [`geometry/MGen/PFEM-MDR-Perturbed-Torus-Beads.ipynb`](geometry/MGen/PFEM-MDR-Perturbed-Torus-Beads.ipynb)  
  → Perturbed torus with **beads-like geometric features**.

- [`geometry/MGen/AngenentTorus.ipynb`](geometry/MGen/AngenentTorus.ipynb)  
  → Shooting method to generate **Angenent Torus**.

- [`geometry/MGen/PFEM-MDR-Perturbed-Torus-Wave.ipynb`](geometry/MGen/PFEM-MDR-Perturbed-Torus-Wave.ipynb)  
  → Perturbed torus with **strong wave-like deformations** (large-amplitude twisting).

- [`geometry/MGen/PFEM-ParamSurface-class.ipynb`](geometry/MGen/PFEM-ParamSurface-class.ipynb)  
  → Demonstration of the parametric surface class and its discretization.

These notebooks can be viewed as a **geometry and mesh generation gallery**, and serve as templates for constructing new initial surfaces for ESFEM-based simulations.


## 一些特殊曲面类的生成

`param_surface` 中的 `ParamSurface` 类用于生成 **参数化的二维曲面**。[说明参考](geometry/tutorial/ParamSurface.ipynb)


## Tutorials

Notebook 路径：`/work/ngs-esfem/tutorial`

- `Function_Set.ipynb`
- `ParamSurface.ipynb`
- `PFEM-Dumbbell-Dziuk-MCF.ipynb`
- `PFEM-Dumbbell-MDR-MCF.ipynb`
- `PFEM-Tanh-MDR-MCF.ipynb`

---

## Additional Details

Further implementation details and notes can be found in:

- [`ngs-esfem/details.m`](ngs-esfem/details.m)

This file documents algorithmic and implementation-specific aspects that are not exposed directly in the notebooks.

---
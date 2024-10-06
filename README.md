# OnsagerNet

This is an implementation of the *onsagernet* library in [`jax`](https://jax.readthedocs.io/en/latest/)
with [`equinox`](https://docs.kidger.site/equinox/).

---

## Installation and setup

*Currently this is for development. Package builds to be done in the future.*


1. Install Poetry by following the instructions on this link: [Poetry](https://python-poetry.org/docs/#installation) 
2. Set up env and install dependencies + package
```shell
make env
```

---

### Selective installs

By default `make env` installs all dependencies.
You can selectively install dependencies, in which case instead of `make env`, run one of the following:

Only the main package dependencies
```shell
poetry shell
poetry install --only main
```

Only the main package and those for examples, but not development packages
```shell
poetry shell
poetry install --only main,examples
```

---

### Package management for development

Add a package
```shell
poetry add <package-name>
```

Remove a package
```shell
poetry remove <package-name>
```

---

## Using GPUs

The current `pyproject.toml` performs a CPU install.
If you have a CUDA-enabled GPU you can install the GPU version of `jax`
```shell
make env
poetry remove jax
poetry add 'jax["cuda12"]'
```

---

## Auto-generate API documentation

Generate API docs (see [pdoc](https://pdoc3.github.io/pdoc/) for details)
```shell
make docs
```
View documentation at [`./docs/index.html`](./docs/) or directly [here](https://liqianxiao.github.io/Research-OnsagerNetJax/onsagernet.html)
for the current tracked version.

---

## Testing

Run unit tests (complete coverage in progress)
```shell
make tests
```

---

## Examples

Currently there are three examples and follow the links for more details.
1. [A test case example](./examples/test_case/README.md)
2. [An example based on homogenisation in Ref [3]](./examples/n_scale_potential/README.md)
3. [An example in learning macroscopic polymer dynamics in Ref [2]](./examples/polymer_dynamics/README.md)

---

## References

1. *Yu, H., Tian, X., E, W. & Li, Q. OnsagerNet: Learning stable and interpretable dynamics using a generalized Onsager principle. Phys. Rev. Fluids 6, 114402 (2021).*
2. *Chen, X. et al. Constructing custom thermodynamics using deep learning. Nature Computational Science 4, 66–85 (2024).*
3. *Duncan, A. B., Duong, M. H. & Pavliotis, G. A. Brownian Motion in an N-Scale Periodic Potential. J Stat Phys 190, 82 (2023).*

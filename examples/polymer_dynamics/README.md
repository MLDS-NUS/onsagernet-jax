# Learning closure models for polymer dynamics

This example is based on reference [1], which aims to learn
a macroscopic description of a complex dynamical system from
observations of its microscopic trajectories.

The basic problem is to learn a 3D macroscopic dynamics
from observations of microscopic trajectories consisting of a
300-bead linear polymer (900 spatial dimensions).
See [1] for more details.

We will use the `ReducedSDE` model since this involves
both a model reduction step and learning the SDE step.
The model reduction/reconstruction will be using PCA-ResNet [2],
and the dynamics will be a general stochastic `OnsagerNet`.

## Running the training script
To run the data generation and training routines, issue
```shell
python polymer_dynamics.py
```

## Configurations
The default configuration file is found in [`./config/polymer_dynamics.yaml`](./config/polymer_dynamics.yaml)


## Results
The training results, logs, model checkpoints etc are saved in `./outputs`, which are automatically time-stamped.

The analysis of the results are found in the notebook [`./polymer_dynamics_analysis.ipynb`](./polymer_dynamics_analysis.ipynb),
which reads from the generated raw results.
*Remember to modify the path in the notebook appropriately to read from your saved checkpoints*.


## References
1. Chen, X. et al. Constructing custom thermodynamics using deep learning. Nature Computational Science 4, 66–85 (2024).
2. Yu, H., Tian, X., E, W. & Li, Q. OnsagerNet: Learning stable and interpretable dynamics using a generalized Onsager principle. Phys. Rev. Fluids 6, 114402 (2021).

"""
# OnsagerNet package

This package implements the basic routines for building and training
(with or without model reduction/closure modelling)
OnsagerNet and variants.

## Main Modules
- `onsagernet.dynamics`: Dynamic models, including SDEs, OnsagerNet, and variants
- `onsagernet.transformations`: Transformations for dimensionality reduction and reconstruction
- `onsagernet.trainers`: Training routines for SDEs and those with dimensionality transformations
- `onsagernet.models`: Basic model definitions (mostly fully connected neural networks) that are used in references [1-3]

## References
1. Chen, X. et al. Constructing custom thermodynamics using deep learning. Nature Computational Science 4, 66-85 (2024).
2. Novoselov, K. S. & Li, Q. Learning physical laws from observations of complex dynamics. Nature Computational Science 1-2 (2024).
3. Yu, H., Tian, X., E, W. & Li, Q. OnsagerNet: Learning stable and interpretable dynamics using a generalized Onsager principle. Phys. Rev. Fluids 6, 114402 (2021).
"""

# Homogenisation of a n-scale potential

This is a further test of `OnsagerNetFD` on a known example
(see the [`../test_case`](../test_case/) example for a first test).
This example is taken from [1].


## Formulation

Here we will use $\sigma>0$ to denote temperature and $\varepsilon>0$ is a scale parameter.

Let us consider a two variable potential

$$
    v(x_0, x_1) = v_0(x_0) + v_1(x_0, x_1)
$$

with $x_0,x_1 \in \mathbb{R}$,
and $v_1$ is periodic in $x_1$, i.e. defined on $\mathbb T$.
and consider the following SDE:

$$
\begin{equation}
    \begin{aligned}
        dX_0(t) &=
        - \left[
            \nabla_{x_0} v(X_0(t), X_1(t))
            + \frac{1}{\varepsilon} \nabla_{x_1} v(X_0(t), X_1(t))
        \right] dt
        +
        \sqrt{2\sigma} dW(t) \\
        dX_1(t) &=
        - \left[
            \frac{1}{\varepsilon}
            \nabla_{x_0} v(X_0(t), X_1(t))
            + \frac{1}{\varepsilon^2} \nabla_{x_1} v(X_0(t), X_1(t))
        \right] dt
        +
        \frac{1}{\varepsilon}\sqrt{2\sigma} dW(t).
    \end{aligned}
\end{equation}
$$

It is proved in [1] that
as $\varepsilon \to 0$, $Z=X_0$ converges to the solution of

$$
    dZ(t) =
    \left[
        - M(Z(t)) \nabla \Psi(Z(t))
        + \sigma \nabla \cdot M(Z(t))
    \right]
    dt + \sqrt{2 \sigma M(Z(t))} dW(t),
$$

where

$$
\begin{equation}
    \begin{aligned}
        \Psi(z) &= - \log \int_{\mathbb{T}} e^{-v(z, x)} dx,\\
        M(z) &= \frac{1}{u_1(z) \hat{u}_1(z)},
        \qquad
        u_1(z) = \int_{\mathbb{T}} e^{-v_1(z, x)} dx,
        \qquad
        \hat{u}_1(z) = \int_{\mathbb{T}} e^{v_1(z, x)} dx.
    \end{aligned}
\end{equation}
$$

Our goal is to test if `OnsagerNetFD` can recover,
up to inherent degeneracies,
the correct the homogenised dynamics
when $\varepsilon$ is small.
We note here that the reduced dimension is $1$ and hence no closure problem
needs to considered -- i.e. the dynamics of $Z = \varphi(X) = X_0$ is automatically closed.

For concreteness, let us take $\sigma=1$, and

$$
\begin{equation}
    v_0(x_0) = x_0^2,
    \qquad
    v_1(x_0, x_1)
    = \log
    \left[
        1 +
        \sin^2(\pi x_0) \sin^2(\pi x_1)
    \right].
\end{equation}
$$

A direct calculation shows

$$
\begin{equation}
\begin{aligned}
    M(z) &=
    \frac{\sqrt{1 + \sin^2(\pi z)}}{1 + \frac{1}{2}\sin^2(\pi z)}
    \\
    V(z) &=
    z^2 + \frac{1}{2}
    \log \left[
        1 + \frac{1}{2} \sin^2(\pi z)
    \right]
\end{aligned}
\end{equation}
$$

We will check if as $\varepsilon \to 0$, if `OnsagerNetFD`
recovers this homogenised potential function and dissiptation function.


## Running the training script
To run the data generation and training routines, issue
```shell
python n_scale_potential.py
```
Since we want to check convergence as $\varepsilon\to 0$,
we will use the multirun feature of `hydra`
```shell
python n_scale_potential.py -m target.eps=0.01,0.05,0.1,0.5,1.0,2.0,5.0
```
This runs in parallel by default using `joblib`. If you want to run it serially, remove the following
from the configuration file, or use command-line overrides
```yaml
defaults:
  - override hydra/launcher: joblib # use joblib for parallel multirun, remove if unwanted
```

## Configurations
The default configuration file is found in [`./config/n_scale_potential.yaml`](./config/n_scale_potential.yaml)


## Results
The training results, logs, model checkpoints etc are saved in `./outputs`, which are automatically time-stamped.

The analysis of the results are found in the notebook [`./n_scale_potential_analysis.ipynb`](./n_scale_potential_analysis.ipynb),
which reads from the generated raw results.
*Remember to modify the path in the notebook appropriately to read from your saved checkpoints*.


## References
1. Duncan, A. B., Duong, M. H. & Pavliotis, G. A. Brownian Motion in an N-Scale Periodic Potential. J Stat Phys 190, 82 (2023).
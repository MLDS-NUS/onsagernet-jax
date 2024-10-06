# A simple test case

This simple example tests the training of a variant of the stochastic OnsagerNet
which further constrains the OnsagerNet to satisfy a fluctuation-dissipation relationship.

More precisely, we implement the following dynamics

$$
    dZ(t) =
    - \left[
        M(Z(t)) + W(Z(t))
    \right]
    \nabla V(Z(t)) dt
    + \epsilon \nabla \cdot M(Z(t)) dt
    + \sqrt{2 \epsilon M(Z(t))} dW(t).
$$

This is implemented in [`onsagernet.dynamics.OnsagerNetFD`](../../onsagernet/dynamics.py)

As a first test, we will generate data according to this dynamics and use it to train a `OnsagerNetFD` dynamics,
and check if we obtain the correct results, at least up to expected degeneracies.
For example, at most the potential function $V$ can only be recovered up to an additive constant.

More precisely, we take dimension $d=2$ and $Z = (Z_1, Z_2)$
with (totally arbitrarily chosen)

$$
    V(z) = \frac{1}{2} | z | ^2
    + 0.3 \cos(z_1)^2 \sin(z_1 - 2 z_2)^2,
$$

and

$$
    M(x) = \frac{1}{5} L(x) L(x)^\top
    + 3 I,
    \qquad
    L(x)
    = \begin{pmatrix}
        \sin(x_1) & x_2 - x_1 \\
        3 x_1 & \cos(x_2)
    \end{pmatrix}.
$$


## Running the training script
To run the data generation and training routines, issue
```shell
python test_case.py
```

## Configurations
The default configuration file is found in [`./config/test_case.yaml`](./config.test_case.yaml)
and you can use command-line overrides to change the problem, model or training settings.
For example
```shell
python test_case.py train.batch_size=4
```
trains the model with `batch_size` equal to 4 and overwrites the default configurations.


## Results
The training results, logs, model checkpoints etc are saved in `./outputs`, which are automatically time-stamped.

The analysis of the results are found in the notebook [`./test_case_analysis.ipynb`](./test_case_analysis.ipynb),
which reads from the generated raw results.
*Remember to modify the path in the notebook appropriately to read from your saved checkpoints*.

r"""
# Basic trainers for SDE models

This module implements basic training routines for SDE models.
The base class is `SDETrainer`, which provides the base training logic.
The sub-classes are required to implement the `SDETrainer.loss_func` that is used to train the model.

## Training routines

We provide here two training routines.

- `MLETrainer`: this implements the maximum likelihood loss [`MLELoss`](./_losses.html#MLELoss)
  to estimate SDE drift and diffusion
  through computing the [maximum-likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) following the
  [Euler-Maruyama discretisation](https://en.wikipedia.org/wiki/Euler–Maruyama_method) of the SDE.
- `ClosureMLETrainer`: this combines [`MLELoss`](./_losses.html#MLELoss) with additional losses to enforce closure,
  namely a reconstruction loss [`ReconLoss`](./_losses.html#ReconLoss) and a
  comparison loss [`CompareLoss`](./_losses.html#CompareLoss).
  This follows the implementation of [1].

The following example shows how to train an `onsagernet.dynamics.OnsagerNet` model using the `MLETrainer`.
```python
from onsagernet.dynamics import OnsagerNet
from onsagernet.trainers import MLETrainer

sde = OnsagerNet(...)
dataset = load_data(...)  # return a datasets.Dataset object

trainer = MLETrainer(opt_options=config.train.opt, rop_options=config.train.rop)
sde, losses, _ = trainer.train(  # trains the model `sde` for 10 epochs with batch size 2
    model=sde,
    dataset=dataset,
    num_epochs=10,
    batch_size=2,  # batch size should be typically small since this yields [n_batch, n_steps, n_dim] data
)
```

## Dataset format
The dataset is assumed
to be a huggingface [`datasets.Dataset`](https://huggingface.co/docs/datasets/en/index)
object with three columns: `t`, `x`, and `args`.
Here, `t` is time, `x` is state, and `args` are additional arguments.
Let `n_examples` be the number of examples, `n_dim` be the dimension of the state x,
`n_args` be the dimension of the arguments, and `n_steps` be the number of time steps.
Then, the shapes of the data entries are as follows:
- t: (`n_examples`, `n_steps`, `1`) - this is the sequence of time steps sampled in the time series,
  which can vary from one example to another (but the `n_steps` must be the same), so consider padding
  if this were not the case
- x: (`n_examples`, `n_steps`, `n_dim`) - this is the corresponding sequence of states
- args: (`n_examples`, `n_steps`, `n_args`) - this is the sequence of additional arguments,
  which can be constant or varying in time. If constant, you must broad-cast the values
  along the time dimension so that the expected shapes are maintained.

We also assume that `n_args` >= `1`, with the first dimension always being
temperature, since it is relevant for all SDE models in physical modelling.
Your custom `loss_func` routine must return a scalar loss value.

*In fact, it is not strictly necessary to use a [`datasets.Dataset`](https://huggingface.co/docs/datasets/en/index)
object, but any object that can yield the correct data format by calling `dataset.iter(batch_size)`.*

## Writing custom training routines

You can write custom training routines (with custom losses, for example)
by sub-classing `SDETrainer`.
Sub-classes must implement the `SDETrainer.loss_func` method.

Given a model partitioned by `eqx.partition(model, ...)`,
into a trainable part `diff_model` and a static part `static_model`,
the `SDETrainer.loss_func` routine computes the loss of the model on the given dataset.
We always assume that the batch data is a tuple of `(t, x, args)`.

See above for more on the format of the dataset and
`MLETrainer.loss_func`, or the more complex `ClosureMLETrainer.loss_func` for examples.

## References

1. Chen, X. et al. *Constructing custom thermodynamics using deep learning*. Nature Computational Science **4**, 66–85 (2024).


"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod

from optax import adam, chain
from optax.contrib import reduce_on_plateau
from optax.tree_utils import tree_get
from jax.tree_util import tree_map

from tqdm import tqdm

from ._losses import MLELoss, ReconLoss, CompareLoss

# ------------------------- Typing imports ------------------------- #

from chex import ArrayTree  # to silence pdoc warnings
from .dynamics import SDE, ReducedSDE
from typing import Optional, Any, Union
from jax.typing import ArrayLike
from jax import Array
from datasets import Dataset
from optax import GradientTransformation, OptState
from logging import Logger

DynamicModel = Union[SDE, ReducedSDE]

# ------------------------------------------------------------------ #
#                              Trainers                              #
# ------------------------------------------------------------------ #


class SDETrainer(ABC):
    """Base class for training SDE models."""

    def __init__(
        self, opt_options: dict, rop_options: dict, loss_options: Optional[dict] = None
    ) -> None:
        """SDE training routine.

        Args:
            opt_options (dict): dictionary of options for the optimiser
            rop_options (dict): dictionary of options for the reduce-on-plateau callback
            loss_options (Optional[dict], optional): dictionary of options for loss computation. Defaults to None.
        """
        self._opt_options = opt_options
        self._rop_options = rop_options
        self._loss_options = loss_options

    @eqx.filter_jit
    @abstractmethod
    def loss_func(
        self,
        diff_model: DynamicModel,
        static_model: DynamicModel,
        t: ArrayLike,
        x: ArrayLike,
        args: ArrayLike,
    ) -> float:
        """Loss function.

        This must be implemented by sub-classes.

        Args:
            diff_model (DynamicModel): the trainable part of the model
            static_model (DynamicModel): the static part of the model
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters. The first dimension is temperature.

        Returns:
            float: loss value
        """
        pass

    def _make_optimiser(
        self, opt_options: dict, rop_options: dict
    ) -> GradientTransformation:
        """Make an optimiser.

        Args:
            opt_options (dict): optimiser options
            rop_options (dict): reduce-on-plateau options

        Returns:
            GradientTransformation: an optimiser object from `optax`
        """
        return chain(adam(**opt_options), reduce_on_plateau(**rop_options))

    @eqx.filter_jit
    def _make_step(
        self,
        model: DynamicModel,
        data: Dataset,
        opt: GradientTransformation,
        opt_state: OptState,
        filter_spec: Any,
    ) -> tuple[DynamicModel, OptState, float]:
        """Make a training step.

        Args:
            model (DynamicModel): the model to be trained
            data (Dataset): the dataset object
            opt (GradientTransformation): optimiser object
            opt_state (OptState): optimiser state
            filter_spec (Any): the filtering logic to determine which parts of the model to train

        Returns:
            tuple[DynamicModel, OptState, float]: trained model, optimiser state, loss value
        """
        diff_model, static_model = eqx.partition(model, filter_spec)

        loss_value, grads = eqx.filter_value_and_grad(self.loss_func)(
            diff_model, static_model, *data
        )
        updates, opt_state = opt.update(grads, opt_state, model, value=loss_value)
        model = eqx.apply_updates(model, updates)
        return model, loss_value, opt_state

    def _train_epoch(
        self,
        model: DynamicModel,
        dataset: Dataset,
        batch_size: int,
        opt: GradientTransformation,
        opt_state: OptState,
        filter_spec: Any,
    ) -> tuple[DynamicModel, float, OptState]:
        """Train the model for an epoch.

        Args:
            model (DynamicModel): the model to be trained
            dataset (Dataset): the dataset object
            batch_size (int): the batch size
            opt (GradientTransformation): the optimiser object
            opt_state (OptState): the optimiser state
            filter_spec (Any): the filtering logic to determine which parts of the model to train

        Returns:
            tuple[DynamicModel, float, OptState]: trained model, loss value, optimiser state
        """

        step_losses = []

        for batch in tqdm(
            dataset.iter(batch_size),
            total=dataset.num_rows // batch_size,
        ):
            data_batch = (batch["t"], batch["x"], batch["args"])
            model, train_loss, opt_state = self._make_step(
                model, data_batch, opt, opt_state, filter_spec
            )
            step_losses.append(train_loss)
        epoch_loss = jnp.mean(jnp.array(step_losses))
        return model, epoch_loss, opt_state

    def train(
        self,
        model: DynamicModel,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int,
        logger: Optional[Logger] = None,
        opt_state: Optional[OptState] = None,
        filter_spec: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: Optional[int] = None,
    ) -> tuple[DynamicModel, list[float], OptState]:
        """The main training routine.

        Args:
            model (DynamicModel): the model to be trained
            dataset (Dataset): the dataset
            num_epochs (int): number of epochs to train
            batch_size (int): the batch size
            logger (Optional[Logger], optional): the logging object. Defaults to None.
            opt_state (Optional[OptState], optional): the starting optimiser state. Defaults to None.
            filter_spec (Optional[Any], optional): the filtering logic. Defaults to None.
            checkpoint_dir (Optional[str], optional): the directory to save checkpoints. Defaults to None.
            checkpoint_every (Optional[int], optional): checkpoints are saved every `checkpoint_every` number of epochs. Defaults to None.

        Returns:
            tuple[DynamicModel, list[float], OptState]: trained model, list of losses, optimiser state
        """
        opt = self._make_optimiser(self._opt_options, self._rop_options)
        if opt_state is None:
            opt_state = opt.init(eqx.filter(model, eqx.is_array))

        if filter_spec is None:
            filter_spec = tree_map(lambda _: True, model)

        losses = []
        for epoch in range(num_epochs):
            model, step_loss, opt_state = self._train_epoch(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                opt=opt,
                opt_state=opt_state,
                filter_spec=filter_spec,
            )
            losses.append(step_loss)
            if logger:
                lr_scale = tree_get(opt_state, "scale")
                logger.info(
                    f"epoch={epoch:05d}, loss={step_loss:.6f}, lr_scale={lr_scale:.4f}"
                )

            if checkpoint_dir is not None and epoch % checkpoint_every == 0:
                model_path = os.path.join(
                    checkpoint_dir, f"model_epoch_{epoch:05d}.eqx"
                )
                eqx.tree_serialise_leaves(model_path, model)

        return model, losses, opt_state


class MLETrainer(SDETrainer):

    @eqx.filter_jit
    def loss_func(
        self,
        diff_model: DynamicModel,
        static_model: DynamicModel,
        t: ArrayLike,
        x: ArrayLike,
        args: ArrayLike,
    ) -> float:
        """The MLE loss function.

        See [`MLELoss`](./_losses.html#MLELoss) for more details.

        Args:
            diff_model (DynamicModel): the trainable part of the model
            static_model (DynamicModel): the static part of the model
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters.

        Returns:
            float: the computed loss
        """
        model = eqx.combine(diff_model, static_model)
        return MLELoss()(model, t, x, args)


class ClosureMLETrainer(MLETrainer):

    @eqx.filter_jit
    def loss_func(
        self,
        diff_model: DynamicModel,
        static_model: DynamicModel,
        t: ArrayLike,
        x: ArrayLike,
        args: ArrayLike,
    ) -> float:
        r"""The combined loss function for MLE training with closure modelling.

        The losses are applied to the combine model
        `model = eqx.combine(diff_model, static_model)`
        with three parts
        - [`MLELoss`](./_losses.html#MLELoss) applied to `model.sde`
        - [`ReconLoss`](./_losses.html#ReconLoss) applied to the `model`
        - [`CompareLoss`](./_losses.html#CompareLoss) applied to the `model`

        The combined loss is given by
        $$
            \text{loss} = \text{loss_sde}
            + \text{recon_weight} \times \text{loss_recon}
            + \text{compare_weight} \times \text{loss_compare}
        $$

        The variables `recon_weight` and `compare_weight` are set
        in the `loss_options` attribute.

        Args:
            diff_model (DynamicModel): the trainable part of the model
            static_model (DynamicModel): the static part of the model
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters.

        Returns:
            float: the computed loss
        """
        model = eqx.combine(diff_model, static_model)
        z = jax.vmap(jax.vmap(model.encoder))(x)
        loss_sde = MLELoss()(model.sde, t, z, args)
        loss_recon = ReconLoss()(model, x)
        loss_compare = CompareLoss()(model, x)
        return (
            loss_sde
            + self._loss_options["recon_weight"] * loss_recon
            + self._loss_options["compare_weight"] * loss_compare
        )

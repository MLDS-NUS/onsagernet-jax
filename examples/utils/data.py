"""Data utilities."""

import jax.numpy as jnp
from datasets import Dataset, Features, Array2D


def shrink_trajectory_len(dataset: Dataset, new_traj_len: int) -> Dataset:
    """Reshapes a dataset to shrink trajectory length and increase number of examples

    This is to optimise GPU usage when the trajectory length is too long.

    Some factors to consider
    - The `new_traj_len` must be at least 2 for the loss function to work.
    - The `new_traj_len` must be smaller than or equal to the original trajectory length.
      It is suggested to use a value that is a divisor of the original trajectory length.

    Args:
        dataset (Dataset): input dataset object
        new_traj_len (int): new trajectory length

    Returns:
        Dataset: processed dataset object
    """
    old_traj_len = dataset.features["t"].shape[0]

    assert new_traj_len >= 2, "new_traj_len must be at least 2"
    assert (
        old_traj_len >= new_traj_len
    ), "new_traj_len must be smaller than the original trajectory length"

    old_dtype = dataset.features["t"].dtype

    # Iterate to load the dataset and change shage
    ts, xs, argss = [], [], []
    shrink_factor = old_traj_len // new_traj_len
    max_time_idx = (
        new_traj_len * shrink_factor
    )  # this truncates the trajectories if old_traj_len is not divisible by new_traj_len
    for example in dataset.iter(batch_size=1):
        for col, dlist in zip(["t", "x", "args"], [ts, xs, argss]):
            arr = example[col]
            new_shape = (new_traj_len, arr.shape[-1])
            arr_reshaped = arr[:, :max_time_idx, :].reshape(1, *new_shape)
            dlist.append(arr_reshaped)

    ts = jnp.concatenate(ts)
    xs = jnp.concatenate(xs)
    argss = jnp.concatenate(argss)

    # return the dataset
    new_features = Features(
        {
            "t": Array2D(shape=ts.shape[1:], dtype=old_dtype),
            "x": Array2D(shape=xs.shape[1:], dtype=old_dtype),
            "args": Array2D(shape=argss.shape[1:], dtype=old_dtype),
        }
    )

    new_dataset = Dataset.from_dict(
        {
            "t": ts,
            "x": xs,
            "args": argss,
        },
        features=new_features,
    )
    return new_dataset.with_format("jax")

"""
Frozen neural-model definitions for the SOLETE forecasting study.

This module extracts the reusable scientific method from Notebook 11:

* leakage-safe 13 x 9 historical sequence contract;
* recurrent architecture grid for LSTM, GRU, and BiLSTM;
* the later executed/frozen causal TCN architecture;
* train-only neural scaling semantics;
* architecture-selection ordering;
* validation-selected frozen epoch counts;
* final train+validation refit policy.

Important
---------
Notebook 11 contains an earlier provisional generic TCN definition.
That provisional TCN is intentionally NOT exported here.

The TCN implementation below is the later executed/frozen version used
for the completed TCN experiments:
    causal Conv1D residual blocks,
    two convolutions per block,
    final historical sequence position,
    selected TCN_C1 receptive field = 29 timesteps.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------
# Common frozen neural protocol
# ---------------------------------------------------------------------

NEURAL_RANDOM_SEED = 42

SEQUENCE_LENGTH = 13
SEQUENCE_STEP_MINUTES = 5
HISTORY_MINUTES = 60

SEQUENCE_CHANNELS = (
    "P_Solar[kW]",
    "GHI[kW1m2]",
    "POA Irr[kW1m2]",
    "TEMPERATURE[degC]",
    "Humidity_clean",
    "Pressure_clean[mbar]",
    "WIND_SPEED[m1s]",
    "wind_direction_sin",
    "wind_direction_cos",
)

FORECAST_HORIZONS = (
    15,
    30,
    60,
)

FEATURE_FAMILIES = (
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
)

AUXILIARY_DIMS = {
    "F0": 1,
    "F1": 16,
    "F2": 21,
    "F3": 30,
    "F4": 127,
}

ARCHITECTURE_SELECTION_HORIZON = 60
ARCHITECTURE_SELECTION_FEATURE_FAMILY = "F0"

BATCH_SIZE = 256
MAX_VALIDATION_EPOCHS = 80
INITIAL_LEARNING_RATE = 1e-3
ADAM_CLIPNORM = 1.0
LOSS = "mae"

EARLY_STOP_MONITOR = "val_loss"
EARLY_STOP_MODE = "min"
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
RESTORE_BEST_WEIGHTS = True

SHUFFLE_TRAINING_SAMPLES = True
CLIP_PRIMARY_PREDICTIONS = False

SCALING_DDOF = 0
ALLOW_ZERO_VARIANCE_CONTINUOUS = False

FINAL_REFIT_USE_TRAIN_PLUS_VALIDATION = True
FINAL_REFIT_REUSE_TRAIN_ONLY_SCALERS = True
FINAL_REFIT_USES_EARLY_STOPPING = False
FINAL_REFIT_USES_VALIDATION_DATA = False


NEURAL_POPULATIONS = {
    15: {
        "train": 87_261,
        "validation": 17_568,
        "trainval": 104_829,
        "test": 26_496,
    },
    30: {
        "train": 87_258,
        "validation": 17_568,
        "trainval": 104_826,
        "test": 26_496,
    },
    60: {
        "train": 87_252,
        "validation": 17_568,
        "trainval": 104_820,
        "test": 26_496,
    },
}


# ---------------------------------------------------------------------
# Recurrent architecture grid
# ---------------------------------------------------------------------

RECURRENT_ARCHITECTURE_CANDIDATES = (
    {
        "candidate_id": "RNN_C1",
        "units": 32,
        "n_recurrent_layers": 1,
        "dropout": 0.0,
        "fusion_dense_units": 32,
    },
    {
        "candidate_id": "RNN_C2",
        "units": 64,
        "n_recurrent_layers": 1,
        "dropout": 0.0,
        "fusion_dense_units": 32,
    },
    {
        "candidate_id": "RNN_C3",
        "units": 64,
        "n_recurrent_layers": 1,
        "dropout": 0.20,
        "fusion_dense_units": 32,
    },
    {
        "candidate_id": "RNN_C4",
        "units": 64,
        "n_recurrent_layers": 2,
        "dropout": 0.20,
        "fusion_dense_units": 32,
    },
)


SELECTED_RECURRENT_ARCHITECTURES = {
    "LSTM": {
        "candidate_id": "RNN_C1",
        "units": 32,
        "n_recurrent_layers": 1,
        "dropout": 0.0,
        "fusion_dense_units": 32,
    },
    "GRU": {
        "candidate_id": "RNN_C2",
        "units": 64,
        "n_recurrent_layers": 1,
        "dropout": 0.0,
        "fusion_dense_units": 32,
    },
    "BiLSTM": {
        "candidate_id": "RNN_C2",
        "units": 64,
        "n_recurrent_layers": 1,
        "dropout": 0.0,
        "fusion_dense_units": 32,
    },
}


# ---------------------------------------------------------------------
# Executed/frozen TCN architecture grid
# ---------------------------------------------------------------------


def compute_tcn_receptive_field(
    kernel_size: int,
    dilations,
    convolutions_per_block: int,
) -> int:
    """Compute the frozen causal-TCN receptive field."""

    kernel_size = int(kernel_size)
    convolutions_per_block = int(
        convolutions_per_block
    )
    dilations = tuple(
        int(value)
        for value in dilations
    )

    if kernel_size < 1:
        raise ValueError(
            "kernel_size must be positive."
        )

    if convolutions_per_block < 1:
        raise ValueError(
            "convolutions_per_block must be positive."
        )

    if not dilations or any(
        value < 1
        for value in dilations
    ):
        raise ValueError(
            "dilations must contain positive integers."
        )

    return int(
        1
        + convolutions_per_block
        * (kernel_size - 1)
        * sum(dilations)
    )


TCN_ARCHITECTURE_CANDIDATES = (
    {
        "candidate_id": "TCN_C1",
        "filters": 32,
        "kernel_size": 3,
        "dilations": (1, 2, 4),
        "convolutions_per_block": 2,
        "dropout": 0.0,
        "fusion_dense_units": 32,
        "receptive_field_timesteps": 29,
    },
    {
        "candidate_id": "TCN_C2",
        "filters": 64,
        "kernel_size": 3,
        "dilations": (1, 2, 4),
        "convolutions_per_block": 2,
        "dropout": 0.0,
        "fusion_dense_units": 32,
        "receptive_field_timesteps": 29,
    },
    {
        "candidate_id": "TCN_C3",
        "filters": 64,
        "kernel_size": 2,
        "dilations": (1, 2, 4),
        "convolutions_per_block": 2,
        "dropout": 0.0,
        "fusion_dense_units": 32,
        "receptive_field_timesteps": 15,
    },
    {
        "candidate_id": "TCN_C4",
        "filters": 64,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8),
        "convolutions_per_block": 2,
        "dropout": 0.0,
        "fusion_dense_units": 32,
        "receptive_field_timesteps": 61,
    },
)


for _tcn_config in TCN_ARCHITECTURE_CANDIDATES:
    _computed_receptive_field = (
        compute_tcn_receptive_field(
            kernel_size=_tcn_config[
                "kernel_size"
            ],
            dilations=_tcn_config[
                "dilations"
            ],
            convolutions_per_block=(
                _tcn_config[
                    "convolutions_per_block"
                ]
            ),
        )
    )

    if _computed_receptive_field != (
        _tcn_config[
            "receptive_field_timesteps"
        ]
    ):
        raise RuntimeError(
            "Frozen TCN receptive-field "
            "definition is inconsistent."
        )

    if _computed_receptive_field < (
        SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Frozen TCN candidate does not "
            "cover the complete sequence."
        )


SELECTED_TCN_ARCHITECTURE = {
    "candidate_id": "TCN_C1",
    "filters": 32,
    "kernel_size": 3,
    "dilations": (1, 2, 4),
    "convolutions_per_block": 2,
    "dropout": 0.0,
    "fusion_dense_units": 32,
    "receptive_field_timesteps": 29,
}


# ---------------------------------------------------------------------
# Frozen validation-selected epoch counts
# ---------------------------------------------------------------------

LSTM_FROZEN_EPOCHS = {
    (15, "F0"): 7,
    (15, "F1"): 1,
    (15, "F2"): 2,
    (15, "F3"): 2,
    (15, "F4"): 7,
    (30, "F0"): 10,
    (30, "F1"): 1,
    (30, "F2"): 6,
    (30, "F3"): 5,
    (30, "F4"): 1,
    (60, "F0"): 10,
    (60, "F1"): 4,
    (60, "F2"): 1,
    (60, "F3"): 3,
    (60, "F4"): 2,
}

GRU_FROZEN_EPOCHS = {
    (15, "F0"): 3,
    (15, "F1"): 4,
    (15, "F2"): 7,
    (15, "F3"): 2,
    (15, "F4"): 4,
    (30, "F0"): 4,
    (30, "F1"): 11,
    (30, "F2"): 2,
    (30, "F3"): 5,
    (30, "F4"): 4,
    (60, "F0"): 3,
    (60, "F1"): 13,
    (60, "F2"): 3,
    (60, "F3"): 5,
    (60, "F4"): 1,
}

BILSTM_FROZEN_EPOCHS = {
    (15, "F0"): 6,
    (15, "F1"): 5,
    (15, "F2"): 2,
    (15, "F3"): 4,
    (15, "F4"): 2,
    (30, "F0"): 16,
    (30, "F1"): 7,
    (30, "F2"): 3,
    (30, "F3"): 17,
    (30, "F4"): 3,
    (60, "F0"): 6,
    (60, "F1"): 6,
    (60, "F2"): 6,
    (60, "F3"): 5,
    (60, "F4"): 4,
}

TCN_FROZEN_EPOCHS = {
    (15, "F0"): 8,
    (15, "F1"): 4,
    (15, "F2"): 9,
    (15, "F3"): 16,
    (15, "F4"): 4,
    (30, "F0"): 5,
    (30, "F1"): 9,
    (30, "F2"): 5,
    (30, "F3"): 12,
    (30, "F4"): 1,
    (60, "F0"): 10,
    (60, "F1"): 7,
    (60, "F2"): 9,
    (60, "F3"): 9,
    (60, "F4"): 1,
}


FROZEN_EPOCHS_BY_MODEL = {
    "LSTM": LSTM_FROZEN_EPOCHS,
    "GRU": GRU_FROZEN_EPOCHS,
    "BiLSTM": BILSTM_FROZEN_EPOCHS,
    "TCN": TCN_FROZEN_EPOCHS,
}


# ---------------------------------------------------------------------
# Expected frozen trainable parameter counts
# ---------------------------------------------------------------------

EXPECTED_PARAMETER_COUNTS = {
    "LSTM": {
        "F0": 6_497,
        "F1": 6_977,
        "F2": 7_137,
        "F3": 7_425,
        "F4": 10_529,
    },
    "GRU": {
        "F0": 16_545,
        "F1": 17_025,
        "F2": 17_185,
        "F3": 17_473,
        "F4": 20_577,
    },
    "BiLSTM": {
        "F0": 42_081,
        "F1": 42_561,
        "F2": 42_721,
        "F3": 43_009,
        "F4": 46_113,
    },
    "TCN": {
        "F0": 17_857,
        "F1": 18_337,
        "F2": 18_497,
        "F3": 18_785,
        "F4": 21_889,
    },
}


# ---------------------------------------------------------------------
# Shared model-state / validation helpers
# ---------------------------------------------------------------------


def reset_neural_state(
    seed: int = NEURAL_RANDOM_SEED,
) -> None:
    """Reset the frozen Python, NumPy, and TensorFlow RNG state."""

    seed = int(seed)

    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(
        seed
    )


def _require_config_keys(
    config: Mapping,
    required,
) -> None:
    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:
        raise ValueError(
            f"Missing configuration keys: {missing}."
        )


def _validate_auxiliary_dim(
    auxiliary_dim: int,
) -> int:
    auxiliary_dim = int(
        auxiliary_dim
    )

    if auxiliary_dim not in set(
        AUXILIARY_DIMS.values()
    ):
        raise ValueError(
            "auxiliary_dim must match one of "
            "the frozen F0-F4 widths."
        )

    return auxiliary_dim


# ---------------------------------------------------------------------
# Recurrent model implementation
# ---------------------------------------------------------------------


def build_recurrent_sequence_branch(
    sequence_input,
    model_name: str,
    candidate_config: Mapping,
):
    """Build the frozen LSTM, GRU, or BiLSTM history encoder."""

    if model_name not in {
        "LSTM",
        "GRU",
        "BiLSTM",
    }:
        raise ValueError(
            "model_name must be LSTM, GRU, or BiLSTM."
        )

    _require_config_keys(
        candidate_config,
        (
            "units",
            "n_recurrent_layers",
            "dropout",
        ),
    )

    units = int(
        candidate_config[
            "units"
        ]
    )

    n_recurrent_layers = int(
        candidate_config[
            "n_recurrent_layers"
        ]
    )

    dropout = float(
        candidate_config[
            "dropout"
        ]
    )

    if n_recurrent_layers not in {
        1,
        2,
    }:
        raise ValueError(
            "Frozen recurrent architecture allows "
            "one or two recurrent layers."
        )

    x = sequence_input

    for layer_index in range(
        n_recurrent_layers
    ):
        is_last_layer = (
            layer_index
            == n_recurrent_layers - 1
        )

        return_sequences = (
            not is_last_layer
        )

        recurrent_layer_name = (
            f"{model_name.lower()}_"
            f"{layer_index + 1}"
        )

        if model_name == "LSTM":
            x = tf.keras.layers.LSTM(
                units=units,
                return_sequences=(
                    return_sequences
                ),
                dropout=dropout,
                recurrent_dropout=0.0,
                name=recurrent_layer_name,
            )(
                x
            )

        elif model_name == "GRU":
            x = tf.keras.layers.GRU(
                units=units,
                return_sequences=(
                    return_sequences
                ),
                dropout=dropout,
                recurrent_dropout=0.0,
                name=recurrent_layer_name,
            )(
                x
            )

        else:
            forward_lstm = (
                tf.keras.layers.LSTM(
                    units=units,
                    return_sequences=(
                        return_sequences
                    ),
                    dropout=dropout,
                    recurrent_dropout=0.0,
                    name=(
                        "bilstm_forward_"
                        f"{layer_index + 1}"
                    ),
                )
            )

            x = (
                tf.keras.layers.Bidirectional(
                    forward_lstm,
                    merge_mode="concat",
                    name=recurrent_layer_name,
                )(
                    x
                )
            )

    return x


def build_recurrent_fusion_model(
    model_name: str,
    auxiliary_dim: int,
    candidate_config: Mapping,
    *,
    seed: int = NEURAL_RANDOM_SEED,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Build one frozen-protocol recurrent two-input model."""

    if model_name not in {
        "LSTM",
        "GRU",
        "BiLSTM",
    }:
        raise ValueError(
            "model_name must be LSTM, GRU, or BiLSTM."
        )

    auxiliary_dim = (
        _validate_auxiliary_dim(
            auxiliary_dim
        )
    )

    _require_config_keys(
        candidate_config,
        (
            "candidate_id",
            "units",
            "n_recurrent_layers",
            "dropout",
            "fusion_dense_units",
        ),
    )

    reset_neural_state(
        seed=seed
    )

    sequence_input = tf.keras.layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            len(
                SEQUENCE_CHANNELS
            ),
        ),
        dtype=tf.float32,
        name="sequence_input",
    )

    sequence_embedding = (
        build_recurrent_sequence_branch(
            sequence_input=sequence_input,
            model_name=model_name,
            candidate_config=(
                candidate_config
            ),
        )
    )

    auxiliary_input = tf.keras.layers.Input(
        shape=(
            auxiliary_dim,
        ),
        dtype=tf.float32,
        name="auxiliary_input",
    )

    fused = tf.keras.layers.Concatenate(
        name="sequence_auxiliary_fusion"
    )(
        [
            sequence_embedding,
            auxiliary_input,
        ]
    )

    fused = tf.keras.layers.Dense(
        units=int(
            candidate_config[
                "fusion_dense_units"
            ]
        ),
        activation="relu",
        name="fusion_dense",
    )(
        fused
    )

    output = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        name="scaled_p_solar_output",
    )(
        fused
    )

    model = tf.keras.Model(
        inputs=[
            sequence_input,
            auxiliary_input,
        ],
        outputs=output,
        name=(
            f"{model_name.lower()}_"
            f"{candidate_config['candidate_id'].lower()}"
        ),
    )

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=(
                    INITIAL_LEARNING_RATE
                ),
                clipnorm=ADAM_CLIPNORM,
            ),
            loss=LOSS,
        )

    return model


# ---------------------------------------------------------------------
# Executed/frozen TCN implementation
# ---------------------------------------------------------------------


def apply_tcn_backbone(
    sequence_tensor,
    config: Mapping,
    *,
    name_prefix: str = "tcn",
):
    """Apply the executed causal residual TCN backbone."""

    _require_config_keys(
        config,
        (
            "filters",
            "kernel_size",
            "dilations",
            "convolutions_per_block",
            "dropout",
        ),
    )

    x = sequence_tensor

    filters = int(
        config[
            "filters"
        ]
    )

    kernel_size = int(
        config[
            "kernel_size"
        ]
    )

    dropout_rate = float(
        config[
            "dropout"
        ]
    )

    convolutions_per_block = int(
        config[
            "convolutions_per_block"
        ]
    )

    if convolutions_per_block != 2:
        raise ValueError(
            "Executed frozen TCN requires exactly "
            "two convolutions per residual block."
        )

    for block_index, dilation_rate in enumerate(
        config[
            "dilations"
        ],
        start=1,
    ):
        residual = x

        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="causal",
            dilation_rate=int(
                dilation_rate
            ),
            activation="relu",
            kernel_initializer="he_normal",
            name=(
                f"{name_prefix}_block"
                f"{block_index}_conv1"
            ),
        )(
            x
        )

        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(
                rate=dropout_rate,
                name=(
                    f"{name_prefix}_block"
                    f"{block_index}_dropout1"
                ),
            )(
                x
            )

        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="causal",
            dilation_rate=int(
                dilation_rate
            ),
            activation="relu",
            kernel_initializer="he_normal",
            name=(
                f"{name_prefix}_block"
                f"{block_index}_conv2"
            ),
        )(
            x
        )

        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(
                rate=dropout_rate,
                name=(
                    f"{name_prefix}_block"
                    f"{block_index}_dropout2"
                ),
            )(
                x
            )

        residual_channels = int(
            residual.shape[
                -1
            ]
        )

        if residual_channels != filters:
            residual = (
                tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding="causal",
                    name=(
                        f"{name_prefix}_block"
                        f"{block_index}_projection"
                    ),
                )(
                    residual
                )
            )

        x = tf.keras.layers.Add(
            name=(
                f"{name_prefix}_block"
                f"{block_index}_add"
            )
        )(
            [
                x,
                residual,
            ]
        )

        x = tf.keras.layers.Activation(
            "relu",
            name=(
                f"{name_prefix}_block"
                f"{block_index}_relu"
            ),
        )(
            x
        )

    return x


def build_tcn_fusion_model(
    config: Mapping,
    auxiliary_dim: int,
    *,
    compile_model: bool = True,
) -> tf.keras.Model:
    """
    Build the later executed/frozen TCN forecasting model.

    Forecasting uses the representation at the final historical
    position, not global average pooling.
    """

    auxiliary_dim = int(
        auxiliary_dim
    )

    if auxiliary_dim < 1:
        raise ValueError(
            "auxiliary_dim must be positive."
        )

    _require_config_keys(
        config,
        (
            "candidate_id",
            "filters",
            "kernel_size",
            "dilations",
            "convolutions_per_block",
            "dropout",
            "fusion_dense_units",
        ),
    )

    sequence_input = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            len(
                SEQUENCE_CHANNELS
            ),
        ),
        name="historical_sequence",
    )

    auxiliary_input = tf.keras.Input(
        shape=(
            auxiliary_dim,
        ),
        name="auxiliary_features",
    )

    temporal_representation = (
        apply_tcn_backbone(
            sequence_tensor=sequence_input,
            config=config,
            name_prefix="tcn",
        )
    )

    last_temporal_position = (
        tf.keras.layers.Lambda(
            lambda tensor:
                tensor[
                    :,
                    -1,
                    :
                ],
            name="tcn_last_position",
        )(
            temporal_representation
        )
    )

    fused = tf.keras.layers.Concatenate(
        name="tcn_auxiliary_fusion"
    )(
        [
            last_temporal_position,
            auxiliary_input,
        ]
    )

    fused = tf.keras.layers.Dense(
        units=int(
            config[
                "fusion_dense_units"
            ]
        ),
        activation="relu",
        name="fusion_dense",
    )(
        fused
    )

    forecast_output = (
        tf.keras.layers.Dense(
            units=1,
            activation="linear",
            name=(
                "standardized_target_"
                "forecast"
            ),
        )(
            fused
        )
    )

    model = tf.keras.Model(
        inputs=[
            sequence_input,
            auxiliary_input,
        ],
        outputs=forecast_output,
        name=(
            "TCN_"
            + str(
                config[
                    "candidate_id"
                ]
            )
        ),
    )

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=(
                    INITIAL_LEARNING_RATE
                ),
                clipnorm=ADAM_CLIPNORM,
            ),
            loss=LOSS,
        )

    return model


# ---------------------------------------------------------------------
# Frozen model dispatcher
# ---------------------------------------------------------------------


def build_frozen_model(
    model_name: str,
    feature_family: str,
    *,
    seed: int = NEURAL_RANDOM_SEED,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Build the validation-selected architecture for one family."""

    if feature_family not in (
        AUXILIARY_DIMS
    ):
        raise ValueError(
            f"Unknown feature family: {feature_family}."
        )

    auxiliary_dim = (
        AUXILIARY_DIMS[
            feature_family
        ]
    )

    if model_name in (
        SELECTED_RECURRENT_ARCHITECTURES
    ):
        return build_recurrent_fusion_model(
            model_name=model_name,
            auxiliary_dim=auxiliary_dim,
            candidate_config=(
                SELECTED_RECURRENT_ARCHITECTURES[
                    model_name
                ]
            ),
            seed=seed,
            compile_model=compile_model,
        )

    if model_name == "TCN":
        reset_neural_state(
            seed=seed
        )

        return build_tcn_fusion_model(
            config=(
                SELECTED_TCN_ARCHITECTURE
            ),
            auxiliary_dim=auxiliary_dim,
            compile_model=compile_model,
        )

    raise ValueError(
        f"Unknown frozen neural model: {model_name}."
    )


def get_frozen_epoch(
    model_name: str,
    horizon_minutes: int,
    feature_family: str,
) -> int:
    """Return the validation-selected final-refit epoch count."""

    if model_name not in (
        FROZEN_EPOCHS_BY_MODEL
    ):
        raise ValueError(
            f"Unknown frozen neural model: {model_name}."
        )

    key = (
        int(
            horizon_minutes
        ),
        str(
            feature_family
        ),
    )

    if key not in (
        FROZEN_EPOCHS_BY_MODEL[
            model_name
        ]
    ):
        raise ValueError(
            "Unknown frozen horizon / feature-family "
            f"combination: {key}."
        )

    return int(
        FROZEN_EPOCHS_BY_MODEL[
            model_name
        ][
            key
        ]
    )


# ---------------------------------------------------------------------
# Architecture selection
# ---------------------------------------------------------------------


def rank_architecture_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the frozen neural architecture-selection ordering.

    1. minimum validation MAE in kW
    2. minimum validation RMSE in kW
    3. fewer trainable parameters
    """

    if not isinstance(
        results,
        pd.DataFrame,
    ):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    if results.empty:
        raise ValueError(
            "results must contain at least one candidate."
        )

    ranking_columns = [
        "validation_MAE_kW",
        "validation_RMSE_kW",
        "trainable_parameter_count",
    ]

    missing = [
        column
        for column in ranking_columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ranking columns: {missing}."
        )

    return (
        results
        .sort_values(
            ranking_columns,
            ascending=True,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


def make_early_stopping_callback():
    """Create the frozen validation-only EarlyStopping callback."""

    return tf.keras.callbacks.EarlyStopping(
        monitor=EARLY_STOP_MONITOR,
        mode=EARLY_STOP_MODE,
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        restore_best_weights=(
            RESTORE_BEST_WEIGHTS
        ),
        verbose=0,
    )


# ---------------------------------------------------------------------
# Train-only scaling
# ---------------------------------------------------------------------


def _validate_positions(
    n_columns: int,
    continuous_positions,
    binary_positions,
) -> tuple[np.ndarray, np.ndarray]:
    continuous = np.asarray(
        continuous_positions,
        dtype=np.int64,
    ).reshape(
        -1
    )

    binary = np.asarray(
        binary_positions,
        dtype=np.int64,
    ).reshape(
        -1
    )

    for name, positions in (
        (
            "continuous_positions",
            continuous,
        ),
        (
            "binary_positions",
            binary,
        ),
    ):
        if (
            positions.size
            and (
                positions.min() < 0
                or positions.max() >= n_columns
            )
        ):
            raise ValueError(
                f"{name} contains an out-of-range index."
            )

        if np.unique(
            positions
        ).size != positions.size:
            raise ValueError(
                f"{name} contains duplicates."
            )

    if np.intersect1d(
        continuous,
        binary,
    ).size:
        raise ValueError(
            "Continuous and binary positions overlap."
        )

    combined = np.sort(
        np.concatenate(
            [
                continuous,
                binary,
            ]
        )
    )

    expected = np.arange(
        n_columns,
        dtype=np.int64,
    )

    if not np.array_equal(
        combined,
        expected,
    ):
        raise ValueError(
            "Continuous and binary positions must "
            "partition the auxiliary columns exactly."
        )

    return (
        continuous,
        binary,
    )


def _validate_raw_neural_arrays(
    sequence,
    auxiliary,
    target,
):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    auxiliary = np.asarray(
        auxiliary
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    ).reshape(
        -1
    )

    if sequence.ndim != 3:
        raise ValueError(
            "sequence must be three-dimensional."
        )

    if sequence.shape[
        1:
    ] != (
        SEQUENCE_LENGTH,
        len(
            SEQUENCE_CHANNELS
        ),
    ):
        raise ValueError(
            "sequence must have shape "
            "(samples, 13, 9)."
        )

    if auxiliary.ndim != 2:
        raise ValueError(
            "auxiliary must be two-dimensional."
        )

    n_samples = (
        sequence.shape[
            0
        ]
    )

    if auxiliary.shape[
        0
    ] != n_samples:
        raise ValueError(
            "sequence and auxiliary sample counts differ."
        )

    if target.shape[
        0
    ] != n_samples:
        raise ValueError(
            "sequence and target sample counts differ."
        )

    if not np.isfinite(
        sequence
    ).all():
        raise ValueError(
            "sequence contains non-finite values."
        )

    if not np.isfinite(
        np.asarray(
            auxiliary,
            dtype=np.float64,
        )
    ).all():
        raise ValueError(
            "auxiliary contains non-finite values."
        )

    if not np.isfinite(
        target
    ).all():
        raise ValueError(
            "target contains non-finite values."
        )

    return (
        sequence,
        auxiliary,
        target,
    )


def fit_train_only_scalers(
    sequence,
    auxiliary,
    target,
    train_mask,
    *,
    continuous_positions,
    binary_positions,
) -> dict:
    """
    Fit the frozen neural scaling statistics on TRAIN rows only.

    Sequence:
        channel-wise mean/std pooled over training samples and timesteps.

    Auxiliary:
        continuous columns use training-only mean/std;
        binary columns stay literal 0/1.

    Target:
        training-only mean/std.

    Population standard deviation (ddof=0) is used throughout.
    """

    (
        sequence,
        auxiliary,
        target,
    ) = _validate_raw_neural_arrays(
        sequence,
        auxiliary,
        target,
    )

    train_mask = np.asarray(
        train_mask,
        dtype=bool,
    ).reshape(
        -1
    )

    if train_mask.shape[
        0
    ] != sequence.shape[
        0
    ]:
        raise ValueError(
            "train_mask length does not match samples."
        )

    if not train_mask.any():
        raise ValueError(
            "train_mask contains no training rows."
        )

    (
        continuous_positions,
        binary_positions,
    ) = _validate_positions(
        auxiliary.shape[
            1
        ],
        continuous_positions,
        binary_positions,
    )

    auxiliary_float64 = np.asarray(
        auxiliary,
        dtype=np.float64,
    )

    if binary_positions.size:
        binary_values = (
            auxiliary_float64[
                :,
                binary_positions,
            ]
        )

        if not np.isin(
            binary_values,
            [
                0.0,
                1.0,
            ],
        ).all():
            raise ValueError(
                "Binary auxiliary columns must "
                "contain only 0 and 1."
            )

    train_sequence_float64 = (
        sequence[
            train_mask
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    sequence_mean = (
        train_sequence_float64
        .mean(
            axis=(
                0,
                1,
            )
        )
    )

    sequence_std = (
        train_sequence_float64
        .std(
            axis=(
                0,
                1,
            ),
            ddof=SCALING_DDOF,
        )
    )

    if (
        not np.isfinite(
            sequence_std
        ).all()
        or not np.all(
            sequence_std > 0.0
        )
    ):
        raise ValueError(
            "Zero-variance or invalid sequence "
            "channel found."
        )

    train_auxiliary_continuous = (
        auxiliary_float64[
            train_mask
        ][
            :,
            continuous_positions,
        ]
    )

    auxiliary_mean = (
        train_auxiliary_continuous
        .mean(
            axis=0
        )
    )

    auxiliary_std = (
        train_auxiliary_continuous
        .std(
            axis=0,
            ddof=SCALING_DDOF,
        )
    )

    if (
        not np.isfinite(
            auxiliary_std
        ).all()
        or not np.all(
            auxiliary_std > 0.0
        )
    ):
        raise ValueError(
            "Zero-variance or invalid continuous "
            "auxiliary feature found."
        )

    train_target = (
        target[
            train_mask
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    target_mean = float(
        train_target.mean()
    )

    target_std = float(
        train_target.std(
            ddof=SCALING_DDOF
        )
    )

    if (
        not np.isfinite(
            target_std
        )
        or target_std <= 0.0
    ):
        raise ValueError(
            "Training target must have positive variance."
        )

    return {
        "sequence_mean":
            sequence_mean.copy(),
        "sequence_std":
            sequence_std.copy(),
        "auxiliary_mean":
            auxiliary_mean.copy(),
        "auxiliary_std":
            auxiliary_std.copy(),
        "target_mean":
            target_mean,
        "target_std":
            target_std,
        "continuous_positions":
            continuous_positions.copy(),
        "binary_positions":
            binary_positions.copy(),
        "ddof":
            SCALING_DDOF,
        "n_train":
            int(
                train_mask.sum()
            ),
    }


def apply_train_only_scalers(
    sequence,
    auxiliary,
    target,
    scaling_statistics: Mapping,
) -> dict:
    """Apply already-frozen train-only neural scaling statistics."""

    (
        sequence,
        auxiliary,
        target,
    ) = _validate_raw_neural_arrays(
        sequence,
        auxiliary,
        target,
    )

    required = (
        "sequence_mean",
        "sequence_std",
        "auxiliary_mean",
        "auxiliary_std",
        "target_mean",
        "target_std",
        "continuous_positions",
        "binary_positions",
    )

    _require_config_keys(
        scaling_statistics,
        required,
    )

    (
        continuous_positions,
        binary_positions,
    ) = _validate_positions(
        auxiliary.shape[
            1
        ],
        scaling_statistics[
            "continuous_positions"
        ],
        scaling_statistics[
            "binary_positions"
        ],
    )

    sequence_mean = np.asarray(
        scaling_statistics[
            "sequence_mean"
        ],
        dtype=np.float64,
    )

    sequence_std = np.asarray(
        scaling_statistics[
            "sequence_std"
        ],
        dtype=np.float64,
    )

    auxiliary_mean = np.asarray(
        scaling_statistics[
            "auxiliary_mean"
        ],
        dtype=np.float64,
    )

    auxiliary_std = np.asarray(
        scaling_statistics[
            "auxiliary_std"
        ],
        dtype=np.float64,
    )

    if sequence_mean.shape != (
        len(
            SEQUENCE_CHANNELS
        ),
    ):
        raise ValueError(
            "Invalid sequence_mean shape."
        )

    if sequence_std.shape != (
        len(
            SEQUENCE_CHANNELS
        ),
    ):
        raise ValueError(
            "Invalid sequence_std shape."
        )

    if auxiliary_mean.shape != (
        continuous_positions.size,
    ):
        raise ValueError(
            "Invalid auxiliary_mean shape."
        )

    if auxiliary_std.shape != (
        continuous_positions.size,
    ):
        raise ValueError(
            "Invalid auxiliary_std shape."
        )

    if not np.all(
        sequence_std > 0.0
    ):
        raise ValueError(
            "sequence_std must be positive."
        )

    if not np.all(
        auxiliary_std > 0.0
    ):
        raise ValueError(
            "auxiliary_std must be positive."
        )

    target_mean = float(
        scaling_statistics[
            "target_mean"
        ]
    )

    target_std = float(
        scaling_statistics[
            "target_std"
        ]
    )

    if (
        not np.isfinite(
            target_std
        )
        or target_std <= 0.0
    ):
        raise ValueError(
            "target_std must be positive."
        )

    scaled_sequence = (
        (
            sequence.astype(
                np.float64
            )
            - sequence_mean
        )
        / sequence_std
    ).astype(
        np.float32
    )

    raw_auxiliary = np.asarray(
        auxiliary,
        dtype=np.float32,
    )

    scaled_auxiliary = (
        raw_auxiliary.copy()
    )

    raw_continuous_float64 = (
        raw_auxiliary[
            :,
            continuous_positions,
        ]
        .astype(
            np.float64
        )
    )

    scaled_auxiliary[
        :,
        continuous_positions,
    ] = (
        (
            raw_continuous_float64
            - auxiliary_mean
        )
        / auxiliary_std
    ).astype(
        np.float32
    )

    if binary_positions.size:
        if not np.array_equal(
            scaled_auxiliary[
                :,
                binary_positions,
            ],
            raw_auxiliary[
                :,
                binary_positions,
            ],
        ):
            raise RuntimeError(
                "Binary auxiliary values changed "
                "during scaling."
            )

    scaled_target = (
        (
            target.astype(
                np.float64
            )
            - target_mean
        )
        / target_std
    ).astype(
        np.float32
    )

    if not np.isfinite(
        scaled_sequence
    ).all():
        raise ValueError(
            "Scaled sequence contains non-finite values."
        )

    if not np.isfinite(
        scaled_auxiliary
    ).all():
        raise ValueError(
            "Scaled auxiliary data contains "
            "non-finite values."
        )

    if not np.isfinite(
        scaled_target
    ).all():
        raise ValueError(
            "Scaled target contains non-finite values."
        )

    return {
        "sequence":
            np.ascontiguousarray(
                scaled_sequence,
                dtype=np.float32,
            ),
        "auxiliary":
            np.ascontiguousarray(
                scaled_auxiliary,
                dtype=np.float32,
            ),
        "target_scaled":
            np.ascontiguousarray(
                scaled_target,
                dtype=np.float32,
            ),
        "target_kw":
            np.ascontiguousarray(
                target,
                dtype=np.float32,
            ),
    }


def inverse_target_scaling(
    scaled_values,
    scaling_statistics: Mapping,
) -> np.ndarray:
    """Inverse-transform standardized predictions to physical kW."""

    _require_config_keys(
        scaling_statistics,
        (
            "target_mean",
            "target_std",
        ),
    )

    scaled_values = np.asarray(
        scaled_values,
        dtype=np.float64,
    )

    if not np.isfinite(
        scaled_values
    ).all():
        raise ValueError(
            "scaled_values contains non-finite values."
        )

    target_mean = float(
        scaling_statistics[
            "target_mean"
        ]
    )

    target_std = float(
        scaling_statistics[
            "target_std"
        ]
    )

    if target_std <= 0.0:
        raise ValueError(
            "target_std must be positive."
        )

    values_kw = (
        scaled_values
        * target_std
        + target_mean
    )

    if not np.isfinite(
        values_kw
    ).all():
        raise ValueError(
            "Inverse-transformed values are non-finite."
        )

    return values_kw


# ---------------------------------------------------------------------
# Final-refit population assembly
# ---------------------------------------------------------------------


def combine_train_validation(
    train_data: Mapping,
    validation_data: Mapping,
) -> dict:
    """
    Concatenate TRAIN followed by VALIDATION for final refitting.

    Input arrays are assumed to have already been transformed using
    the original frozen TRAIN-only scaling statistics.
    """

    required = (
        "sequence",
        "auxiliary",
        "target_scaled",
    )

    _require_config_keys(
        train_data,
        required,
    )

    _require_config_keys(
        validation_data,
        required,
    )

    train_sequence = np.asarray(
        train_data[
            "sequence"
        ],
        dtype=np.float32,
    )

    validation_sequence = np.asarray(
        validation_data[
            "sequence"
        ],
        dtype=np.float32,
    )

    train_auxiliary = np.asarray(
        train_data[
            "auxiliary"
        ],
        dtype=np.float32,
    )

    validation_auxiliary = np.asarray(
        validation_data[
            "auxiliary"
        ],
        dtype=np.float32,
    )

    train_target = np.asarray(
        train_data[
            "target_scaled"
        ],
        dtype=np.float32,
    )

    validation_target = np.asarray(
        validation_data[
            "target_scaled"
        ],
        dtype=np.float32,
    )

    if train_sequence.shape[
        1:
    ] != (
        SEQUENCE_LENGTH,
        len(
            SEQUENCE_CHANNELS
        ),
    ):
        raise ValueError(
            "Invalid TRAIN sequence shape."
        )

    if validation_sequence.shape[
        1:
    ] != train_sequence.shape[
        1:
    ]:
        raise ValueError(
            "TRAIN and VALIDATION sequence shapes differ."
        )

    if train_auxiliary.ndim != 2:
        raise ValueError(
            "TRAIN auxiliary array must be two-dimensional."
        )

    if validation_auxiliary.shape[
        1:
    ] != train_auxiliary.shape[
        1:
    ]:
        raise ValueError(
            "TRAIN and VALIDATION auxiliary widths differ."
        )

    if train_target.shape[
        1:
    ] != validation_target.shape[
        1:
    ]:
        raise ValueError(
            "TRAIN and VALIDATION target shapes differ."
        )

    if train_sequence.shape[
        0
    ] != train_auxiliary.shape[
        0
    ]:
        raise ValueError(
            "TRAIN sample counts are inconsistent."
        )

    if train_sequence.shape[
        0
    ] != train_target.shape[
        0
    ]:
        raise ValueError(
            "TRAIN target count is inconsistent."
        )

    if validation_sequence.shape[
        0
    ] != validation_auxiliary.shape[
        0
    ]:
        raise ValueError(
            "VALIDATION sample counts are inconsistent."
        )

    if validation_sequence.shape[
        0
    ] != validation_target.shape[
        0
    ]:
        raise ValueError(
            "VALIDATION target count is inconsistent."
        )

    sequence = np.concatenate(
        [
            train_sequence,
            validation_sequence,
        ],
        axis=0,
    )

    auxiliary = np.concatenate(
        [
            train_auxiliary,
            validation_auxiliary,
        ],
        axis=0,
    )

    target_scaled = np.concatenate(
        [
            train_target,
            validation_target,
        ],
        axis=0,
    )

    return {
        "sequence":
            np.ascontiguousarray(
                sequence,
                dtype=np.float32,
            ),
        "auxiliary":
            np.ascontiguousarray(
                auxiliary,
                dtype=np.float32,
            ),
        "target_scaled":
            np.ascontiguousarray(
                target_scaled,
                dtype=np.float32,
            ),
        "n_train":
            int(
                train_sequence.shape[
                    0
                ]
            ),
        "n_validation":
            int(
                validation_sequence.shape[
                    0
                ]
            ),
        "n_trainval":
            int(
                sequence.shape[
                    0
                ]
            ),
    }

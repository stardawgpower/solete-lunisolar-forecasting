import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from neural_models import (  # noqa: E402
    ADAM_CLIPNORM,
    AUXILIARY_DIMS,
    BATCH_SIZE,
    BILSTM_FROZEN_EPOCHS,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    EXPECTED_PARAMETER_COUNTS,
    FEATURE_FAMILIES,
    FORECAST_HORIZONS,
    FROZEN_EPOCHS_BY_MODEL,
    GRU_FROZEN_EPOCHS,
    INITIAL_LEARNING_RATE,
    LOSS,
    LSTM_FROZEN_EPOCHS,
    MAX_VALIDATION_EPOCHS,
    NEURAL_POPULATIONS,
    NEURAL_RANDOM_SEED,
    RECURRENT_ARCHITECTURE_CANDIDATES,
    SELECTED_RECURRENT_ARCHITECTURES,
    SELECTED_TCN_ARCHITECTURE,
    SEQUENCE_CHANNELS,
    SEQUENCE_LENGTH,
    SHUFFLE_TRAINING_SAMPLES,
    TCN_ARCHITECTURE_CANDIDATES,
    TCN_FROZEN_EPOCHS,
    apply_train_only_scalers,
    build_frozen_model,
    combine_train_validation,
    compute_tcn_receptive_field,
    fit_train_only_scalers,
    get_frozen_epoch,
    inverse_target_scaling,
    make_early_stopping_callback,
    rank_architecture_results,
)


def test_frozen_neural_contract():
    assert NEURAL_RANDOM_SEED == 42
    assert SEQUENCE_LENGTH == 13
    assert len(SEQUENCE_CHANNELS) == 9

    assert FORECAST_HORIZONS == (
        15,
        30,
        60,
    )

    assert FEATURE_FAMILIES == (
        "F0",
        "F1",
        "F2",
        "F3",
        "F4",
    )

    assert AUXILIARY_DIMS == {
        "F0": 1,
        "F1": 16,
        "F2": 21,
        "F3": 30,
        "F4": 127,
    }

    assert BATCH_SIZE == 256
    assert MAX_VALIDATION_EPOCHS == 80
    assert INITIAL_LEARNING_RATE == pytest.approx(
        1e-3
    )
    assert ADAM_CLIPNORM == pytest.approx(
        1.0
    )
    assert LOSS == "mae"
    assert EARLY_STOP_PATIENCE == 8
    assert EARLY_STOP_MIN_DELTA == pytest.approx(
        1e-4
    )
    assert SHUFFLE_TRAINING_SAMPLES is True

    assert NEURAL_POPULATIONS[
        15
    ] == {
        "train": 87_261,
        "validation": 17_568,
        "trainval": 104_829,
        "test": 26_496,
    }

    assert NEURAL_POPULATIONS[
        60
    ] == {
        "train": 87_252,
        "validation": 17_568,
        "trainval": 104_820,
        "test": 26_496,
    }


def test_recurrent_candidate_grid_is_frozen():
    assert len(
        RECURRENT_ARCHITECTURE_CANDIDATES
    ) == 4

    assert [
        config[
            "candidate_id"
        ]
        for config
        in RECURRENT_ARCHITECTURE_CANDIDATES
    ] == [
        "RNN_C1",
        "RNN_C2",
        "RNN_C3",
        "RNN_C4",
    ]

    assert (
        RECURRENT_ARCHITECTURE_CANDIDATES[
            0
        ]
        == {
            "candidate_id": "RNN_C1",
            "units": 32,
            "n_recurrent_layers": 1,
            "dropout": 0.0,
            "fusion_dense_units": 32,
        }
    )

    assert (
        RECURRENT_ARCHITECTURE_CANDIDATES[
            3
        ][
            "n_recurrent_layers"
        ]
        == 2
    )


def test_selected_recurrent_architectures_are_frozen():
    assert (
        SELECTED_RECURRENT_ARCHITECTURES[
            "LSTM"
        ][
            "candidate_id"
        ]
        == "RNN_C1"
    )

    assert (
        SELECTED_RECURRENT_ARCHITECTURES[
            "LSTM"
        ][
            "units"
        ]
        == 32
    )

    for model_name in (
        "GRU",
        "BiLSTM",
    ):
        assert (
            SELECTED_RECURRENT_ARCHITECTURES[
                model_name
            ][
                "candidate_id"
            ]
            == "RNN_C2"
        )

        assert (
            SELECTED_RECURRENT_ARCHITECTURES[
                model_name
            ][
                "units"
            ]
            == 64
        )


def test_executed_tcn_grid_is_frozen():
    assert len(
        TCN_ARCHITECTURE_CANDIDATES
    ) == 4

    assert [
        config[
            "candidate_id"
        ]
        for config
        in TCN_ARCHITECTURE_CANDIDATES
    ] == [
        "TCN_C1",
        "TCN_C2",
        "TCN_C3",
        "TCN_C4",
    ]

    assert SELECTED_TCN_ARCHITECTURE == {
        "candidate_id": "TCN_C1",
        "filters": 32,
        "kernel_size": 3,
        "dilations": (
            1,
            2,
            4,
        ),
        "convolutions_per_block": 2,
        "dropout": 0.0,
        "fusion_dense_units": 32,
        "receptive_field_timesteps": 29,
    }

    observed_receptive_fields = [
        config[
            "receptive_field_timesteps"
        ]
        for config
        in TCN_ARCHITECTURE_CANDIDATES
    ]

    assert observed_receptive_fields == [
        29,
        29,
        15,
        61,
    ]

    for config in (
        TCN_ARCHITECTURE_CANDIDATES
    ):
        assert (
            compute_tcn_receptive_field(
                config[
                    "kernel_size"
                ],
                config[
                    "dilations"
                ],
                config[
                    "convolutions_per_block"
                ],
            )
            ==
            config[
                "receptive_field_timesteps"
            ]
        )

        assert (
            config[
                "receptive_field_timesteps"
            ]
            >= SEQUENCE_LENGTH
        )


def test_frozen_epoch_maps_are_complete_and_exact():
    expected_keys = {
        (
            horizon,
            family,
        )
        for horizon in FORECAST_HORIZONS
        for family in FEATURE_FAMILIES
    }

    for epochs in (
        LSTM_FROZEN_EPOCHS,
        GRU_FROZEN_EPOCHS,
        BILSTM_FROZEN_EPOCHS,
        TCN_FROZEN_EPOCHS,
    ):
        assert set(
            epochs
        ) == expected_keys

        assert len(
            epochs
        ) == 15

    assert LSTM_FROZEN_EPOCHS[
        (
            15,
            "F0",
        )
    ] == 7

    assert LSTM_FROZEN_EPOCHS[
        (
            60,
            "F4",
        )
    ] == 2

    assert GRU_FROZEN_EPOCHS[
        (
            30,
            "F1",
        )
    ] == 11

    assert BILSTM_FROZEN_EPOCHS[
        (
            30,
            "F3",
        )
    ] == 17

    assert TCN_FROZEN_EPOCHS[
        (
            15,
            "F3",
        )
    ] == 16

    assert TCN_FROZEN_EPOCHS[
        (
            60,
            "F4",
        )
    ] == 1


def _assert_parameter_counts(
    model_name,
):
    for family in FEATURE_FAMILIES:
        model = build_frozen_model(
            model_name,
            family,
            compile_model=False,
        )

        assert int(
            model.count_params()
        ) == (
            EXPECTED_PARAMETER_COUNTS[
                model_name
            ][
                family
            ]
        )


def test_lstm_parameter_counts_match_frozen_protocol():
    _assert_parameter_counts(
        "LSTM"
    )


def test_gru_parameter_counts_match_frozen_protocol():
    _assert_parameter_counts(
        "GRU"
    )


def test_bilstm_parameter_counts_match_frozen_protocol():
    _assert_parameter_counts(
        "BiLSTM"
    )


def test_tcn_parameter_counts_match_frozen_protocol():
    _assert_parameter_counts(
        "TCN"
    )


def test_recurrent_model_structure_and_compile():
    model = build_frozen_model(
        "LSTM",
        "F0",
        compile_model=True,
    )

    assert len(
        model.inputs
    ) == 2

    lstm_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.LSTM,
        )
    ]

    assert len(
        lstm_layers
    ) == 1

    assert lstm_layers[
        0
    ].units == 32

    assert lstm_layers[
        0
    ].return_sequences is False

    assert model.loss == "mae"

    assert isinstance(
        model.optimizer,
        tf.keras.optimizers.Adam,
    )

    learning_rate = float(
        tf.keras.backend.get_value(
            model.optimizer.learning_rate
        )
    )

    assert learning_rate == pytest.approx(
        1e-3
    )

    assert model.optimizer.clipnorm == pytest.approx(
        1.0
    )


def test_bilstm_is_bidirectional_only_within_history():
    model = build_frozen_model(
        "BiLSTM",
        "F0",
        compile_model=False,
    )

    bidirectional_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Bidirectional,
        )
    ]

    assert len(
        bidirectional_layers
    ) == 1

    assert isinstance(
        bidirectional_layers[
            0
        ].forward_layer,
        tf.keras.layers.LSTM,
    )

    assert model.inputs[
        0
    ].shape[
        1:
    ] == (
        13,
        9,
    )


def test_tcn_structure_is_causal_and_uses_last_position():
    model = build_frozen_model(
        "TCN",
        "F0",
        compile_model=False,
    )

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Conv1D,
        )
    ]

    assert len(
        conv_layers
    ) == 7

    assert all(
        layer.padding == "causal"
        for layer in conv_layers
    )

    assert all(
        tuple(
            layer.strides
        )
        == (
            1,
        )
        for layer in conv_layers
    )

    assert (
        model.get_layer(
            "tcn_last_position"
        )
        is not None
    )

    assert not any(
        isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling1D,
        )
        for layer in model.layers
    )

    prohibited = (
        tf.keras.layers.LSTM,
        tf.keras.layers.GRU,
        tf.keras.layers.Bidirectional,
    )

    assert not any(
        isinstance(
            layer,
            prohibited,
        )
        for layer in model.layers
    )

    first_main_conv = (
        model.get_layer(
            "tcn_block1_conv1"
        )
    )

    assert (
        first_main_conv
        .kernel_initializer
        .__class__
        .__name__
        == "HeNormal"
    )


def test_architecture_ranking_uses_three_frozen_keys():
    results = pd.DataFrame(
        {
            "candidate_id":
                [
                    "higher_mae",
                    "larger_tie",
                    "best",
                    "worse_rmse",
                ],
            "validation_MAE_kW":
                [
                    0.20,
                    0.10,
                    0.10,
                    0.10,
                ],
            "validation_RMSE_kW":
                [
                    0.10,
                    0.20,
                    0.20,
                    0.30,
                ],
            "trainable_parameter_count":
                [
                    1,
                    200,
                    100,
                    50,
                ],
        }
    )

    ranked = rank_architecture_results(
        results
    )

    assert ranked[
        "candidate_id"
    ].tolist() == [
        "best",
        "larger_tie",
        "worse_rmse",
        "higher_mae",
    ]


def test_early_stopping_callback_matches_frozen_protocol():
    callback = (
        make_early_stopping_callback()
    )

    assert callback.monitor == "val_loss"
    assert callback.mode == "min"
    assert callback.patience == 8
    assert callback.min_delta == pytest.approx(
        1e-4
    )

    assert (
        callback.restore_best_weights
        is True
    )


def _scaling_fixture():
    sequence = np.arange(
        4 * 13 * 9,
        dtype=np.float32,
    ).reshape(
        4,
        13,
        9,
    )

    sequence[
        2:
    ] += 10_000.0

    auxiliary = np.array(
        [
            [
                1.0,
                0.0,
                10.0,
            ],
            [
                3.0,
                1.0,
                14.0,
            ],
            [
                100.0,
                0.0,
                1_000.0,
            ],
            [
                200.0,
                1.0,
                2_000.0,
            ],
        ],
        dtype=np.float64,
    )

    target = np.array(
        [
            1.0,
            3.0,
            101.0,
            103.0,
        ],
        dtype=np.float32,
    )

    train_mask = np.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    return (
        sequence,
        auxiliary,
        target,
        train_mask,
    )


def test_train_only_scaling_ignores_validation_and_test_for_fit():
    (
        sequence,
        auxiliary,
        target,
        train_mask,
    ) = _scaling_fixture()

    stats = fit_train_only_scalers(
        sequence,
        auxiliary,
        target,
        train_mask,
        continuous_positions=[
            0,
            2,
        ],
        binary_positions=[
            1,
        ],
    )

    expected_sequence_mean = (
        sequence[
            :2
        ]
        .astype(
            np.float64
        )
        .mean(
            axis=(
                0,
                1,
            )
        )
    )

    expected_sequence_std = (
        sequence[
            :2
        ]
        .astype(
            np.float64
        )
        .std(
            axis=(
                0,
                1,
            ),
            ddof=0,
        )
    )

    np.testing.assert_allclose(
        stats[
            "sequence_mean"
        ],
        expected_sequence_mean,
    )

    np.testing.assert_allclose(
        stats[
            "sequence_std"
        ],
        expected_sequence_std,
    )

    np.testing.assert_allclose(
        stats[
            "auxiliary_mean"
        ],
        np.array(
            [
                2.0,
                12.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        stats[
            "auxiliary_std"
        ],
        np.array(
            [
                1.0,
                2.0,
            ]
        ),
    )

    assert stats[
        "target_mean"
    ] == pytest.approx(
        2.0
    )

    assert stats[
        "target_std"
    ] == pytest.approx(
        1.0
    )

    assert stats[
        "n_train"
    ] == 2


def test_apply_scaling_preserves_binary_columns():
    (
        sequence,
        auxiliary,
        target,
        train_mask,
    ) = _scaling_fixture()

    stats = fit_train_only_scalers(
        sequence,
        auxiliary,
        target,
        train_mask,
        continuous_positions=[
            0,
            2,
        ],
        binary_positions=[
            1,
        ],
    )

    transformed = (
        apply_train_only_scalers(
            sequence,
            auxiliary,
            target,
            stats,
        )
    )

    assert transformed[
        "sequence"
    ].dtype == np.float32

    assert transformed[
        "auxiliary"
    ].dtype == np.float32

    assert transformed[
        "target_scaled"
    ].dtype == np.float32

    np.testing.assert_array_equal(
        transformed[
            "auxiliary"
        ][
            :,
            1,
        ],
        auxiliary[
            :,
            1,
        ].astype(
            np.float32
        ),
    )

    np.testing.assert_allclose(
        transformed[
            "target_scaled"
        ][
            :2
        ],
        np.array(
            [
                -1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def test_zero_variance_scaling_input_is_rejected():
    sequence = np.ones(
        (
            3,
            13,
            9,
        ),
        dtype=np.float32,
    )

    auxiliary = np.array(
        [
            [
                1.0,
                0.0,
            ],
            [
                2.0,
                1.0,
            ],
            [
                3.0,
                0.0,
            ],
        ]
    )

    target = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="sequence",
    ):
        fit_train_only_scalers(
            sequence,
            auxiliary,
            target,
            [
                True,
                True,
                False,
            ],
            continuous_positions=[
                0,
            ],
            binary_positions=[
                1,
            ],
        )


def test_inverse_target_scaling_roundtrip():
    (
        sequence,
        auxiliary,
        target,
        train_mask,
    ) = _scaling_fixture()

    stats = fit_train_only_scalers(
        sequence,
        auxiliary,
        target,
        train_mask,
        continuous_positions=[
            0,
            2,
        ],
        binary_positions=[
            1,
        ],
    )

    transformed = (
        apply_train_only_scalers(
            sequence,
            auxiliary,
            target,
            stats,
        )
    )

    restored = inverse_target_scaling(
        transformed[
            "target_scaled"
        ],
        stats,
    )

    np.testing.assert_allclose(
        restored,
        target,
        rtol=0.0,
        atol=1e-6,
    )


def test_combine_train_validation_preserves_order():
    train = {
        "sequence":
            np.full(
                (
                    2,
                    13,
                    9,
                ),
                1.0,
                dtype=np.float32,
            ),
        "auxiliary":
            np.array(
                [
                    [
                        10.0,
                    ],
                    [
                        11.0,
                    ],
                ],
                dtype=np.float32,
            ),
        "target_scaled":
            np.array(
                [
                    -1.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
    }

    validation = {
        "sequence":
            np.full(
                (
                    1,
                    13,
                    9,
                ),
                2.0,
                dtype=np.float32,
            ),
        "auxiliary":
            np.array(
                [
                    [
                        20.0,
                    ],
                ],
                dtype=np.float32,
            ),
        "target_scaled":
            np.array(
                [
                    1.0,
                ],
                dtype=np.float32,
            ),
    }

    combined = combine_train_validation(
        train,
        validation,
    )

    assert combined[
        "n_train"
    ] == 2

    assert combined[
        "n_validation"
    ] == 1

    assert combined[
        "n_trainval"
    ] == 3

    assert combined[
        "auxiliary"
    ][
        :,
        0,
    ].tolist() == [
        10.0,
        11.0,
        20.0,
    ]

    assert combined[
        "target_scaled"
    ].tolist() == [
        -1.0,
        0.0,
        1.0,
    ]


def test_frozen_epoch_lookup():
    assert get_frozen_epoch(
        "LSTM",
        15,
        "F0",
    ) == 7

    assert get_frozen_epoch(
        "GRU",
        60,
        "F4",
    ) == 1

    assert get_frozen_epoch(
        "BiLSTM",
        30,
        "F3",
    ) == 17

    assert get_frozen_epoch(
        "TCN",
        60,
        "F0",
    ) == 10

    assert (
        FROZEN_EPOCHS_BY_MODEL[
            "TCN"
        ]
        is TCN_FROZEN_EPOCHS
    )


def test_invalid_frozen_model_requests_are_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown frozen neural model",
    ):
        build_frozen_model(
            "Unknown",
            "F0",
            compile_model=False,
        )

    with pytest.raises(
        ValueError,
        match="Unknown feature family",
    ):
        build_frozen_model(
            "LSTM",
            "F9",
            compile_model=False,
        )

    with pytest.raises(
        ValueError,
        match="Unknown frozen",
    ):
        get_frozen_epoch(
            "Unknown",
            15,
            "F0",
        )

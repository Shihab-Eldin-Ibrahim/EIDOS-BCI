def prepare_classification_data(
    epochs,
    conditions,
    tmin=1.0,
    tmax=4.0
):
    """
    Prepare EEG data for classification.

    Returns
    -------
    X : ndarray
        Shape: (trials, channels, samples)

    y : ndarray
        Integer class labels.

    class_names : list
        Class names in label order.
    """

    print("Conditions:", conditions)

    # --------------------------------------------------------
    # Select only the required conditions
    # --------------------------------------------------------

    selected_epochs = epochs[
        list(conditions)
    ].copy()

    # --------------------------------------------------------
    # Explicitly remove EOG channels
    # --------------------------------------------------------

    eog_channels = [
        ch for ch in selected_epochs.ch_names
        if "EOG" in ch.upper()
    ]

    if eog_channels:
        selected_epochs.drop_channels(
            eog_channels
        )

    print(
        "EOG channels removed:",
        len(eog_channels)
    )

    # --------------------------------------------------------
    # Check EEG channel count
    # --------------------------------------------------------

    print(
        "EEG channels used:",
        len(selected_epochs.ch_names)
    )

    print(
        "Expected EEG channels: 22"
    )

    if len(selected_epochs.ch_names) != 22:
        raise ValueError(
            f"Expected 22 EEG channels, "
            f"but found {len(selected_epochs.ch_names)} "
            f"after EOG removal."
        )

    print("\nChannel names:")

    for index, name in enumerate(
        selected_epochs.ch_names,
        start=1
    ):
        print(
            f"{index:2d}. {name}"
        )

    # --------------------------------------------------------
    # Crop to motor-imagery period
    # --------------------------------------------------------

    selected_epochs.crop(
        tmin=tmin,
        tmax=tmax,
        include_tmax=False
    )

    # --------------------------------------------------------
    # Build X and y in the SAME order
    # --------------------------------------------------------

    X_parts = []
    labels = []

    for class_index, condition in enumerate(
        conditions
    ):

        condition_data = selected_epochs[
            condition
        ].get_data(
            copy=True
        )

        X_parts.append(
            condition_data
        )

        labels.extend(
            [class_index] * len(condition_data)
        )

    X = np.concatenate(
        X_parts,
        axis=0
    )

    y = np.asarray(
        labels,
        dtype=int
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print(
        "\nEEG data shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print(
        "Total trials:",
        len(X)
    )

    print(
        f"Time window: {tmin} - {tmax} seconds"
    )

    print(
        "Channels used for classification:",
        X.shape[1],
        "EEG"
    )

    print(
        "EOG channels used: 0"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_index, condition in enumerate(
        conditions
    ):

        count = np.sum(
            y == class_index
        )

        print(
            f"{condition}: {count}"
        )

    return X, y, list(conditions)
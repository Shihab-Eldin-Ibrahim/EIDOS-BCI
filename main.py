from core.loader import EEGLoader
from core.metadata import EEGMetadata
from preprocessing.epoching import EEGEpoching


def main():

    loader = EEGLoader()

    raw = loader.load("dataset/2a/A01T.gdf")

    EEGMetadata.print_summary(raw)

    epochs = EEGEpoching.create_motor_imagery_epochs(raw)

    print("\n" + "=" * 60)
    print("MOTOR IMAGERY EPOCHS")
    print("=" * 60)

    print(f"Number of epochs : {len(epochs)}")
    print(f"Epoch shape      : {epochs.get_data().shape}")
    print(f"Sampling rate    : {epochs.info['sfreq']} Hz")
    print(f"Channels         : {len(epochs.ch_names)}")
    print(f"Time range       : {epochs.tmin} to {epochs.tmax} seconds")

    print("\nEpoch counts:")
    print(epochs["left"])
    print(epochs["right"])
    print(epochs["feet"])
    print(epochs["tongue"])

    print("\nEpoch event IDs:")
    print(epochs.event_id)

    # Visualize one left-hand imagery trial
    epochs["left"].plot(
        n_epochs=1,
        n_channels=22,
        scalings="auto",
        block=True
    )


if __name__ == "__main__":
    main()
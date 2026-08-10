from core.loader import  EEGLoader
from core.metadata import EEGMetadata
import mne


def main():

    loader = EEGLoader()

    raw = loader.load("dataset/2a/A01T.gdf")

    EEGMetadata.print_summary(raw)

    events, event_dict = mne.events_from_annotations(raw)

    print("\nEvent Dictionary:")
    print(event_dict)

    print("\nNumber of events:")
    print(len(events))

    print("\nFirst 20 events:")
    print(events[:20])


if __name__ == "__main__":
    main()
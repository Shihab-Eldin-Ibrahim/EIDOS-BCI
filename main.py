from core.loader import EEGloader
from core.metadata import EEGMetadata

def main():

    loader = EEGloader()

    raw = loader.load("dataset/2a/A01T.gdf")
    EEGMetadata.print_summary(raw)

    print(raw)


if __name__ == "__main__":
    main()
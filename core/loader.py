from pathlib import Path
import mne

class EEGloader():

    def load(self,file_path:str ):
        path= Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        raw = mne.io.read_raw_gdf (path, preload=True)
        return raw

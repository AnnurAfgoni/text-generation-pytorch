import gdown
from pathlib import Path


def downloader(output_file):
    file_id = "1M1-1Hs198XDD6Xx-kSWLThv1elZBzJ0j"
    prefix = 'https://drive.google.com/uc?/export=download&id='

    url_download = prefix+file_id
    
    if not Path(output_file).exists():
        print("Downloading...")
        gdown.download(url_download, output_file)
        print("Download Finish...")
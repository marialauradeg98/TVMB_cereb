"""Download the Conn_Allen connectivity data into conn_data/Conn_Allen.

These files (ConnAllen.h5, RegionVolumeMapping.h5, TemplateAllen.h5,
VolumeAllen.h5) are too large for GitHub, so they are hosted on Google
Drive and pulled down on demand instead of being tracked by git.

Usage:
    python scripts/download_conn_allen.py
"""

import sys
from pathlib import Path

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1AetN53mHFb83PTSPVHqvUVBgAfvklvze"
DEST_DIR = Path(__file__).resolve().parent.parent / "conn_data" / "Conn_Allen"


def main() -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(DEST_DIR), quiet=False)


if __name__ == "__main__":
    main()

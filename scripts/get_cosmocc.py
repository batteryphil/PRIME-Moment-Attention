#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile
import stat

VERSION = "3.3.3"
URL = f"https://github.com/jart/cosmopolitan/releases/download/{VERSION}/cosmocc-{VERSION}.zip"

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cosmocc_dir = os.path.join(base_dir, "cosmocc")
    cosmocc_bin = os.path.join(cosmocc_dir, "bin", "cosmocc")

    if os.path.exists(cosmocc_bin) and os.access(cosmocc_bin, os.X_OK):
        print(f"cosmocc is already installed at {cosmocc_dir}")
        sys.exit(0)

    print(f"Downloading cosmocc v{VERSION}...")
    os.makedirs(cosmocc_dir, exist_ok=True)
    zip_path = os.path.join(cosmocc_dir, "cosmocc.zip")
    
    try:
        urllib.request.urlretrieve(URL, zip_path)
    except Exception as e:
        print(f"Failed to download cosmocc: {e}")
        sys.exit(1)

    print("Extracting cosmocc...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cosmocc_dir)
    except Exception as e:
        print(f"Failed to extract cosmocc.zip: {e}")
        sys.exit(1)
        
    os.remove(zip_path)

    # Ensure binaries in bin/ are executable (ZipFile doesn't always preserve permissions)
    bin_dir = os.path.join(cosmocc_dir, "bin")
    if os.path.exists(bin_dir):
        for filename in os.listdir(bin_dir):
            filepath = os.path.join(bin_dir, filename)
            st = os.stat(filepath)
            os.chmod(filepath, st.st_mode | stat.S_IEXEC)

    print(f"cosmocc installed successfully to {cosmocc_dir}!")

if __name__ == "__main__":
    main()

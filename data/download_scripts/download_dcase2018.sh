#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/dcase2018"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/api/records/1228142/files-archive"
zip_path="$download_dir/files-archive.zip"

echo "Downloading DCASE 2018 dataset from $url ..."
wget -O "$zip_path" "$url"

echo "Download complete. Extracting archive..."
unzip -q "$zip_path" -d "$download_dir"

# Delete the main zip archive
rm "$zip_path"
echo "Main archive extracted and removed."

# Change to download directory
cd "$download_dir"

# Loop over all zip files and extract them one by one
for file in *.zip; do
  echo "Extracting $file ..."
  unzip -q "$file" -d "$download_dir"
  rm "$file"
  echo "Removed $file after extraction."
done

echo "All files extracted and cleaned up. Files are located in: $download_dir"

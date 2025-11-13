#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/cas2023"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/api/records/10616533/files-archive"
zip_path="$download_dir/files-archive.zip"

echo "Downloading CAS 2023 dataset from $url ..."
curl -L "$url" -o "$zip_path"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"
unzip -q "$download_dir/ICME2024_GC_ASC_dev.zip" -d "$download_dir" 

# Delete the zip file
rm "$zip_path"
rm "$download_dir/ICME2024_GC_ASC_dev.zip"

echo "Extraction complete. Files are located in: $download_dir"
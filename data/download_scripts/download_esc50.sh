#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/esc50"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://github.com/karoldvl/ESC-50/archive/master.zip"
zip_path="$download_dir/master.zip"

echo "Downloading ESC-50 dataset from $url ..."
curl -L "$url" -o "$zip_path"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"

echo "Extraction complete. Files are located in: $download_dir"

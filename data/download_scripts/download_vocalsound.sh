#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/vocalsound"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1"
zip_path="$download_dir/ReaLISED_Dataset.zip?download=1"

echo "Downloading VocalSound dataset from $url ..."
wget -O "$zip_path" "$url"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"

echo "Extraction complete. Files are located in: $download_dir"

#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/realised"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/records/6488321/files/ReaLISED_Dataset.zip?download=1"
zip_path="$download_dir/ReaLISED_Dataset.zip?download=1"

echo "Downloading ReaLISED dataset from $url ..."
curl -L "$url" -o "$zip_path"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"

echo "Extraction complete. Files are located in: $download_dir"

#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/one-shot_percussive_sounds"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/api/records/4687854/files-archive"
zip_path="$download_dir/files-archive.zip"

echo "Downloading Freesound One-Shot Percussive Sounds dataset from $url ..."
wget -O "$zip_path" "$url"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"


analysis_zip="$download_dir/analysis.zip"
data_path="$download_dir/one_shot_percussive_sounds.zip"

unzip -q "$analysis_zip" -d "$download_dir"
unzip -q "$data_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"
rm "$analysis_zip"
rm "$data_path"

echo "Extraction complete. Files are located in: $download_dir"

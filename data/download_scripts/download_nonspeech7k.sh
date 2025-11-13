#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/nonspeech7k"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/api/records/6967442/files-archive"
zip_path="$download_dir/files-archive"

echo "Downloading VocalSound dataset from $url ..."
wget -O "$zip_path" "$url"

echo "Download complete. Extracting..."
unzip -q "$zip_path" -d "$download_dir"


train_path="$download_dir/train.zip"
test_path="$download_dir/test.zip"

unzip -q "$train_path" -d "$download_dir"
unzip -q "$test_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"
rm "$train_path"
rm "$test_path"

echo "Extraction complete. Files are located in: $download_dir"

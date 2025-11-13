#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/visc_son"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/records/5606504/files/VISC%20Dataset%20SON.rar?download=1"
rar_path="$download_dir/VISC_Dataset_SON.rar"

# Check for unrar
if ! command -v unrar &> /dev/null; then
  echo "Error: 'unrar' is not installed. Please install it first:"
  echo "  sudo apt update && sudo apt install unrar"
  exit 1
fi

echo "📥 Downloading Vehicle Interior Sound Dataset from:"
echo "   $url"
curl -L "$url" -o "$rar_path"

echo "✅ Download complete."
echo "📂 Extracting..."
unrar x -y "$rar_path" "$download_dir" > /dev/null

# Delete the rar file
rm "$rar_path"

echo "✅ Extraction complete."
echo "Files are located in: $download_dir"

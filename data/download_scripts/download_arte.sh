#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/arte"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url="https://zenodo.org/api/records/2261633/files-archive"
zip_path="$download_dir/files-archive.zip"

echo "Downloading ARTE dataset from $url ..."
wget -O "$zip_path" "$url"

echo "Download complete. Extracting ZIP archive..."
unzip -q "$zip_path" -d "$download_dir"

# Delete the zip file
rm "$zip_path"

echo "ZIP extraction complete. Beginning 7z extraction..."

# Find and extract all .7z files
find "$download_dir" -type f -name "*.7z" | while read -r file; do
  echo "Extracting $(basename "$file") ..."
  7z x -y -o"$download_dir" "$file" >/dev/null
  echo "Extraction complete for $(basename "$file")"
  rm "$file"
done

echo "All 7z archives extracted and removed."
echo "Final files are located in: $download_dir"

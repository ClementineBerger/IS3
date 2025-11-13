#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Assign the first argument to a variable
download_dir="$1/cochlscene"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# Define URL and paths
url1="https://zenodo.org/records/7080122/files/CochlScene.z01?download=1"
url2="https://zenodo.org/records/7080122/files/CochlScene.z02?download=1"
url3="https://zenodo.org/records/7080122/files/CochlScene.z03?download=1"
url4="https://zenodo.org/records/7080122/files/CochlScene.z04?download=1"
url5="https://zenodo.org/records/7080122/files/CochlScene.z05?download=1"
url6="https://zenodo.org/records/7080122/files/CochlScene.zip?download=1"

zip_path1="$download_dir/CochlScene.z01"
zip_path2="$download_dir/CochlScene.z02"
zip_path3="$download_dir/CochlScene.z03"
zip_path4="$download_dir/CochlScene.z04"
zip_path5="$download_dir/CochlScene.z05"
zip_path6="$download_dir/CochlScene.zip"

echo "📥 Downloading CochlScene dataset from https://zenodo.org/records/7080122 ..."
wget -O "$zip_path1" "$url1"
wget -O "$zip_path2" "$url2"
wget -O "$zip_path3" "$url3"
wget -O "$zip_path4" "$url4"
wget -O "$zip_path5" "$url5"
wget -O "$zip_path6" "$url6"

echo "✅ Download complete."

cd "$download_dir"

echo "🔍 Checking for multipart ZIP files..."

echo "📦 Extracting..."

7z x "$download_dir/CochlScene.zip"

rm "$zip_path1"
rm "$zip_path2"
rm "$zip_path3"
rm "$zip_path4"
rm "$zip_path5"
rm "$zip_path6"

echo "🎉 Done! Files are located in: $download_dir"

#!/bin/bash
set -e
echo "Cleaning up the old project files..."
rm -rf build
echo "Alright, setting things up with CMake..."
cmake -S . -B build
echo "Building the project now..."
cmake --build build
echo "Running tinybitnet! Here we go..."
./build/tinybitnet
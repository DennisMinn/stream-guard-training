#!/bin/bash

# Directory containing the input files
input_directory="data/raw"

# Directory where you want to store the processed files
output_directory="data/preprocess"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through each file in the input directory
for file in "$input_directory"/*; do
    # Check if the item is a file
    if [ -f "$file" ]; then
        # Run the Python script with input and output paths as arguments
        python preprocessing.py --input_path "$file" --output_directory "$output_directory"
        
        # Echo a message when the script has been run for the file
        echo "Processed: $file"
    fi
done


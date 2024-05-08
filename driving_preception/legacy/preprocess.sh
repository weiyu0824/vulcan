#!/bin/bash

# Define the ranges of parameters
g_params=(0.4 0.5)
v_params=(0.1 0.2 0.3 0.4 0.5)

# Iterate over parameter combinations
for g in "${g_params[@]}"; do
    for v in "${v_params[@]}"; do
        # Construct the command to run pipeline.py with different params
        command="python3 pipeline.py -g $g -v $v"
        
        # Print the command and then execute it
        echo "Running: $command"
        $command
    done
done

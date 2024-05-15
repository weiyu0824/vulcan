#!/bin/bash

# Define the ranges of parameters
g_params=(40 50)
v_params=(10 20 30 40 50)
m_params=(centerpoint_dcn centerpoint_circlenms pointpillars_secfpn ssn_second)

# for m in "${m_params[@]}"; do  # Corrected: added [@]
#     # Construct the command to run pipeline.py with different params
#     command="python3 test.py config/$m.py config/$m.pth --ds standard"
    
#     # Print the command and then execute it
#     echo "Running: $command"
#     echo "Running: $command" >> record.txt
#     $command
# done


# Iterate over parameter combinations
for g in "${g_params[@]}"; do
    for v in "${v_params[@]}"; do
        for m in "${m_params[@]}"; do  # Corrected: added [@]
            # Construct the command to run pipeline.py with different params
            command="python3 test.py config/$m.py config/$m.pth --ds g${g}-v${v}"
            
            # Print the command and then execute it
            echo "Running: $command"
            echo "Running: $command" >> record.txt
            $command
        done
    done
done
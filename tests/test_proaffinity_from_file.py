"""Test script for ProAffinity prediction using file-based approach."""

import sys
import os
sys.path.append("..")

from ionerdss.model.proaffinity_predictor import (
    predict_proaffinity_binding_energy_from_file,
    predict_proaffinity_binding_energy
)

# Test 1: Using the new file-based function
print("=" * 80)
print("Test 1: predict_proaffinity_binding_energy_from_file")
print("=" * 80)

pdb_file = "../data/8erq-assembly1.cif"  # Assuming this file exists
chains = "A,H"
adfr_path = '/home/local/WIN/msang2/ADFRsuite-1.0/bin/prepare_receptor'

if os.path.exists(pdb_file):
    print(f"Testing with file: {pdb_file}")
    binding_energy = predict_proaffinity_binding_energy_from_file(
        pdb_file=pdb_file,
        chains=chains,
        verbose=True,
        adfr_path=adfr_path
    )
    print(f"\nPredicted binding energy: {binding_energy:.4f} kJ/mol")
    
    # Convert to binding constant
    import numpy as np
    R = 8.314 / 1000  # kJ/(mol*K)
    T = 298  # K
    K = np.exp(-binding_energy / (R * T))
    print(f"Predicted binding constant K: {K:.4e}")
else:
    print(f"PDB file not found: {pdb_file}")
    print("Skipping file-based test")

print("\n" + "=" * 80)
print("Test 2: predict_proaffinity_binding_energy (original, PDB ID-based)")
print("=" * 80)

# Test 2: Using the original PDB ID-based function (for comparison)
pdb_id = "8erq"
print(f"Testing with PDB ID: {pdb_id}")

binding_energy_from_id = predict_proaffinity_binding_energy(
    pdb_id=pdb_id,
    chains=chains,
    verbose=True,
    adfr_path=adfr_path
)
print(f"\nPredicted binding energy: {binding_energy_from_id:.4f} kJ/mol")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("Both functions should work and produce similar results.")
print("The file-based function avoids re-downloading the PDB file.")

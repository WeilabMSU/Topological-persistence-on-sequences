# Topological-persistence-on-sequences
This repository contains code for extracting topological features from DNA sequences using fragment occurrence frequencies as a filtration function.
## File Descriptions

1. **`ebolavirus_record`**  
   Stores the names of Ebola virus strains and their corresponding family classifications.

2. **`ebolavirus_sequence`**  
   Contains the DNA sequences of each virus strain.

3. **`stopology`**  
   Computes the homology and the smallest positive eigenvalues of the Laplacian for the DNA sequences.

4. **`delta_complex_geq`**  
   Generates a filtration of delta complexes based on fragment frequencies, sorted in descending order.

5. **`Main`**  
   Computes the topological features, including:
   - Betti numbers (dimensions 0, 1, 2, and 3)
   - The smallest positive eigenvalues as functions of the frequency parameter for each virus.  
   Performs clustering analysis using the extracted topological features. Viruses belonging to the same family are visually distinguished with the same color.

## Usage

Follow the instructions provided in each script to extract and analyze the topological features of the given DNA sequences.  

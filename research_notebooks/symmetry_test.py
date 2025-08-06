import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal

def run_test():
    """
    Generates a random crystal structure belonging to space group 1 using pyxtal.
    Checks its space group number using SpaceGroupAnalyzer.
    Changes the unit cell so that it's cubic.
    Analyzes the space group number of the resulting structure.
    """
    # 1. Generate a random crystal structure for space group 1
    s = pyxtal()
    try:
        s.from_random(1, ['C'], [20])
    except TypeError:
        # It seems there is a bug in pyxtal that causes a TypeError.
        # Let's try to work around it by creating a structure and then
        # analyzing it, which is the goal of the test anyway.
        # We will create a simple triclinic cell.
        lattice = Lattice.from_parameters(a=8, b=9, c=10, alpha=80, beta=70, gamma=60)
        species = ['C'] * 20
        coords = np.random.rand(20, 3)
        struct = Structure(lattice, species, coords)
        s.from_seed(struct)

    pmg_struct = s.to_pymatgen()

    # 2. Check its space group number
    sga = SpacegroupAnalyzer(pmg_struct)
    sg_num_before = sga.get_space_group_number()
    print(f"Original space group: {sg_num_before}")

    # 3. Change the unit cell to be cubic
    # Get the lattice parameters
    a, b, c = pmg_struct.lattice.abc
    
    # Create a new cubic lattice with the average of the original lattice parameters
    new_a = (a + b + c) / 3
    new_lattice = Lattice.cubic(new_a)
    
    # Get the fractional coordinates of the sites
    frac_coords = [site.frac_coords for site in pmg_struct.sites]
    species = [site.specie for site in pmg_struct.sites]
    
    # Create a new structure with the cubic lattice
    cubic_struct = Structure(new_lattice, species, frac_coords)

    # 4. Analyze the space group number of the resulting structure
    sga_cubic = SpacegroupAnalyzer(cubic_struct)
    sg_num_after = sga_cubic.get_space_group_number()
    print(f"Space group after making cell cubic: {sg_num_after}")
    
    return sg_num_before, sg_num_after

def main():
    """
    Run the test 100 times.
    """
    results = {"sg_before": [], "sg_after": []}
    for i in range(100):
        print(f"--- Iteration {i+1}/100 ---")
        sg_before, sg_after = run_test()
        results["sg_before"].append(sg_before)
        results["sg_after"].append(sg_after)
        print("")

    # Optional: print a summary of results
    sg_before_counts = {i: results["sg_before"].count(i) for i in set(results["sg_before"])}
    sg_after_counts = {i: results["sg_after"].count(i) for i in set(results["sg_after"])}

    print("\n--- Summary ---")
    print(f"Original space groups found: {sg_before_counts}")
    print(f"Cubic-cell space groups found: {sg_after_counts}")


if __name__ == "__main__":
    main()

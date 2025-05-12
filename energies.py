import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import codata2014 as const

import numpy as np

# define atomic masses
atomic_masses = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
    "N": 14.007}

def read_coordinates(file_path):
    """
    reads atomic coordinates and element types from a file.
    file format: Each line should contain 'Element X Y Z', e.g., 'C 0.0 1.0 2.0'.
    """
    atoms = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            element = parts[0]
            x, y, z = map(float, parts[1:])
            atoms.append((element, x, y, z))
    return atoms

def calculate_com(fragment):
    """
    calculates the center of mass for a given fragment.
    input: list of tuples (element, x, y, z)
    returns: tuple (COM_x, COM_y, COM_z)
    """
    total_mass = 0
    weighted_sum = np.array([0.0, 0.0, 0.0])
    for element, x, y, z in fragment:
        mass = atomic_masses[element]
        total_mass += mass
        weighted_sum += mass * np.array([x, y, z])
    return weighted_sum / total_mass

def calculate_ionic_radius(fragment, com):
    """
    calculate the ionic radius of a fragment as the maximum distance of atoms
    from the center of mass (COM)
    """
    max_distance = 0
    for _, x, y, z in fragment:
        distance = np.linalg.norm(np.array([x, y, z]) - np.array(com))
        max_distance = max(max_distance, distance)
    return max_distance

def calculate_distance(com1, com2):
    """
    calculates the distance between two centers of mass.
    """
    return np.linalg.norm(np.array(com2) - np.array(com1))

def Hartree_to_eV(energy):
    """
    converts energy in [Hartree] to [eV]
    """
    return energy * 27.2114079527 * u.eV

def ionization_potential(cation, neutral):
    """
    calculates ionazation potential, both input and output should be in [eV]
    """
    return cation - (neutral)

def electron_affinity(anion, neutral):
    """
    calculates electron affinity, both input and output should be in [eV]
    """
    return neutral - anion

def coulomb_vacuum(radius):     # unit of eps0 from packages is [F/m]
    """
    calculates coulomb energy in vacuum 
    input: radii in [m]
    output: [eV]
    """
    return - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1) * radius)).to(u.eV)

def css_vacuum(IP, EA, EC):
    """
    calculates energy of CSS. input should be in eV
    """
    return IP - EA + EC

def coulomb_solution(radius, radius1, radius2, eps_stat_tol, eps_op_tol):
    """
    calculates coulomb energy in solution 
    input: radii in [m]
    output: [eV]
    """
    # return - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1) * radius) + const.e**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1)) * (1/(2*radius1) + 1/(2*radius2)) * (1 - 1/eps_tol)).to(u.eV)  # johan
    # return - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1) * eps_tol * radius )).to(u.eV)   # atkins
    return - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1) * eps_stat_tol * radius )).to(u.eV) - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1)) * (1/(2*radius1) + 1/(2*radius2) - 1/radius) * (1/(eps_op_tol) - 1/eps_stat_tol)).to(u.eV)  # atkins + reorganization energy
    # return - ((const.e)**2/(4* np.pi * const.eps0.to(u.C**2 * u.m**-1 * u.J**-1)) * (1/(2*radius1) + 1/(2*radius2) - 1/radius) * (1/(eps_op_tol) - 1/eps_stat_tol)).to(u.eV)

def css_solvent(cation_donor, neutral_donor, anion_acceptor, neutral_acceptor, rad_btw_frag, ionic_rad_donor, ionic_rad_acceptor, eps_stat_tol, eps_op_tol):
    """
    calculates energy of CSS. energy input should be in [eV] and radii in [m]
    output: [eV]
    """
    return ionization_potential(cation_donor, neutral_donor) - electron_affinity(anion_acceptor, neutral_acceptor) + coulomb_solution(rad_btw_frag, ionic_rad_donor, ionic_rad_acceptor, eps_stat_tol, eps_op_tol)


def css_vacuum(cation_donor, neutral_donor, anion_acceptor, neutral_acceptor, radius):
    return ionization_potential(cation_donor, neutral_donor) - electron_affinity(anion_acceptor, neutral_acceptor) + coulomb_vacuum(radius)

def main():
    # read the full molecule's coordinates
    atoms = read_coordinates("molecule.txt") 
    atoms_sol = read_coordinates("sol_molecule.txt")

    # define fragments by their indices (1-based indexing from the molecule.txt file)
    fragment_MeOAn_indices = list(range(34, 50))  # atoms 34 to 49
    fragment_bANIb_indices = (
        list(range(1, 34)) +   # atoms 1 to 33
        list(range(50, 58)) +  # atoms 50 to 57
        list(range(83, 91))    # atoms 83 to 90
    )
    fragment_NDI_indices = list(range(58, 83))  # atoms 58 to 82

    # split atoms into fragments
    fragment_MeOAn = [atoms[i - 1] for i in fragment_MeOAn_indices]
    fragment_bANIb = [atoms[i - 1] for i in fragment_bANIb_indices]
    fragment_NDI = [atoms[i - 1] for i in fragment_NDI_indices]

    fragment_MeOAn_sol = [atoms_sol[i - 1] for i in fragment_MeOAn_indices]
    fragment_bANIb_sol = [atoms_sol[i - 1] for i in fragment_bANIb_indices]
    fragment_NDI_sol = [atoms_sol[i - 1] for i in fragment_NDI_indices]

    ### all the following lines of code calculate and print quantities

    # calculate centers of masses
    com_MeOAn = calculate_com(fragment_MeOAn)
    com_bANIb = calculate_com(fragment_bANIb)
    com_NDI = calculate_com(fragment_NDI)

    com_MeOAn_sol = calculate_com(fragment_MeOAn_sol)
    com_bANIb_sol = calculate_com(fragment_bANIb_sol)
    com_NDI_sol = calculate_com(fragment_NDI_sol)

    # calculate distance
    distance_MeOAn_NDI = calculate_distance(com_MeOAn, com_NDI)
    distance_MeOAn_NDI_sol = calculate_distance(com_MeOAn_sol, com_NDI_sol)

    distance_MeOAn_ANI = calculate_distance(com_MeOAn, com_bANIb)
    distance_MeOAn_ANI_sol = calculate_distance(com_MeOAn_sol, com_bANIb_sol)

    # calculate the ionic radius
    ionic_radius_MeOAn = calculate_ionic_radius(fragment_MeOAn, com_MeOAn)
    ionic_radius_bANIb = calculate_ionic_radius(fragment_bANIb, com_bANIb)
    ionic_radius_NDI = calculate_ionic_radius(fragment_NDI, com_NDI)

    ionic_radius_MeOAn_sol = calculate_ionic_radius(fragment_MeOAn_sol, com_MeOAn_sol)
    ionic_radius_bANIb_sol = calculate_ionic_radius(fragment_bANIb_sol, com_bANIb_sol)
    ionic_radius_NDI_sol = calculate_ionic_radius(fragment_NDI_sol, com_NDI_sol)

    # output results
    # print(f"Ionic Radius of the MeOAn: {ionic_radius_MeOAn:.3f} Å")
    # print(f"Ionic Radius of the bANIb: {ionic_radius_bANIb:.3f} Å")
    # print(f"Ionic Radius of the NDI: {ionic_radius_NDI:.3f} Å")

    # print(f"Ionic Radius of the MeOAn (in solvent): {ionic_radius_MeOAn_sol:.3f} Å")
    # print(f"Ionic Radius of the bANIb (in solvent): {ionic_radius_bANIb_sol:.3f} Å")
    # print(f"Ionic Radius of the NDI (in solvent): {ionic_radius_NDI_sol:.3f} Å")

    # output results
    print(f"Center of Mass for Fragment MeOAn: {com_MeOAn}")
    print(f"Center of Mass for Fragment bANIb: {com_bANIb}")
    print(f"Center of Mass for Fragment NDI: {com_NDI}")
    print(f"Distance between centers of mass, sol: {distance_MeOAn_NDI_sol:.3f} Å")
    print(f"Distance between centers of mass, sol: {distance_MeOAn_ANI_sol:.3f} Å")

    # print(f"Center of Mass for Fragment MeOAn (in solvent): {com_MeOAn_sol}")
    # print(f"Center of Mass for Fragment bANIb (in solvent): {com_bANIb_sol}")
    # print(f"Center of Mass for Fragment NDI (in solvent): {com_NDI_sol}")
    # print(f"Distance between centers of mass (in solvent): {distance_MeOAn_NDI_sol:.3f} Å")


    # define energies from optimized geometeries
    E_full_vac = Hartree_to_eV(-2533.726673)
    E_full_sol = Hartree_to_eV(-2533.743262)

    E_MeOAn_vac = Hartree_to_eV(-401.850565)
    E_cat_MeOAn_vac = Hartree_to_eV(-401.601877)
    E_MeOAn_sol = Hartree_to_eV(-401.854855)
    E_cat_MeOAn_sol = Hartree_to_eV(-401.646169)

    E_ANI_vac = Hartree_to_eV(-1226.842931)
    E_cat_ANI_vac = Hartree_to_eV(-1226.654099) 
    E_anion_ANI_vac = Hartree_to_eV(-1226.842931)
    E_cat_ANI_sol = Hartree_to_eV(-3) ### insert value
    E_ANI_sol = Hartree_to_eV(-1226.840322)
    E_anion_ANI_sol = Hartree_to_eV(-1226.898716) 

    E_NDI_vac = Hartree_to_eV(-947.218667)
    E_anion_NDI_vac = Hartree_to_eV(-947.305075)
    E_NDI_sol = Hartree_to_eV(-947.226683)  
    E_anion_NDI_sol = Hartree_to_eV(-947.345922)

    # define radii
    r_MeOAn_NDI_vac = (distance_MeOAn_NDI * u.AA).to(u.m) 
    r_MeOAn_NDI_sol = (distance_MeOAn_NDI_sol * u.AA).to(u.m) 

    r_MeOAn_ANI_vac = (distance_MeOAn_ANI * u.AA).to(u.m) 
    r_MeOAn_ANI_sol = (distance_MeOAn_ANI_sol * u.AA).to(u.m) 

    # define di electric constants for toluene
    eps_sta_tol = 2.38 #
    eps_optical_tol = 2.2407
    eps_sta_chlorobenzene = 5.54
    eps_optical_chlorobenzene = 2.32

    # calc electron affinities and ionization potentials
    get_IP_MeOAn_vac = ionization_potential(E_cat_MeOAn_vac, E_MeOAn_vac)
    get_EA_NDI_vac = electron_affinity(E_anion_NDI_vac, E_NDI_vac)

    get_c_vac_3 = coulomb_vacuum(r_MeOAn_NDI_vac)
    get_c_vac_1 = coulomb_vacuum(r_MeOAn_ANI_vac)
    print("Coulomb MEOAn-NDI, vac", get_c_vac_3)
    print("Coulomb MEOAn-ANI, vac", get_c_vac_1)

    get_c_solution_3 = coulomb_solution(r_MeOAn_NDI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_NDI_sol* u.AA).to(u.m), eps_sta_tol, eps_optical_tol)
    get_c_solution_1 = coulomb_solution(r_MeOAn_ANI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_bANIb_sol* u.AA).to(u.m), eps_sta_tol, eps_optical_tol)
    print("Coulomb MEOAn-NDI, sol", get_c_solution_3)
    print("Coulomb MEOAn-ANI, sol", get_c_solution_1)

    # get energy of charge separated state(s) !
    get_css_1_vac = css_vacuum(E_cat_MeOAn_vac, E_MeOAn_vac, E_anion_ANI_vac, E_ANI_vac, r_MeOAn_ANI_vac)
    get_css_1_sol = css_solvent(E_cat_MeOAn_sol, E_MeOAn_sol, E_anion_ANI_sol, E_ANI_sol, r_MeOAn_ANI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_bANIb_sol* u.AA).to(u.m), eps_sta_tol, eps_optical_tol)
    get_css_1_sol_diChMe = css_solvent(E_cat_MeOAn_sol, E_MeOAn_sol, E_anion_ANI_sol, E_ANI_sol, r_MeOAn_ANI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_bANIb_sol* u.AA).to(u.m), eps_sta_chlorobenzene, eps_optical_chlorobenzene)

    get_css_3_vac = css_vacuum(E_cat_MeOAn_vac, E_MeOAn_vac, E_anion_NDI_vac, E_NDI_vac, r_MeOAn_NDI_vac)
    get_css_3_sol = css_solvent(E_cat_MeOAn_sol, E_MeOAn_sol, E_anion_NDI_sol, E_NDI_sol, r_MeOAn_NDI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_NDI_sol* u.AA).to(u.m), eps_sta_tol, eps_optical_tol)
    get_css_3_sol_diChMe = css_solvent(E_cat_MeOAn_sol, E_MeOAn_sol, E_anion_NDI_sol, E_NDI_sol, r_MeOAn_NDI_sol, (ionic_radius_MeOAn_sol* u.AA).to(u.m), (ionic_radius_NDI_sol* u.AA).to(u.m), eps_sta_chlorobenzene, eps_optical_chlorobenzene)

    # output results
    print("energy of CCS MeOAn+ ANI- NDI vacuum:", get_css_1_vac)
    print("energy of CCS MeOAn+ ANI- NDI solution:", get_css_1_sol)     
    print("energy of CCS MeOAn+ ANI NDI- vacuum:", get_css_3_vac)
    print("energy of CCS MeOAn+ ANI NDI- solution:", get_css_3_sol) 


if __name__ == "__main__":
    main()


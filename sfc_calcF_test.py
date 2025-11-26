import torch
import gemmi
import numpy as np
import pandas as pd
from SFC_Torch.Fmodel import F_protein
from SFC_Torch.symmetry import generate_reciprocal_asu, expand_to_p1
from SFC_Torch.utils import assert_tensor, aniso_scaling, vdw_rad_tensor, unitcell_grid_center, bin_by_logarithmic
from SFC_Torch.mask import reciprocal_grid, rsgrid2realmask, realmask2Fmask
from SFC_Torch.packingscore import packingscore_voxelgrid_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/minhuanli/deeprefine/blob/a03fc78916a7125792e680e065774d6a76506246/deeprefine/utils/models.py#L6
def plddt2pseudoB(plddt):
    '''
    Convert PLDDT to pseudo Bfactors, using equations from Beak et al. Science. 2021 Aug 20; 373(6557): 871–876.

    Args:
        plddt (torch.Tensor | np.ndarray): The tensor/array of plddt value

    Returns:
        torch.Tensor | np.ndarray : The converted pseudo B factors 
    '''
    if isinstance(plddt, torch.Tensor):
        rmsd = 1.5 * torch.exp(4. * (0.7 - 0.01 * plddt))
        pseudoB = 8.*np.pi**2 * rmsd**2 / 3.
    elif isinstance(plddt, np.ndarray):
        rmsd = 1.5 * np.exp(4. * (0.7 - 0.01 * plddt))
        pseudoB = 8.*np.pi**2 * rmsd**2 / 3.
    else:
        raise ValueError("plddt should be torch.Tensor or np.ndarray!")
    return pseudoB

def calc_fprotein(
    HKL_array,
    Hasu_array,
    dr2asu_array,
    fullsf_tensor,
    R_G_tensor_stack,
    T_G_tensor_stack,
    orth2frac_tensor,
    atom_pos_frac,
    atoms_position_tensor=None,
    atoms_biso_tensor=None,
    atoms_aniso_uw_tensor=None,
    atoms_occ_tensor=None,
):
    """
    Calculate the structural factor from a single atomic model, without solvent masking

    Parameters
    ----------
    atoms_positions_tensor: 2D [N_atoms, 3] tensor or default None
        Positions of atoms in the model, in unit of angstrom; If not given, the model stored in attribute `atom_pos_orth` will be used

    atoms_biso_tensor: 1D [N_atoms,] tensor or default None
        Isotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_iso` will be used

    atoms_aniso_uw_tensor: 3D [N_atoms, 3, 3] tensor or default None
        Anisotropic B factors of each atoms in the model, in matrix form; If not given, the info stored in attribute `atoms_aniso_uw` will be used

    atoms_occ_tensor: 1D [N_atoms,] tensor or default None
        Occupancy of each atoms in the model; If not given, the info stored in attribute `atom_occ` will be used

    Returns
    -------
    Fprotein
    """

    Fprotein_asu = F_protein(
        Hasu_array,
        dr2asu_array,
        fullsf_tensor,
        R_G_tensor_stack,
        T_G_tensor_stack,
        orth2frac_tensor,
        atom_pos_frac,
        atoms_biso_tensor,
        atoms_aniso_uw_tensor,
        atoms_occ_tensor,
    )
    # if not HKL_array is None:
    #     Fprotein_HKL = Fprotein_asu[self.asu2HKL_index]
    #     Fmask_HKL = torch.zeros_like(Fprotein_HKL)
    #     return Fprotein_HKL
    # else:
    Fmask_asu = torch.zeros_like(Fprotein_asu)
    return Fprotein_asu

# exit()
# initialize the three tensors from Boltz (currently just reading in from output csv file of Boltz prediction, just for test purpose)
df = pd.read_csv('/global/homes/k/kminseo/boltz/src/boltz_predictions.csv')
df_last5 = df.iloc[:, -6:]  # This assumes the last 5 are x,y,z,bfactor,occupancy in order
atoms_position_tensor = torch.tensor(df_last5[['x', 'y', 'z']].values, dtype=torch.float32).to(device) # (N_atoms, 3)
atoms_biso_tensor = torch.tensor(df_last5['bfactor'].values, dtype=torch.float32).to(device) # (N_atoms,) 
# atoms_biso_bfactor_tensor = torch.tensor(df_last5['bfactor_token_bin'].values, dtype=torch.float32).to(device) # (N_atoms,) 
# this is not true B factor, just plddt scores. need to convert them to true B factor using plddt2pseudoB() function from deeprefine
atoms_biso_tensor = plddt2pseudoB(atoms_biso_tensor)
# atoms_biso_tensor = torch.rand_like(atoms_biso_tensor) * 100.0

atoms_occ_tensor = torch.tensor(df_last5['occupancy'].values, dtype=torch.float32).to(device) # (N_atoms,)
print(atoms_position_tensor.shape)
print(atoms_biso_tensor.shape)
print(atoms_occ_tensor.shape)

# save predicted Biso to csv 
atoms_biso_cpu = atoms_biso_tensor.detach().cpu()   # move back to numpy-accessible host memory
atoms_biso_np  = atoms_biso_cpu.numpy()             # shape (N_atoms,)
# atoms_biso_bfactor_cpu = atoms_biso_bfactor_tensor.detach().cpu()   # move back to numpy-accessible host memory
# atoms_biso_bfactor_np  = atoms_biso_bfactor_cpu.numpy()             # shape (N_atoms,)
atoms_position_cpu = atoms_position_tensor.detach().cpu()   # move back to numpy-accessible host memory
atoms_position_np  = atoms_position_cpu.numpy()             # shape (N_atoms,)

# one value per line
# with open('atoms_biso_random.csv', 'w') as f:
#     for val in atoms_biso_np:
#         f.write(f"{val}\n")

# with open('atoms_biso_xray_bfactor.csv', 'w') as f:
#     for val in atoms_biso_bfactor_np:
#         f.write(f"{val}\n")
        
# np.savetxt("atoms_random_coords.csv", atoms_position_np, delimiter=",")
        
# exit()

N_atoms = atoms_position_tensor.shape[0]
atoms_aniso_uw_tensor = torch.zeros((N_atoms, 3, 3), dtype=torch.float32).to(device)

# priors known from experiment
# unit_cell = gemmi.UnitCell(25.12, 39.50, 45.07, 90, 90, 90) # boltz prot.yaml
unit_cell = gemmi.UnitCell(30.52, 37.75, 39.37, 90, 90, 90) # 1dur.cif 
space_group = gemmi.SpaceGroup('P212121')
dmin = 2.5

Hasu_array = generate_reciprocal_asu(unit_cell, space_group, dmin, anomalous=False)
dHasu = unit_cell.calculate_d_array(Hasu_array).astype("float32")
dr2asu_array = 1.0 / (unit_cell.calculate_d_array(Hasu_array).astype("float32") ** 2) # equivalent to calculate_1_d2_array() which is not found in gemmi.UnitCell

HKL_array = None
# initialize atomic scattering factor
# protein sequence to atom names
# 1-letter to 3-letter amino acid code mapping
# 1-letter to 3-letter amino acid code mapping
aa1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

# Standard atom names
residue_atom_names = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

def sequence_to_atom_names(protein_seq):
    atom_names = []
    for aa in protein_seq:
        resname = aa1to3.get(aa.upper())
        if resname is None:
            raise ValueError(f"Unknown amino acid: {aa}")
        atom_names.extend(residue_atom_names[resname])
    return atom_names

# --- NEW: map protein atom name to chemical element ---
def atom_name_to_element(atom_name):
    """Map typical PDB atom name to element symbol for Gemmi."""
    name = atom_name.strip().upper()
    if name in ('N', 'O', 'S', 'C'):  # main chain or single letter, easy
        return name
    # Special cases for common heavy atom ions
    if name.startswith("FE"): return "Fe"
    if name.startswith("ZN"): return "Zn"
    if name.startswith("MG"): return "Mg"
    if name.startswith("CA"): return "C"   # backbone CA alpha (not calcium!)
    if name == "SG": return "S"
    if name.startswith("CL"): return "Cl"
    # For BN, CD1... basically all protein backbone/sidechain heavy atoms that are carbons
    if name[0] == "C": return "C"
    if name[0] == "O": return "O"
    if name[0] == "N": return "N"
    if name[0] == "S": return "S"
    # default/fallback
    return name[0]

# Input
# protein_seq = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"
protein_seq = "AYVINDSCIACGACKPECPVNCIQEGSIYAIDADSCIDCGSCASVCPVGAPNPED"

# Output
atom_name = sequence_to_atom_names(protein_seq)
atom_name_element = list(atom_name_to_element(name) for name in atom_name)
print(f"Total number of atoms: {len(atom_name)}")

print('dr2asu_array shape: ', dr2asu_array.shape)

unique_atom = set(atom_name)
element_symbols = list(set(atom_name_to_element(name) for name in unique_atom))

full_atomic_sf_asu = {}
for atom_type in element_symbols:
    element = gemmi.Element(atom_type)
    f0 = np.array(
        [element.it92.calculate_sf(dr2 / 4.0) for dr2 in dr2asu_array]
    )
    full_atomic_sf_asu[atom_type] = f0 # when anomalous = False
    
fullsf_tensor = torch.tensor(np.array([full_atomic_sf_asu[atom] for atom in atom_name_element]),device=device).type(torch.float32) # when anomalous = False
print(fullsf_tensor.shape)
# initialize space group related property
operations = space_group.operations()
R_G_tensor_stack = assert_tensor(
            np.array([np.array(sym_op.rot) / sym_op.DEN for sym_op in operations]),
            arr_type=torch.float32, device=device,
        )
T_G_tensor_stack = assert_tensor(
            np.array(
                [np.array(sym_op.tran) / sym_op.DEN for sym_op in operations]
            ),
            arr_type=torch.float32, device=device,
        )

# initialize unit cell related property
orth2frac_tensor = torch.tensor(unit_cell.fractionalization_matrix.tolist(), device=device).type(torch.float32)
atom_pos_orth = atoms_position_tensor.to(device)
atom_pos_frac = torch.einsum("n...x,yx->n...y", atom_pos_orth, orth2frac_tensor)

# Run main 
Fprotein = calc_fprotein(
    HKL_array,
    Hasu_array,
    dr2asu_array,
    fullsf_tensor,
    R_G_tensor_stack,
    T_G_tensor_stack,
    orth2frac_tensor,
    atom_pos_frac,
    atoms_position_tensor,
    atoms_biso_tensor,
    atoms_aniso_uw_tensor,
    atoms_occ_tensor
)

print(Fprotein)
print(Fprotein.dtype)
print(Fprotein.shape)

fprotein_flat = Fprotein.cpu().detach().numpy().flatten()
# np.savetxt("Fprotein_random_bfactor.csv", fprotein_flat, delimiter="\n")


########## F_solvent #############
def inspect_data(verbose=False, spacing=4.5):
    """
    Do an inspection of data, for hints about
    1. solvent percentage for mask calculation
    2. suitable grid size
    """
    # solvent percentage
    vdw_rad = vdw_rad_tensor(atom_name, device=device)
    uc_grid_orth_tensor = unitcell_grid_center(
        unit_cell, spacing=spacing, return_tensor=True, device=device
    )
    occupancy, _ = packingscore_voxelgrid_torch(
        atom_pos_orth,
        unit_cell,
        space_group,
        vdw_rad,
        uc_grid_orth_tensor,
    )
    solventpct = 1 - occupancy
    # grid size
    mtz = gemmi.Mtz(with_base=True)
    mtz.cell = unit_cell
    mtz.spacegroup = space_group
    mtz.set_data(Hasu_array)
    gridsize = mtz.get_size_for_hkl(sample_rate=3.0)
    if verbose:
        print("Solvent Percentage:", solventpct)
        print("Grid size:", gridsize)
    
    return solventpct, gridsize
        
def calc_fsolvent(
        solventpct=None,
        gridsize=None,
        dmin_mask=5.0,
        dmin_nonzero=3.0,
        exponent=10.0,
        Return=True,
    ):
        """
        Calculate the structure factor of solvent mask in a differentiable way

        Parameters
        ----------
        solventpct: 0 - 1 Float, default None
            An approximate value of volume percentage of solvent in the unitcell.
            run `inspect_data` before to use a suggested value

        gridsize: [Int, Int, Int], default None
            The size of grid to construct mask map.
            run `inspect_data` before to use a suggected value

        dmin_mask: np.float32, Default 6 angstroms.
            Minimum resolution cutoff, in angstroms, for creating the solvent mask

        Return: Boolean, default False
            If True, it will return the Fmask as the function output; Or It will just be saved in the `Fmask_asu` and `Fmask_HKL` attributes
        """

        if solventpct is None and gridsize is None:
            solventpct, gridsize = inspect_data()

        # Shape [N_HKL_p1, 3], [N_HKL_p1,]
        Hp1_array, Fp1_tensor = expand_to_p1(
            space_group,
            Hasu_array,
            Fprotein,
            dmin_mask=dmin_mask,
            unitcell=unit_cell,
            anomalous=False,
        )
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, gridsize)
        real_grid_mask = rsgrid2realmask(
            rs_grid, solvent_percent=solventpct, exponent=exponent, 
        )  # type: ignore
        Fmask_asu = realmask2Fmask(real_grid_mask, Hasu_array)
        zero_hkl_bool = torch.tensor(dHasu <= dmin_nonzero, device=device)
        Fmask_asu[zero_hkl_bool] = torch.tensor(
            0.0, device=device, dtype=torch.complex64
        )
        if Return:
            return Fmask_asu

Fmask = calc_fsolvent()
print('Fmask shape: ', Fmask.shape)
            
######## F_total per resolution bin ##########
# only use when we don't have data
n_bins = 10 # Number of resolution bins used in the reciprocal space (default in SFCalc class)
Nmin = 100
kiso=1.0
kmask=0.35
uaniso=[0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4]
kmasks = [torch.tensor(kmask).to(atom_pos_frac) for i in range(n_bins)]
kisos = [torch.tensor(kiso).to(atom_pos_frac) for i in range(n_bins)]
uanisos = [torch.tensor(uaniso).to(atom_pos_frac) for i in range(n_bins)]
reciprocal_cell = unit_cell.reciprocal()  # gemmi.UnitCell object
# [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
reciprocal_cell_paras = torch.tensor(
    [
        reciprocal_cell.a,
        reciprocal_cell.b,
        reciprocal_cell.c,
        np.cos(np.deg2rad(reciprocal_cell.alpha)),
        np.cos(np.deg2rad(reciprocal_cell.beta)),
        np.cos(np.deg2rad(reciprocal_cell.gamma)),
    ],
    device=device,
).type(torch.float32)

def calc_ftotal_bini(bin_i, index_i, HKL_array, Fprotein, Fmask, scale_mode=False):
    """
    calculate ftotal for bin i
    """
    if scale_mode:
        Fmask = Fmask.detach()
        Fprotein = Fprotein.detach()
    scaled_fmask_i = Fmask[index_i] * kmasks[bin_i] 
    # we only have one batch in first dimension for now 
    # Fmask[:, index_i] * kmasks[bin_i]
    
    print('index_i: ', index_i)
    print("kisos[bin_i]:", kisos[bin_i].item())
    print('uniso item: ', uanisos[bin_i])
    print('reciprocal_cell_paras: ', reciprocal_cell_paras)
    print("aniso_scaling:", aniso_scaling(
        uanisos[bin_i],
        reciprocal_cell_paras,
        HKL_array[index_i],
    ).shape)
    # print("Fprotein[index_i]:", Fprotein[index_i])
    # print("scaled_fmask_i:", scaled_fmask_i)

    fmodel_i = (
        kisos[bin_i].item()
        * aniso_scaling(
            uanisos[bin_i],
            reciprocal_cell_paras,
            HKL_array[index_i],
        )
        * (Fprotein[index_i] + scaled_fmask_i)
    )
    return fmodel_i

def calc_ftotal(bins=None, Return=True, scale_mode=False):
        """
        Calculate Ftotal = kiso * exp(-2*pi^2*s^T*Uaniso*s) * (Fprotein + kmask * Fmask)

        kiso, uaniso and kmask are stored for each resolution bin

        Parameters
        ----------
        bins: None or List[int], default None
            Specify which resolution bins to calculate the ftotal, if None, calculate for all

        Return: Boolean, default True
            Whether to return the results

        Returns
        -------
        torch.tensor, complex
        """
        if bins is None:
            bins = range(n_bins)

        assignments, edges = bin_by_logarithmic(dHasu, n_bins, Nmin)
        ftotal_asu = torch.zeros_like(Fprotein)
        for bin_i in bins:
            index_i = assignments == bin_i
            ftotal_asu[index_i] = calc_ftotal_bini(
                bin_i, index_i, Hasu_array, Fprotein, Fmask, scale_mode=scale_mode
            )
        if Return:
            return ftotal_asu

Fmodel = calc_ftotal()
print(Fmodel)
print(Fmodel.dtype)
print(Fmodel.shape)

fmodel_flat = Fmodel.cpu().detach().numpy().flatten()
np.savetxt("Fmodel_my_pred.csv", fmodel_flat, delimiter="\n")
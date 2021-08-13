from pathlib import Path
import timemachine
from rdkit import Chem

root = Path(timemachine.__file__).parent.parent
path_to_hif2a = root.joinpath('datasets/fep-benchmark/hif2a')
ligand_sdf = str(path_to_hif2a.joinpath('ligands.sdf').resolve())
protein_pdb = str(path_to_hif2a.joinpath('5tbm_prepared.pdb').resolve())


def get_ligands():
    print(f'loading ligands from {ligand_sdf}')
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mols = [x for x in suppl]
    return mols
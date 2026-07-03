from Bio import PDB
from Bio.PDB import PDBParser, Select
import numpy as np

class AntibodyExtractor:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def extract_chains(self, pdb_path, h_chain_id, l_chain_id):
        """
        Extracts the VH and VL coordinates and sequences from the given PDB file.
        """
        structure = self.parser.get_structure("complex", pdb_path)
        # Implementation for extracting antibody residues
        model = structure[0]

        vh_coords = []
        vl_coords = []
        vh_sequence = []
        vl_sequence = []

        for chain in model:
            cid = chain.id.strip()
            if cid == h_chain_id:
                for residue in chain:
                    if residue.id[0] == ' ':  # 排除 HETATM
                        vh_coords.append(residue['CA'].coord)
                        vh_sequence.append(residue.resname)
            elif cid == l_chain_id:
                for residue in chain:
                    if residue.id[0] == ' ':
                        vl_coords.append(residue['CA'].coord)
                        vl_sequence.append(residue.resname)

        return {
            'vh_coords': np.array(vh_coords),
            'vl_coords': np.array(vl_coords),
            'vh_seq': self._residue_to_aa(vh_sequence),
            'vl_seq': self._residue_to_aa(vl_sequence),
            'h_chain_id': h_chain_id,
            'l_chain_id': l_chain_id
        }
    
    def _residue_to_aa(self, res_list):
        """将3字母残基转为1字母"""
        aa_map = {
            'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
            'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
            'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
            'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
        }
        return ''.join([aa_map.get(r, 'X') for r in res_list])
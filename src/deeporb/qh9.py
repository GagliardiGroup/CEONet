import os
import pyscf
from pyscf import gto, dft
from QH9.datasets import QH9Stable, QH9Dynamic
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from argparse import Namespace

convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
            [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
}

def matrix_transform(hamiltonian, atoms, convention='pyscf_def2svp'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new

def get_orbital_mask(atom_idx):
    idx_1s_2s = torch.tensor([0, 1])
    idx_2p = torch.tensor([3, 4, 5])
    orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
    orbital_mask_line2 = torch.arange(14)
    orbital_mask = orbital_mask_line1 if atom_idx <= 2 else orbital_mask_line2
    return orbital_mask

def build_final_matrix(data, diagonal_matrix, non_diagonal_matrix):
    final_matrix = []
    matrix_block_col = []
    non_diagonal_idx = 0
    for src_idx in range(data.atoms.shape[0]):
        matrix_col = []
        for dst_idx in range(data.atoms.shape[0]):
            if src_idx == dst_idx:
                matrix_col.append(diagonal_matrix[src_idx].index_select(
                    -2, get_orbital_mask(data.atoms[dst_idx].item())).index_select(
                    -1, get_orbital_mask(data.atoms[src_idx].item())
                ))
            else:
                matrix_col.append(
                    non_diagonal_matrix[non_diagonal_idx].index_select(
                        -2, get_orbital_mask(data.atoms[dst_idx].item())).index_select(
                        -1, get_orbital_mask(data.atoms[src_idx].item())))
                non_diagonal_idx = non_diagonal_idx + 1

        matrix_block_col.append(torch.cat(matrix_col, dim=-2))
    final_matrix.append(torch.cat(matrix_block_col, dim=-1))
    final_matrix = torch.stack(final_matrix, dim=0)
    return final_matrix

def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients

class QH9data():
    def __init__(self,root='datasets'):
        self.dataset = QH9Stable(root=root)

    def get(self,i):
        #get mol
        data = self.dataset[i]
        data.atoms = data.atoms.squeeze()
        t = [[data.atoms[atom_idx].cpu().item(), data.pos[atom_idx].cpu().numpy()]
             for atom_idx in range(data.num_nodes)]
        mol = pyscf.gto.Mole()
        mol.build(atom=t, basis='def2svp', unit='ang')
        
        h_not_pyscf = build_final_matrix(data, data.diagonal_hamiltonian, data.non_diagonal_hamiltonian)
        h_pyscf = matrix_transform(h_not_pyscf, data.atoms, convention='back2pyscf')

        s_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        mo_e, mos = cal_orbital_and_energies(s_pyscf, h_pyscf)

        h_pyscf = h_pyscf.squeeze().numpy()
        mo_e = mo_e.squeeze().numpy()
        mos = mos.squeeze().numpy()
        nocc = mol.nelectron//2
        mo_occ = np.array([2]*nocc + [0]*(mol.nao-nocc))
        
        return mol, h_pyscf, mo_e, mos, mo_occ
"""Resolve explicit source-ligand atom pairs; never infer correspondence by order."""
from collections.abc import Mapping

import numpy as np


def normalize_ligand_pairs(pairs):
    if pairs is None:
        return []
    if not isinstance(pairs, (list, tuple)):
        raise ValueError('coupled_ligand_atom_pairs must be a list of track_1/track_2 selectors')
    result = []
    seen = [set(), set()]
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != {'track_1', 'track_2'}:
            raise ValueError('Each ligand pair requires exactly track_1 and track_2')
        clean = {}
        for i, track in enumerate(('track_1', 'track_2')):
            selector = pair[track]
            if not isinstance(selector, Mapping) or set(selector) != {'chain', 'residue', 'atom'}:
                raise ValueError('Atom selectors require chain, residue (integer), and atom')
            chain, residue, atom = (selector[k] for k in ('chain', 'residue', 'atom'))
            if not isinstance(chain, str) or not chain or not isinstance(atom, str) or not atom:
                raise ValueError('Ligand chain and atom names must be nonempty strings')
            if isinstance(residue, bool) or not isinstance(residue, int):
                raise ValueError('Ligand residue must be an integer source residue ID')
            key = (chain, residue, atom)
            if key in seen[i]:
                raise ValueError(f'Duplicate ligand atom selector in {track}: {key}')
            seen[i].add(key)
            clean[track] = dict(chain=chain, residue=residue, atom=atom)
        result.append(clean)
    return result


def resolve_ligand_pairs(pairs, source_1, source_2, example_1, example_2):
    """Validate the induced molecular subgraph, allowing different substituents.

    Selectors refer to source CIF chain/residue/atom identifiers. Native RFD
    concatenation may compact chain IDs; gt_atom_name and source component or
    unique residue identity carry correspondence into the prepared AtomArray.
    Ambiguity is an error, never a first-match selection.
    """
    pairs = normalize_ligand_pairs(pairs)
    indices = {'mapped_1': [], 'mapped_2': [], 'all_1': [], 'all_2': []}
    if not pairs:
        return indices, {}
    raw_indices = [[], []]
    boundary = [[], []]
    for track, source, example in ((1, source_1, example_1), (2, source_2, example_2)):
        if source is None or source.bonds is None:
            raise ValueError('Coupled ligand atoms require source structures with typed bonds')
        atoms = example['atom_array']
        ligand = atoms.is_ligand.astype(bool)
        indices[f'all_{track}'] = np.flatnonzero(ligand).tolist()
        fixed = np.asarray(example['feats']['is_motif_atom_with_fixed_coord'], dtype=bool)
        chemistry = np.asarray(example['feats']['is_motif_atom_with_fixed_seq'], dtype=bool)
        for pair in pairs:
            sel = pair[f'track_{track}']
            match = np.flatnonzero((source.chain_id == sel['chain']) & (source.res_id == sel['residue']) &
                                   (source.atom_name == sel['atom']))
            if len(match) != 1:
                raise ValueError(f'Ligand source selector is missing or ambiguous: {sel}')
            raw = int(match[0])
            candidate = ligand & (atoms.gt_atom_name == sel['atom']) & (atoms.res_name == source.res_name[raw])
            if 'src_component' in atoms.get_annotation_categories():
                by_source = candidate & (atoms.src_component == f"{sel['chain']}{sel['residue']}")
                if by_source.any():
                    candidate = by_source
            found = np.flatnonzero(candidate)
            if len(found) != 1:
                raise ValueError(f'Prepared ligand selector is missing or ambiguous: {sel}')
            index = int(found[0])
            if atoms.chain_id[index] == 'A' and atoms.is_protein[index]:
                raise ValueError('Ligand coupling cannot select protein atoms')
            if fixed[index] or not chemistry[index]:
                raise ValueError('Coupled ligand atoms must have free coordinates and fixed chemistry')
            if index in indices[f'mapped_{track}']:
                raise ValueError('Two source selectors resolved to the same prepared ligand atom')
            indices[f'mapped_{track}'].append(index)
            raw_indices[track-1].append(raw)
    for a, b in zip(*raw_indices):
        if source_1.element[a] != source_2.element[b]:
            raise ValueError('Mapped ligand elements differ')
        for annotation in ('charge',):
            if annotation in source_1.get_annotation_categories() and annotation in source_2.get_annotation_categories():
                if getattr(source_1, annotation)[a] != getattr(source_2, annotation)[b]:
                    raise ValueError('Mapped ligand formal charges differ')
    graphs = []
    for track, (source, selected) in enumerate(zip((source_1, source_2), raw_indices)):
        lookup = {index: position for position, index in enumerate(selected)}
        graph = {}
        for a, b, bond in source.bonds.as_array():
            a, b, bond = int(a), int(b), int(bond)
            if a in lookup and b in lookup:
                graph[tuple(sorted((lookup[a], lookup[b])))] = bond
            elif (a in lookup) != (b in lookup):
                inside, outside = (a, b) if a in lookup else (b, a)
                boundary[track].append(dict(pair_index=lookup[inside],
                    outside_atom=str(source.atom_name[outside]), bond_type=bond))
        graphs.append(graph)
    if graphs[0] != graphs[1]:
        raise ValueError('Mapped ligand subgraphs differ in connectivity or bond order')
    metadata = dict(atom_pairs=pairs, mapped_atom_count=len(pairs),
                    indices=indices, boundary_bonds={'track_1': boundary[0], 'track_2': boundary[1]},
                    policy='same cutoff and mixing coefficient as protein; shared noise remains after release',
                    kappa_solve_policy='protein-only kappa solve; its scalar also mixes mapped ligand updates',
                    noise_policy='mapped ligand churn reuses track 1 native draws; original protein RNG draws unchanged')
    return indices, metadata

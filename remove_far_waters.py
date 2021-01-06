import argparse
import numpy as np
import mdtraj as md
import itertools
import os
import json


class RemoveWaters:
    '''
    Remove all waters of a protein simulation that are not in the proximity of a given residue, e.g. protein residue or ligand.

    Attributes
    ----------
    traj : md.Trajectory
        An mdtraj trajectory. Must contain topology information.
    sel_query : str or list
        Selection string for atoms which neighbouring waters are kept. Or list
        of atom ids.
    sel : str or list
        Selection string for extracted subsystem. Or list of atom ids.
    n_waters : int or None, default=100
        Number of waters kept in output trajectory. Can be smaller due to cutoff.
    cutoff : float, default=1.
        Distance cutoff for md.compute_neighbors() in nm. If less water
        molecules than n_waters are found in cutoff distance,
        this smaller number of water molecules will be kept in output trajectory.
    water_type : str
        Water type (tip3p, tip4p).
    verbose : bool
        verbose mode.

    Methods
    -------
    static_search()
        Remove waters based on distance in the first frame only.
    dynamic_search()
        Remove waters based on dynamic distance throughout the trajectory. Water identity is lost.
    dynamic_zero()
        Set the location of every water molecule further away than cutoff to the origin of the simulation box.
    '''

    def __init__(self,
                 traj,
                 sel_query,
                 sel='protein',
                 n_waters=100,
                 cutoff=1.,
                 water_type='tip3p',
                 verbose=False):
        '''See class documentation.'''
        self.traj = traj
        self.top = self.traj.top
        self.n_frames = self.traj.n_frames
        self.n_atoms = self.traj.n_atoms

        self.verbose = verbose

        if self.verbose:
            print(f'Trajectory with {self.n_frames} frames and {self.n_atoms} atoms.')

        self.traj_new_static = None
        self.traj_new_dynamic = None
        self.traj_new_dynamic_zero = None

        # water atoms from water type
        with open('water_types.json', 'r') as f:
            self.water_atoms = json.loads(f.read())[water_type]
        self.n_water_atoms = len(self.water_atoms)

        # either cutoff or n_waters must be given
        if n_waters is None and cutoff is None:
            raise ValueError('n_waters and cutoff cannot both be None.')

        self.sel = sel
        self.sel_query = sel_query
        self.n_waters = n_waters
        self.cutoff = cutoff

        # get indices of sel and sel_query
        if isinstance(self.sel, str):
            self.sel_ids = self.top.select(self.sel)
        elif isinstance(self.sel, list):
            self.sel_ids = self.sel
        else:
            raise TypeError(f'sel is {type(sel)}. Must be list or str.')

        if isinstance(self.sel_query, str):
            self.query_ids = self.top.select(self.sel_query)
        elif isinstance(self.sel_query, list):
            self.query_ids = self.sel_query
        else:
            raise TypeError(f'sel is {type(sel)}. Must be list or str.')

        # water atom indices
        # np.concatenate broadcasts shape to (n_water_atoms,)
        self.all_water_ids = [[a.index for a in r.atoms] for r in self.top.residues if r.is_water]
        self.all_water_ids = np.concatenate(self.all_water_ids)

        # water residue indices
        self.all_water_res = np.array([r.index for r in self.top.residues if r.is_water])

        if self.verbose:
            print(f'{len(self.all_water_ids)} water atoms in trajectory.')

        # ensure water molecules are in correct order for water type
        w_atoms = [self.top.atom(x) for x in self.all_water_ids[:self.n_water_atoms]]
        if not all([w_atoms[i].name == self.water_atoms[i]['name'] for i in range(self.n_water_atoms)]):
            raise ValueError(
                f'Water residues in trj are {[x.name for x in w_atoms]}, must be {[x["name"] for x in self.water_atoms]}')

    def static_search(self):
        '''Remove waters based on distance in the first frame only.'''
        # search neighbouring water molecules to query within cutoff in first frame
        neighbour_water_ids = md.compute_neighbors(self.traj[0],
                                                   cutoff=self.cutoff,
                                                   query_indices=self.query_ids,
                                                   haystack_indices=self.all_water_ids,
                                                   periodic=True)[0]

        neighbour_water_res = self._residue_from_id(neighbour_water_ids)

        water_count = len(neighbour_water_res)
        if water_count <= self.n_waters or self.n_waters is None:
            # restrict number of waters to the number found within the cutoff distance
            self.n_waters = water_count

            if self.verbose:
                print(f'Found {self.n_waters} water molecules within {self.cutoff} nm.')

            # no distance calculation if n_waters <= water_count
            closest_water_res = neighbour_water_res
            closest_water_ids = []
            for r in closest_water_res:
                for a in r.atoms:
                    closest_water_ids.append(a.index)

        else:
            if self.verbose:
                print(
                    f'There are more than {self.n_waters} water molecules within cutoff. Saving the closest {self.n_waters}.')

            # calculate distances
            distances = np.full(water_count, 99999.)
            for i_wat, wat in neighbour_water_res:
                # minimum distance between any water atom and any query atom
                pairs_itt = itertools.product(self.query_ids, [a.index for a in wat.atoms])
                pairs = np.array(list(pairs_itt))

                dist = md.compute_distances(traj=self.traj[0],
                                            atom_pairs=pairs)
                distances[i_wat] = min(dist[0])

            # find n_waters closest waters
            closest_water_ids = []
            i_wats_closest = np.argpartition(distances, self.n_waters)

            for i_wat in i_wats_closest:
                for a in neighbour_water_res[i_wat].atoms:
                    closest_water_ids.append(a.index)

        # make new trajectory from sel and closest waters
        allowed_ids = np.concatenate([self.sel_ids, closest_water_ids])

        self.traj_new_static = self.traj.atom_slice(allowed_ids)

        return self.traj_new_static

    def dynamic_search(self):
        '''Remove waters based on dynamic distance throughout the trajectory. Water identity is lost.'''
        # pairs of query_ids with all water atom ids
        pairs_itt = itertools.product(self.query_ids, self.all_water_ids)
        pairs = np.array(list(pairs_itt))

        # map water residues to pairs
        res_of_pairs = np.zeros(len(pairs), dtype=int)
        for i_pair, pair in enumerate(pairs):
            a = self.top.atom(pair[1])
            res_of_pairs[i_pair] = a.residue.index

        # distance calculation
        # returns distances of shape (n_frames, n_pairs)
        dist = md.compute_distances(traj=self.traj,
                                    atom_pairs=pairs)

        if self.verbose:
            print('Distance calculation completed.')

        # calculate minimum distance of neighbour residues to query
        dist_res = np.array([dist[:, res_of_pairs == rid] for rid in self.all_water_res])
        min_dist_res = np.amin(dist_res, axis=2)
        trj_water_dist = min_dist_res.T

        if self.verbose:
            print('Minimum distance to water residues calculated.')

        # water residues sorted per distance in frame
        # distances corresponding to water residues sorted
        trj_water_res_sorted = np.zeros((self.n_frames, len(self.all_water_res)), dtype=int)
        trj_water_dist_sorted = np.full((self.n_frames, len(self.all_water_res)), 9999.)

        for i_frame, frame in enumerate(trj_water_dist):
            trj_water_dist_sorted[i_frame] = frame[np.argsort(frame)]
            trj_water_res_sorted[i_frame] = self.all_water_res[np.argsort(frame)]

        # find maximum number of water molecules in cutoff
        n_wat_cutoff = np.zeros(self.n_frames, dtype=int)
        for i_frame, frame in enumerate(trj_water_dist_sorted):
            max_n = len(frame[frame < self.cutoff])
            n_wat_cutoff[i_frame] = max_n

        # update self.n_waters
        self.n_waters = max(n_wat_cutoff)
        if self.verbose:
            print(
                f'Found a maximum of {self.n_waters} water molecules within a cutoff of {self.cutoff} nm.')

        # closest self.n_waters water residues to query_ids for each frame in correct order
        closest_water_res = trj_water_res_sorted[:, :self.n_waters]

        # atom indices of closest water atoms
        closest_water_ids = np.zeros((self.n_frames, self.n_waters * self.n_water_atoms), dtype=int)
        for i_frame, frame in enumerate(closest_water_res):
            residues = [self.top.residue(x) for x in frame]
            a_ids = []
            for r in residues:
                [a_ids.append(a.index) for a in r.atoms]
            closest_water_ids[i_frame] = np.array(a_ids)

        # atom slice topology and make new trajectory xyz
        t_sel = self.traj.atom_slice(self.sel_ids)
        top_new = t_sel.top.copy()

        # add chain with water molecules
        top_new.add_chain()
        for i in range(self.n_waters):
            top_new.add_residue('HOH', top_new.chain(top_new.n_chains - 1), resSeq=i + 1)

        for i in range(self.n_waters):
            for w_atom in self.water_atoms:
                top_new.add_atom(w_atom['name'],
                                 w_atom['element'],
                                 residue=top_new.chain(top_new.n_chains - 1).residue(i))

        # make new xyz matrix
        xyz_new = np.zeros((t_sel.xyz.shape[0],
                            t_sel.xyz.shape[1] + self.n_waters * self.n_water_atoms,
                            t_sel.xyz.shape[2]))
        xyz_new[:t_sel.xyz.shape[0], :t_sel.xyz.shape[1], :t_sel.xyz.shape[2]] = t_sel.xyz

        # make new trajectory
        traj_new = md.Trajectory(xyz=xyz_new,
                                 topology=top_new, time=self.traj.time,
                                 unitcell_lengths=self.traj.unitcell_lengths,
                                 unitcell_angles=self.traj.unitcell_angles)

        # set coordinates of water molecules to closest waters
        for frame in range(self.traj.n_frames):
            traj_new.xyz[frame][len(self.sel_ids):] = self.traj.xyz[frame][closest_water_ids[frame]]

        self.traj_new_dynamic = traj_new
        return traj_new

    def dynamic_zero(self):
        '''Set the location of every water molecule further away than cutoff to the origin of the simulation box.'''
        # get atom ids for each frame that get closer to query_ids than cutoff
        trj_neighbour_water_ids = md.compute_neighbors(
            self.traj, cutoff=self.cutoff, query_indices=self.query_ids, haystack_indices=self.all_water_ids, periodic=False)

        # complete to full water molecules
        for frame in trj_neighbour_water_ids:
            frame = self._complete_to_full_res(frame)

        # mask for neighbour ids + selection ids
        neg_neighbour_masks = []
        for frame in trj_neighbour_water_ids:
            mask = np.ones(self.traj.n_atoms, dtype=bool)
            mask[frame] = False
            mask[self.sel_ids] = False
            neg_neighbour_masks.append(mask)

        # copy trj and set all coordinates of atoms not in trj_neighbour_ids to zero
        traj_new = self.traj

        for i, frame in enumerate(traj_new.xyz):
            frame[neg_neighbour_masks[i]] = np.zeros(3, dtype=int)

        return traj_new

    def save_closest_water_dynamic(self, dest):
        if os.path.isdir(dest):
            fname = os.path.join(dest, 'closest_water_dynamic.txt')
        else:
            fname = dest
        np.savetxt(fname, self.closest_water_res, fmt='%i')

    def _residue_from_id(self, id_list):
        res_list = []
        for id in id_list:
            a = self.top.atom(id)
            r = a.residue
            if r not in res_list:
                res_list.append(r)
        return res_list

    def _complete_to_full_res(self, id_array):
        res_list = self._residue_from_id(id_array)

        full_atom_list = []
        for r in res_list:
            for a in r.atoms:
                full_atom_list.append(a)

        full_id_array = np.zeros(len(full_atom_list))
        for i, a in enumerate(full_atom_list):
            full_id_array[i] = a.index

        return full_id_array


def main():
    pass


if __name__ == '__main__':
    main()

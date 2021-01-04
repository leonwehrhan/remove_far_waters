'''
Script for removing all waters of a protein simulation that are not in the proximity of a given residue, e.g. protein residue or ligand.
'''

import argparse
import numpy as np
import mdtraj as md
import itertools
import os
import json


water_types = {'tip3p': 3,
               'spc': 3,
               'tip4p': 4,
               'tip5p': 5}


class RemoveWaters:
    '''
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

        self.verbose

        if self.verbose:
            print(f'Trajectory with {self.traj.n_frames} frames and {self.traj.n_atoms} atoms.')

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

        if self.verbose:
            print(f'{len(self.all_water_ids)} water atoms in trajectory.')

    def static_search(self):
        '''Remove waters based on distance in the first frame only.'''
        # search neighbouring water molecules to query within cutoff in first frame
        neighbour_water_ids = md.compute_neighbors(self.traj[0],
                                                   cutoff=self.cutoff,
                                                   query_indices=self.query_ids,
                                                   haystack_indices=self.all_water_ids,
                                                   periodic=False)[0]

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

    def dynamic_search(self, water_type='tip3p'):
        '''Remove waters based on dynamic distance throughout the trajectory. Water identity is lost.'''
        if water_type == 'tip3p':
            self.water_atoms = 3
        elif water_type == 'tip4p':
            self.water_atoms = 4
        else:
            raise ValueError(f'Invalid value for water_type {water_type}. Use "tip3p" or "tip4p".')

        # get water molecules that during the trajectory ever get closer to query_ids than cutoff
        trj_neighbour_water_ids = md.compute_neighbors(
            self.traj, cutoff=self.cutoff, query_indices=self.query_ids, haystack_indices=self.all_water_ids, periodic=False)
        trj_neighbour_water_res = []
        for frame in trj_neighbour_water_ids:
            trj_neighbour_water_res.append(self._residue_from_id(frame))

        neighbour_water_res_flat = np.concatenate(trj_neighbour_water_res)
        neighbour_water_res_flat = list(set(neighbour_water_res_flat))
        print(
            f'There is a total of {len(neighbour_water_res_flat)} unique water molecules that ever get closer than {self.cutoff} nm to the query residue.')

        # restrict n_waters to minimum number of waters within cutoff per frame
        n_waters_per_frame = [len(frame) for frame in trj_neighbour_water_res]
        if any([x < self.n_waters for x in n_waters_per_frame]):
            self.n_waters = min(n_waters_per_frame)
            print(f'Found a minimum of {self.n_waters} water molecules within {self.cutoff} nm.')
        else:
            print(
                f'Found a minimum of {min(n_waters_per_frame)} water molecules within {self.cutoff} nm. Saving the closest {self.n_waters} water molecules.')

        # calculate distances between water molecules and query_res
        # QUERY_RES MUST ONLY HAVE ONE ENTRY
        res_pairs = list(itertools.product([r.index for r in self.query_res], [
                         r.index for r in neighbour_water_res_flat]))
        contacts = md.compute_contacts(self.traj, contacts=res_pairs, scheme='closest')

        contacts_zipped = []
        mapped_pairs = contacts[1]
        for frame in range(self.traj.n_frames):
            contacts_zipped.append(list(zip(contacts[0][frame], mapped_pairs)))
        del contacts

        contacts_sorted = []
        for frame in contacts_zipped:
            frame.sort(key=lambda x: x[0])
            frame_sorted = []
            for i, j in enumerate(frame):
                if j[1][0] == self.query_res[0].index:
                    frame_sorted.append([j[0], j[1][1]])
                elif j[1][1] == self.query_res[0].index:
                    frame_sorted.append([j[0], j[1][0]])
                else:
                    raise ValueError('Query_res not included in contact pair.')
            contacts_sorted.append(frame_sorted)
        contacts_sorted = np.array(contacts_sorted)
        self.contacts_sorted = contacts_sorted

        closest_water_res = []
        closest_water_ids = []
        for frame in range(self.traj.n_frames):
            res_closest_in_frame = list(contacts_sorted[frame][:, 1].astype(int))
            res_closest_in_frame = res_closest_in_frame[:self.n_waters]
            closest_water_res.append(res_closest_in_frame)

            id_closest_in_frame = []
            for ri in res_closest_in_frame:
                r = self.top.residue(ri)
                for a in r.atoms:
                    id_closest_in_frame.append(a.index)
            closest_water_ids.append(id_closest_in_frame)

        # closest_water_res = self._optimize_closest_water_res(closest_water_res)

        closest_water_res = np.array(closest_water_res)
        closest_water_ids = np.array(closest_water_ids)
        self.closest_water_res = closest_water_res

        # atom slice topology and make new trajectory xyz
        t_sel = self.traj.atom_slice(self.sel_ids)
        top_new = t_sel.top.copy()

        # add chain with water molecules
        top_new.add_chain()
        for i in range(self.n_waters):
            top_new.add_residue('HOH', top_new.chain(top_new.n_chains-1), resSeq=i+1)
        if water_type == 'tip3p':
            for i in range(self.n_waters):
                top_new.add_atom('O', md.element.oxygen, residue=top_new.chain(
                    top_new.n_chains-1).residue(i))
                top_new.add_atom('H1', md.element.hydrogen,
                                 residue=top_new.chain(top_new.n_chains-1).residue(i))
                top_new.add_atom('H2', md.element.hydrogen,
                                 residue=top_new.chain(top_new.n_chains-1).residue(i))
        elif water_type == 'tip4p':
            for i in range(self.n_waters):
                top_new.add_atom('O', md.element.oxygen, residue=top_new.chain(
                    top_new.n_chains-1).residue(i))
                top_new.add_atom('H1', md.element.hydrogen,
                                 residue=top_new.chain(top_new.n_chains-1).residue(i))
                top_new.add_atom('H2', md.element.hydrogen,
                                 residue=top_new.chain(top_new.n_chains-1).residue(i))
                top_new.add_atom('MW', md.element.virtual,
                                 residue=top_new.chain(top_new.n_chains-1).residue(i))
        # make new xyz matrix
        xyz_new = np.zeros((t_sel.xyz.shape[0], t_sel.xyz.shape[1] +
                            self.n_waters*self.water_atoms, t_sel.xyz.shape[2]))
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

    def _optimize_closest_water_res(self, closest_water_res):
        cw_new = []
        cw_new.append(np.array(closest_water_res[0]).astype('int'))
        for i in range(1, len(closest_water_res)):
            prev_frame = np.array(cw_new[i - 1]).astype('int')
            cur_frame = np.array(closest_water_res[i]).astype('int')
            cur_new = np.zeros(len(cur_frame))

            mask = np.in1d(cur_frame, prev_frame)
            mask_invert = np.in1d(cur_frame, prev_frame, invert=True)
            cur_inprev = cur_frame[mask]
            cur_notinprev = cur_frame[mask_invert]

            for x in cur_inprev:
                i = np.where(prev_frame == x)
                cur_new[i] = prev_frame[i]
            for x in cur_notinprev:
                for i, j in enumerate(cur_new):
                    if j == 0:
                        cur_new[i] = x
                        break
            cur_new = cur_new.astype('int')
            # if not np.in1d(cur_frame, prev_frame):
            #    pass
            cw_new.append(cur_new)

        cw_new = np.array(cw_new)
        return cw_new

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

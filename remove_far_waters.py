'''
Script for removing all waters of a protein simulation that are not in the proximity of a given residue, e.g. protein residue or ligand.
'''

import numpy as np
import mdtraj as md
import itertools
import os
import argparse


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
    sel_query : str
        Selection string for residue which neighbouring waters are kept.
        The selection must not contain atoms of more than one residue.
    sel : str
        Selection string for extracted subsystem.
    n_waters : int, default=100
        Number of waters kept in output trajectory. Can be smaller due to cutoff.
    cutoff : float, default=1.
        Distance cutoff for md.compute_neighbors() in nm. If less water
        molecules than n_waters are found in cutoff distance,
        this smaller number of water molecules will be kept in output trajectory.

    Methods
    -------
    static_search()
        Remove waters based on distance in the first frame only.
    dynamic_search()
        Remove waters based on dynamic distance throughout the trajectory. Water identity is lost.
    dynamic_zero()
        Set the location of every water molecule further away than cutoff to the origin of the simulation box.
    '''

    def __init__(self, traj, sel_query, sel='protein', n_waters=100, cutoff=1.):
        '''See class documentation.'''
        self.traj = traj

        self.traj_new_static = None
        self.traj_new_dynamic = None
        self.traj_new_dynamic_zero = None

        self.water_atoms = 3

        self.sel = sel
        self.sel_query = sel_query
        self.n_waters = n_waters
        self.cutoff = cutoff

        self.top = self.traj.top

        # get indices of sel and sel_query
        self.sel_ids = self.top.select(self.sel)
        self.query_ids = self.top.select(self.sel_query)
        self.query_res = self._residue_from_id(self.query_ids)
        if len(self.query_res) > 1:
            raise ValueError('sel_query includes indices of atoms for more than 1 residue.')

        # get water atom indices as list of lists for each water residue
        self.all_water_ids = [[a.index for a in r.atoms] for r in self.top.residues if r.is_water]
        self.all_water_ids = np.concatenate(self.all_water_ids)

    def static_search(self):
        '''Remove waters based on distance in the first frame only.'''
        # search neighbouring water molecules to query within cutoff in first frame
        neighbour_water_ids = md.compute_neighbors(self.traj[0],
                                                   cutoff=self.cutoff,
                                                   query_indices=self.query_ids,
                                                   haystack_indices=self.all_water_ids, periodic=False)[0]

        neighbour_water_res = [a.residue for a in [self.top.atom(i) for i in neighbour_water_ids]]
        neighbour_water_res = list(set(neighbour_water_res))

        water_count = len(neighbour_water_res)
        if water_count <= self.n_waters:
            # restrict number of waters to the number found within the cutoff distance
            self.n_waters = water_count
            print(f'found {self.n_waters} water molecules within {self.cutoff} nm.')

            # no distance calculation if n_waters < water_count
            closest_water_res = neighbour_water_res
            closest_water_ids = []
            for r in closest_water_res:
                for a in r.atoms:
                    closest_water_ids.append(a.index)

        else:
            # calculate distances
            res_pairs = list(itertools.product([r.index for r in self.query_res], [
                             r.index for r in neighbour_water_res]))
            contacts = md.compute_contacts(self.traj[0], contacts=res_pairs, scheme='closest')
            distances = contacts[0][0]
            mapped_pairs = contacts[1]
            del contacts
            del res_pairs

            # find n_waters closest waters
            closest_water_res = []
            closest_dist_idx = np.argpartition(distances, self.n_waters)
            closest_dist_idx = closest_dist_idx[:self.n_waters]
            for i in closest_dist_idx:
                cpair = mapped_pairs[i]
                if self.top.residue(cpair[1]).is_water:
                    closest_water_res.append(self.top.residue(cpair[1]))
                else:
                    closest_water_res.append(self.top.residue(cpair[0]))
            closest_water_ids = []
            for r in closest_water_res:
                for a in r.atoms:
                    closest_water_ids.append(a.index)

        # make new trajectory from sel and closest waters
        if self.del_ions:
            allowed_ids = np.concatenate([self.sel_ids, closest_water_ids])
        else:
            allowed_ids = np.concatenate([self.sel_ids, closest_water_ids, self.ion_ids])

        self.traj_new_static = self.traj.atom_slice(allowed_ids)

        return self.traj_new_static

    def dynamic_search(self, water_type='tip3p'):
        '''
        Remove waters based on dynamic distance throughout the trajectory. Water identity is lost.

        Parameters
        ----------
        water_type : str
            Water type (tip3p, tip4p, tip5p or spc).
        '''
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

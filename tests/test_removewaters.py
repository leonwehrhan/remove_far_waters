import pytest
import mdtraj as md
import numpy as np
from context import remove_far_waters

sample_trj = 'data/bc_1.xtc'
sample_ref = 'data/md_ref.pdb'


def test_load_trj():

    t = md.load(sample_trj, top=sample_ref)

    assert t.n_frames == 1001
    assert t.n_atoms == 40833

    assert t.top.residue(237).name + str(t.top.residue(237).resSeq) == 'LYS15'

    lys15_atoms = np.array([3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453])
    lys15_trj_atoms = t.top.select('resid 237')
    assert (lys15_atoms == lys15_trj_atoms).all()

    n_water_molecules_total = 0
    for r in t.top.residues:
        if r.is_water:
            n_water_molecules_total += 1
    assert n_water_molecules_total == 12235

    water_ids = []
    for r in t.top.residues:
        if r.is_water:
            for a in r.atoms:
                water_ids.append(a.index)
    assert len(water_ids) == 36705

    assert len(t.top.select('protein')) == 4114


def test_init():
    t = md.load(sample_trj, top=sample_ref)

    a = remove_far_waters.RemoveWaters(t,
                                       sel_query='resid 237',
                                       sel='protein',
                                       n_waters=100,
                                       cutoff=1.,
                                       del_ions=True,
                                       ions_list=None)

    assert len(a.query_res) == 1
    assert len(a.all_water_ids) == 36705
    assert len(a.sel_ids) == 4114

    assert type(a.query_res[0].index) == type(237)


def test_static_search_small_cutoff():
    t = md.load(sample_trj, top=sample_ref)

    a = remove_far_waters.RemoveWaters(t, sel_query='resid 237', sel='protein', n_waters=10, cutoff=0.5, del_ions=True, ions_list=None)
    a.static_search()

    assert a.traj_new_static.n_frames == 1001
    assert a.traj_new_static.n_atoms == 4114 + 5*3

    n_neighbour_waters = 0
    for r in a.traj_new_static.top.residues:
        if r.is_water:
            n_neighbour_waters += 1
    assert n_neighbour_waters == 5


def test_static_search_large_cutoff():
    t = md.load(sample_trj, top=sample_ref)

    a = remove_far_waters.RemoveWaters(t, sel_query='resid 237', sel='protein', n_waters=10, cutoff=1., del_ions=True, ions_list=None)
    a.static_search()

    assert a.traj_new_static.n_frames == 1001
    assert a.traj_new_static.n_atoms == 4114 + 10*3

    n_neighbour_waters = 0
    for r in a.traj_new_static.top.residues:
        if r.is_water:
            n_neighbour_waters += 1
    assert n_neighbour_waters == 10


def test_static_search_large_cutoff_large_nwaters():
    t = md.load(sample_trj, top=sample_ref)

    a = remove_far_waters.RemoveWaters(t, sel_query='resid 237', sel='protein', n_waters=100, cutoff=1., del_ions=True, ions_list=None)
    a.static_search()

    assert a.traj_new_static.n_frames == 1001
    assert a.traj_new_static.n_atoms == 4114 + 47*3

    n_neighbour_waters = 0
    for r in a.traj_new_static.top.residues:
        if r.is_water:
            n_neighbour_waters += 1
    assert n_neighbour_waters == 47


def test_dynamic_search():
    t = md.load(sample_trj, top=sample_ref)

    a = remove_far_waters.RemoveWaters(t, sel_query='resid 237', sel='protein', n_waters=10, cutoff=0.5, del_ions=True, ions_list=None)
    a.dynamic_search()

    assert a.contacts_sorted.shape == (1001, 2145, 2)
    assert a.closest_water_res_ids.shape == (1001, 2145)

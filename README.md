# remove_far_waters
Removes water from molecular trajectories that are far away from specified slection (e.g. inside a binding pocket). This reduced the size of the trajectory file.

The water molecules can be deleted dynamically, which means in every snapshot of the trajectory the far water molecules will be identified and deleted. The remaining water molecules get new indices. The water molecules can also be deleted statically, meaning the deleted water molecules are picked only based on the first snapshot of the trajectory. This is not recommended, but keeps the indices of the water molecules the same.

## Usage

```python
import remove_far_waters as rmw
import mdtraj as md

t = md.load('example.xtc', top='example.pdb')

W = rmw.RemoveWaters(t, 'index 1', 'example', 'protein', cutoff=0.5, water_type='tip3p')
t_new = W.dynamic_search()

t_new.save('example_reduced_water.xtc')
t_new[0].save('example_reduced_water.pdb')
```

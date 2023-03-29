# remove_far_waters

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

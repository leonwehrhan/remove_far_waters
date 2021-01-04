import json
import os


def add_water_type(name, atoms, f='water_types.json'):
    '''
    Add water type to water type database file.

    Parameters
    ----------
    f : str
        Path to water type database file.
    atoms : list of tuple
        List of atoms specified by (name, element).
    '''
    if os.path.exists(f):
        with open(f, 'r') as fi:
            water_types = json.loads(fi.read())
    else:
        water_types = {}

    water_type = []
    for atom in atoms:
        water_type.append({'name': atom[0], 'element': atom[1]})

    water_types[name] = water_type

    with open(f, 'w') as fi:
        fi.write(json.dumps(water_types))


def basic_types():
    '''Make water type database file with a few basic water types.'''
    # tip3p
    add_water_type('tip3p',
                   [('O', 'O'),
                    ('H1', 'H'),
                    ('H2', 'H')])
    # tip4p
    add_water_type('tip4p',
                   [('O', 'O'),
                    ('H1', 'H'),
                    ('H2', 'H'),
                    ('MW', 'VS')])


def main():
    basic_types()


if __name__ == '__main__':
    main()

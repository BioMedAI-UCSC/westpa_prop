from openmm.app import *
from openmm import *
from openmm.unit import *
# from sys import stdout
# import sys
import os
# import json
import numpy as np

# from rdkit import Chem
# from rdkit.Chem import AllChem

# from openff.toolkit import Molecule
# from openmmforcefields.generators import GAFFTemplateGenerator

# def add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path=None):
#     """
#     Add a GAFFTemplateGenerator to forcefield for the molecules listed in small_molecules.

#     Parameters:
#     - forcefield (openmm.app.ForceField): The forcefield object to add the generator to
#     - small_molecules_smiles (list(str)): A list of SMILES strings to build templates for
#     """
#     small_molecules = []
#     for smiles in small_molecules_smiles:
#         #TODO: Should we add hydrogens? It doesn't seem to impact matching...
#         template = Chem.MolFromSmiles(smiles)
#         template_mol = Molecule.from_rdkit(template, allow_undefined_stereo=True)
#         small_molecules.append(template_mol)

#     gaff = GAFFTemplateGenerator(molecules=small_molecules, cache=cache_path)
#     forcefield.registerTemplateGenerator(gaff.generator)

#     print(f"Added {len(small_molecules)} small molecule templates to forcefield")

def get_active_pdbid():
    pdbid_path = os.path.join(os.environ["WEST_SIM_ROOT"], "active_pdbid.txt")
    with open(pdbid_path, mode="rt", encoding="utf-8") as file:
        return file.read().strip()

RAND = int(os.environ["WEST_RAND16"])
print(f"RAND={RAND}")

with open(f"rand16.txt", mode="wt", encoding="utf-8") as file:
    file.write(str(RAND))

pdb = PDBFile('bstate.pdb')
# forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

# pdbid = get_active_pdbid()
# small_molecules_path = os.path.join(os.environ["WEST_SIM_ROOT"], f"../inputs/{pdbid}_processed_ligands_smiles.json")
# template_cache_path = os.path.join(os.environ["WEST_SIM_ROOT"], f"../inputs/{pdbid}_processed_ligands_cache.json")
# small_molecules = json.load(open(small_molecules_path, mode="r", encoding="utf-8"))
# add_ff_template_generator_from_smiles(forcefield, small_molecules, cache_path=template_cache_path)

device_index = '0'
if "WESTPA_GPU_LIST" in os.environ:
    gpu_list = [i for i in os.environ["WESTPA_GPU_LIST"].split(",")]
    device_index = gpu_list[int(os.environ["WEST_CURRENT_SEG_ID"]) % len(gpu_list)]

precision = 'single'
if "WESTPA_GPU_PRECICION" in os.environ:
    precision = os.environ["WESTPA_GPU_PRECICION"]

platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': precision,
                      'UseBlockingSync': 'true',
                      'DeviceIndex': device_index,
                      'DeterministicForces': 'true',
                      }

system = os.path.join(os.environ["WEST_SIM_ROOT"], "system.xml")
integrator = os.path.join(os.environ["WEST_SIM_ROOT"], "integrator.xml")
simulation = Simulation(pdb.topology, system, integrator, platform, platformProperties)
simulation.integrator.setRandomNumberSeed(RAND)

simulation.context.setPositions(pdb.positions)

simulation.loadState('parent.xml')
### Old dcd based code from ligand unbinding
# Disable checkpoints for protein_explore, we won't need to redensify these
# simulation.saveCheckpoint('start.chk')

# # simulation.reporters.append(StateDataReporter('seg.log', 100, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True)) 
# simulation.reporters.append(DCDReporter('seg.dcd', 500)) 

# simulation.step(1000)
# simulation.saveState('seg.xml')
###
total_steps = 1000
report_steps = 500
cur_steps = 0

times = []
forces = []
positions = []
energy_k = []
energy_u = []

assert total_steps % report_steps == 0, "total_steps must be divisible by report_steps"
for i in range(total_steps//report_steps):
    simulation.step(report_steps)
    state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
    forces.append(state.getForces(asNumpy=True).value_in_unit(kilojoules/mole/nanometer))
    positions.append(state.getPositions(asNumpy=True).value_in_unit(nanometer))
    times.append(state.getTime().value_in_unit(picoseconds))
    # What are the units for these
    energy_k.append(state.getKineticEnergy().value_in_unit(kilojoules/mole))
    energy_u.append(state.getPotentialEnergy().value_in_unit(kilojoules/mole))

times = np.array(times)
forces = np.array(forces)
positions = np.array(positions)
energy_k = np.array(energy_k)
energy_u = np.array(energy_u)
np.savez("seg.npz", times=times, forces=forces, positions=positions, energy_k=energy_k, energy_u=energy_u)

simulation.saveState('seg.xml')
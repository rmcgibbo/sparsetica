from simtk import unit
import simtk.openmm as mm
from simtk.openmm import app
from sys import stdout

prmtop = app.AmberPrmtopFile('ligand_1a.prmtop')
inpcrd = app.AmberInpcrdFile('ligand_1a.crd')


system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
     constraints=app.HBonds, rigidWater=True)
integrator = mm.LangevinIntegrator(290*unit.kelvin, 1.0/unit.picoseconds,
    2.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)



simulation = app.Simulation(prmtop.topology, system, integrator, platform=mm.Platform.getPlatformByName('CUDA'))
simulation.context.setPositions(inpcrd.positions)

print('Minimizing...')
simulation.minimizeEnergy()

N_steps = int(1*unit.microsecond/(2*unit.femtoseconds))
simulation.reporters.append(app.DCDReporter('trajectory.dcd', 10000))
simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=N_steps, separator='\t'))

print('Running Production...')
simulation.step(N_steps)

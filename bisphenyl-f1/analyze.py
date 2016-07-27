import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

t = md.load('trajectory.dcd', top='ligand_1a.prmtop')
t.time = 20*np.arange(len(t)) / 1e3

bonds, angles, dihedrals = [], [], []
for line in open('structure.intco'):
    line = line.strip()
    if line.startswith('R'):
        bonds.append(tuple(map(lambda x:int(x)-1, line[1:].split())))
    if line.startswith('B'):
        angles.append(tuple(map(lambda x:int(x)-1, line[1:].split())))
    if line.startswith('D'):
        dihedrals.append(tuple(map(lambda x:int(x)-1, line[1:].split())))


bonds = []
for a in t.top.atoms:
    for b in list(t.top.atoms)[a.index:]:
        if a.element.symbol == 'C' and b.element.symbol == 'C':
            bonds.append((a.index, b.index))


A = md.compute_distances(t, bonds)
B = md.compute_angles(t, angles)
C = np.sin(md.compute_dihedrals(t, dihedrals))
D = np.cos(md.compute_dihedrals(t, dihedrals))
X = np.hstack((A,B,C,D))
describe = bonds + angles + ['sin%s' % repr(s) for s in dihedrals] + ['cos%s' % repr(s) for s in dihedrals]
describe = np.asarray(describe)
mask = np.var(X, axis=0) > 1e-10

X = X[:, mask]
describe = describe[mask]
print('Data shape', X.shape)

X = np.concatenate([X[[0],:], X])
X = np.split(X, 5)
np.savez('coordinates.npz', X=X, describe=describe)


# from msmbuilder.decomposition import tICA
#
# plt.plot(tICA(n_components=1).fit([X]).eigenvectors_[:,0])
#
#
#
# from msmbuilder.decomposition import SparseTICA
# for rho in [1e-5, 1e-3, 1e-1, 1]:
#     stica = SparseTICA(n_components=1, rho=rho)
#     stica.fit([X])
#     print('rho', rho)
#     #print(stica.eigenvectors_[:,0])
#     where = np.where(stica.eigenvectors_[:,0])[0]
#     print(where)
#     for dof in where:
#         print(dof, [t.top.atom(i) for i in describe[dof]])
#     print()


# a = t.top.select('name C4')[0]
# b = t.top.select('name C5')[0]
# c = t.top.select('name C6')[0]
# d = t.top.select('name C7')[0]
# print(a,b,c,d)
#
# angle = md.compute_dihedrals(t, [[a,b,c,d]])[:,0]

#plt.figure(figsize=(10,6))
#plt.subplot(1,2,1)
#plt.plot(t.time, np.rad2deg(X[:, 182]), 'x')
#plt.hist(np.rad2deg(X[:, 182]), bins=100)

#plt.subplot(1,2,2)
#plt.plot(t.time[:10000], np.rad2deg(X[:10000, 190]), 'x')

#plt.xlabel('Time [ns]')
#plt.ylabel('Torsion [deg]')
# plt.savefig('dihedral.pdf')

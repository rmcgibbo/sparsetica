import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)

import matplotlib.pyplot as plt
from msmbuilder.decomposition import tICA
from msmbuilder.decomposition import SparseTICA
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import KFold
#from sklearn.utils import safe_indexing

if not os.path.exists('tica-results.npz'):
    X = np.load('coordinates.npz')['X']
    X = [X[0]]
    describe = np.load('coordinates.npz')['describe']

    eigvecs = []
    eigvals = []
    for rho in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
        model = SparseTICA(n_components=1, rho=rho).fit(X)
        eigvecs.append(model.eigenvectors_[:,0])
        eigvals.append(model.eigenvalues_[0])
    np.savez('tica-results.npz', eigvecs=np.array(eigvecs), eigvals=eigvals)
else:
    file = np.load('tica-results.npz')
    eigvecs = file['eigvecs']
    eigvals = file['eigvals']
    p = len(eigvecs[0])

plt.figure(figsize=(12,5))
plt.subplot(1,4,1)
plt.plot(eigvecs[0], color='#348ABD')
plt.text(0.05, 0.92, r'$\hat{\lambda}=%.4f$' % eigvals[0], transform=plt.gca().transAxes)
plt.xlim(0, p)
plt.xticks([0, 250, 500])
plt.title(r'$\mathrm{tICA}$')
plt.xlabel(r'$\mathrm{Feature}\;\mathrm{Index}$')
plt.ylabel(r'$\mathrm{Loading}$')


plt.subplot(1,4,2)
plt.title(r'$\mathrm{Sparse}\;\mathrm{tICA}$')
plt.plot(eigvecs[1], color='#A60628')
plt.text(0.05, 0.82, r'$\rho=10^{-4}$', transform=plt.gca().transAxes)
plt.text(0.05, 0.92, r'$\hat{\lambda}=%.4f$' % eigvals[1], transform=plt.gca().transAxes)
plt.xlim(0, p)
plt.xticks([0, 250, 500])
plt.xlabel(r'$\mathrm{Feature}\;\mathrm{Index}$')


plt.subplot(1,4,3)
plt.title(r'$\mathrm{Sparse}\;\mathrm{tICA}$')
plt.plot(eigvecs[2], color='#A60628')
plt.text(0.05, 0.82, r'$\rho=10^{-3}$', transform=plt.gca().transAxes)
plt.text(0.05, 0.92, r'$\hat{\lambda}=%.4f$' % eigvals[2], transform=plt.gca().transAxes)
plt.xlim(0, p)
plt.xticks([0, 250, 500])
plt.xlabel(r'$\mathrm{Feature}\;\mathrm{Index}$')


plt.subplot(1,4,4)
plt.title(r'$\mathrm{Sparse}\;\mathrm{tICA}$')
plt.plot(eigvecs[3], color='#A60628')
plt.text(0.05, 0.82, r'$\rho=10^{-2}$', transform=plt.gca().transAxes)
plt.text(0.05, 0.92, r'$\hat{\lambda}=%.4f$' % eigvals[3], transform=plt.gca().transAxes)
plt.xlim(0, p)
plt.xticks([0, 250, 500])
plt.xlabel(r'$\mathrm{Feature}\;\mathrm{Index}$')
#
# plt.subplot(1,5,5)
# plt.title(r'$\mathrm{Sparse}\;\mathrm{tICA}$')
# plt.plot(eigvecs[4], color='#A60628')
# plt.text(0.05, 0.82, r'$\rho=10^{-1}$', transform=plt.gca().transAxes)
# plt.text(0.05, 0.92, r'$\hat{\lambda}=%.4f$' % eigvals[4], transform=plt.gca().transAxes)
# plt.xlim(0, p)
# plt.ylim(-1.4, 1.4)
# plt.xticks([0, 250, 500])
# plt.xlabel(r'$\mathrm{Feature}\;\mathrm{Index}$')

plt.tight_layout()
plt.savefig('tics.pdf')
os.system('pdfcrop tics.pdf')

import scipy.linalg as la
import numpy

M = numpy.matrix('1 2; 2 1; 3 4; 4 3')

U, E, V = la.svd(M, full_matrices=False)
v = numpy.array(V).transpose()
print("U: " + str(U))
print("E: " + str(E))
print("V: " + str(V))
print()
Evals, Evectors = la.eigh(M.transpose() * M)

# sanity check
# print(numpy.all(((M.transpose()*M) * Evectors - Evals * Evectors) < 0.00000001))
# print(Evals)

# sorting of Evals in descending order
idx = Evals.argsort()[::-1]
Evals = Evals[idx]
Evectors = Evectors[:, idx]

#print(numpy.all(((M.transpose() * M) * Evectors - Evals * Evectors) < 0.00000001))
print("Eigenvalues " + str(Evals))
print("Eigenvectors " + str(Evectors))
print()

# E^2 equals Evals
print("E^2 equals Evals")
print("E^2= "+str(E * E))
print("Evals= "+str(Evals))

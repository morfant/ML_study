import numpy as np
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])

# print(np.ndim(A))
# print(A.shape)


# print(np.dot(A, B))

# A = np.array([[1, 2], [3, 4], [5, 6]])
# B = np.array([[7, 8]])

# print(A.shape)
# print(B.shape)

# C = np.dot(A, B)
# print(C)
# print(C.shape)

X = np.array([1, 2])
# X = np.array([[1, 2]])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)

Y = np.dot(X, W)
print(Y)
print(Y.shape)

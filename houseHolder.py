#!/usr/bin/env python3

import numpy as np

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

# task 1: show qr decomp of wp example
a = np.array(((
    (1,2,3),
    (4,5,6),
    (7,8,9),
)))

def autovalores(A):
    T = A
    pQ = np.eye(a.shape[0])
    for i in range(1, 10000, 1):
        Q, R = qr(T)
        T = np.matmul(R, Q)
        pQ = np.matmul(pQ, Q)

    return np.diag(T), pQ



diagonal, autovectores = autovalores(a)
q, r = qr(a)
print('diagonal:\n',diagonal)
print('------------------')
print('autovectores:\n',autovectores)
print('------------------')
print('q:\n', q.round(6))
print('r:\n', r.round(6))

# task 2: use qr decomp for polynomial regression example
def polyfit(x, y, n):
    return lsqr(x[:, None]**np.arange(n + 1), y.T)

def lsqr(a, b):
    q, r = qr(a)
    _, n = r.shape
    return np.linalg.solve(r[:n, :], np.dot(q.T, b)[:n])

x = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
y = np.array((1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321))

print('\npolyfit:\n', polyfit(x, y, 2))
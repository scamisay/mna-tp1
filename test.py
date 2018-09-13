#!/usr/bin/env python3

import numpy as np

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def eigImplementation(A):
    T = A
    pQ = np.eye(A.shape[0])

    err = np.inf
    tolerance = 1e-15

    t_prev = T[0,0]
    t_curr = T[0,0]

    while err > tolerance:
        Q, R = qr(T)
        T = np.matmul(R, Q)
        pQ = np.matmul(pQ, Q)
        t_curr = T[0,0]
        err = np.abs(t_prev - t_curr)
        t_prev = t_curr
        print ('+++++', err)
    print ('------', np.diag(T))
    return np.diag(T), pQ

def SVD(A):
    M = np.matmul(np.transpose(A), A)

    lambda_matrix, eigenvectors = eigImplementation(M)

    sigma = np.sqrt(lambda_matrix)
    V = np.zeros(eigenvectors.shape)

    for i in range(eigenvectors.shape[1]):
        V[:,1] = eigenvectors[:,i] / np.norm(eigenvectors[:,i])


    return sigma,V






# task 1: show qr decomp of wp example
a = np.array(((
    (1,2,3),
    (4,5,6),
    (7,8,9),
    (10,11,12),
)))

S, V = SVD(a)
print('S:\n',S)
print('V:\n',V)

diagonal, autovectores = eigImplementation(a)
q, r = qr(a)
print('diagonal:\n',diagonal)
print('autovectores:\n',autovectores)

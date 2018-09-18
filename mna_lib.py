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

def svd(A):
    M = np.matmul(np.transpose(A), A)

    lambda_matrix, eigenvectors = eigImplementation(M)
    r = min(A.shape)
    sigma = np.sqrt(abs(lambda_matrix[0:r]))
    V = np.zeros((A.shape[1],A.shape[1]))

    for i in range(sigma.size):
        V[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])

    U = np.zeros((A.shape[0], sigma.size))
    for i in range(sigma.size):
        U[:,i] = 1/sigma[i]*A.dot(V[:,i])

    S = np.zeros(A.shape)
    S[0:sigma.size,0:sigma.size] = np.diag(sigma)



    return U,S,V


def qr_wilkinson(B):
    A = B + np.zeros(B.shape)
    tol = 1e-2
    porc = 1e-3

    pQ = np.eye(A.shape[0])
    n = A.shape[0]
    for k in range(n, 1, -1):
        T = np.zeros((k,k))
        T = A[0:k, 0:k]
        mu = T[k - 1, k - 1]
        while abs(T[k - 1, k - 2]) > porc * (abs(T[k - 2, k - 2]) + abs(T[k - 1, k - 1])):
            W = T - mu*np.identity(T.shape[0])
            #Q, R = qr(W)
            Q, R = np.linalg.qr(W)
            T = np.matmul(R, Q) + mu*np.identity(T.shape[0])

            temp = pQ
            temp[:Q.shape[0], :Q.shape[1]] = Q
            pQ = np.matmul(pQ, temp)
        A[0:k, 0:k] = T[0:k, 0:k]
    return np.diag(A), pQ


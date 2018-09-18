from mna_lib import *

A = np.array(((
    (1,2,3,4,5,6),
    (7,8,9,10,11,12),
    (13,14,15,16,17,18),
    (19,20,21,22,23,24)
)))
n=5
#M = np.random.rand(n,n)*25
M = np.matmul(np.transpose(A),A)
D,V = qr_wilkinson(M)
print(D)
print('fin')
# Adaptive Smoothing Method as MC in Traffic State Estimation
import numpy as np
import random
import numpy.linalg as LA
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn.impute import KNNImputer

# Data input
#K_true = np.load('data/US101_lane2.npy')
K_true = np.load('data/HW25_lane4.npy')
#K_true = K_true[:,:300]
K_true = K_true[:,:1000]
K_shape = K_true.shape

### trajectory data
# parameters
h_act=67
max_speed=100

# output data
output_Y = np.load('data/out_data_Ngsim_us101_lane2.npy')
y_test = output_Y[:, :h_act, :, :]
y_map = (y_test.copy().squeeze(axis=-1))*max_speed
K_true = y_map[0]
#K_true = K_true[:,:600]
K_true = K_true[:,1500:]
K_shape = K_true.shape

# input data
input_X = np.load('data/inp_data_Ngsim_us101_lane2_3p.npy')
x_test = input_X[:, :h_act, :, :]
x_bin = (x_test.sum(axis=3) != 0)
x_map = (x_bin * y_map)*max_speed
Z = x_map[0]
#Z = Z[:,:600]
#Z = Z[:,1000:]
Mask_mat = (Z!=0).astype(int)

### detector data
random.seed(1)
Mask_col = np.random.choice([True, False], size=K_shape[0], p=[0.1, 0.9])
Mask_mat = np.zeros(K_shape)
for i in range(K_shape[1]):
    Mask_mat[:,i] = Mask_col
Z = K_true * Mask_mat



# Data preparation
random.seed(10)
Mask_mat = np.random.choice([True, False], size=K_shape, p=[0.03, 0.97])
Z = K_true * Mask_mat

J = np.ones(K_shape)
X = np.zeros(K_shape)
for j in range(K_shape[0]):
    X[j,:] = (K_shape[0]-j)*np.ones((1,K_shape[1]))
T = np.zeros(K_shape)
for j in range(K_shape[1]):
    T[:,j] = (j+1)*np.ones((K_shape[0],1)).ravel()
obs_num = int(sum(sum(Mask_mat)))
#obs_num = sum(sum(Mask_mat))
Phi_list = [0] * obs_num
index_list = [0] * obs_num
obsValue_list = [0] * obs_num
position_list = [0] * obs_num
counter = 0
for (i, j), element in np.ndenumerate(K_true):
    if Mask_mat[i, j]:
        obsValue_list[counter] = element
        index_list[counter] = [i, j]
        position_list[counter] = [K_shape[0] - i, j + 1]
        counter += 1

def smothing_kernel(X,Y,sigma,zeta):
    gram_matrix = np.zeros(X.shape)
    for (i, j), x in np.ndenumerate(X):
        y = Y[i,j]
        # gram_matrix[i, j] = np.exp(- np.power(x, 2) / (2 * sigma ** 2) - np.power(y, 2) / (2 * zeta ** 2))
        gram_matrix[i, j] = np.exp(- np.abs(x) / sigma - np.abs(y) /zeta)
    return gram_matrix

# congested case
c1 = -15 # km/h
# c = -4.17 # m/s
sigma = 0.6 # km
# sigma = 600
zeta = 1.1/60 # min
# zeta = 1.1*60
for i, x in enumerate(position_list):
    print(i)
    x_i = x[0]
    t_j = x[1]
    U = x_i * J - X
    #V = T - t_j * J - U/c1
    V = t_j * J - T - U / c1
    Phi_list[i] = smothing_kernel(U,V,sigma,zeta)

K_est_c = [a*b for (a,b) in zip(obsValue_list,Phi_list)]
K_est_c = np.divide(sum(K_est_c),sum(Phi_list))

# free flow case
Phi_list_f = [0] * obs_num
c = 80 #km/h
# c = 22.22
for i, x in enumerate(position_list):
    print(i)
    x_i = x[0]
    t_j = x[1]
    U = x_i * J - X
    #V = T - t_j * J - U/c
    V = t_j * J -T - U / c
    Phi_list_f[i] = smothing_kernel(U,V,sigma,zeta)
K_est_f = [a*b for (a,b) in zip(obsValue_list,Phi_list_f)]
K_est_f = np.divide(sum(K_est_f),sum(Phi_list_f))

# both case in the same loop
c1 = -15 # km/h
c = 80
sigma = 0.6 # km
zeta = 2.5 # min
Phi_list_f = [0] * obs_num
for i, x in enumerate(position_list):
    print(i)
    x_i = x[0]
    t_j = x[1]
    U = x_i * J - X
    V = t_j * J - T - U/c1
    V_f = t_j * J - T - U / c
    Phi_list[i] = smothing_kernel(U,V,sigma,zeta)
    Phi_list_f[i] = smothing_kernel(U, V_f, sigma, zeta)

K_est_c = [a*b for (a,b) in zip(obsValue_list,Phi_list)]
K_est_c = np.divide(sum(K_est_c),sum(Phi_list))
K_est_f = [a*b for (a,b) in zip(obsValue_list,Phi_list_f)]
K_est_f = np.divide(sum(K_est_f),sum(Phi_list_f))

# GASM: Weighting and summing step
#v_thr = np.mean(np.concatenate((K_est_c,K_est_f)))
v_thr = 60 #km/h
# v_thr = 60*1000/(60*60)
V_thr = v_thr*np.ones(K_shape)
v_tw = 20
# v_tw = 20*1000/(60*60)
W = (np.ones(K_shape)+np.tanh((V_thr-np.minimum(K_est_c, K_est_f))/v_tw))/2
Z_est_GASM = W * K_est_c + (1 - W) * K_est_f

# check the trend for congested and free flow component
nan_ind = np.argwhere(np.isnan(K_est_f))
len(nan_ind) #10180/10359

imputer = KNNImputer(n_neighbors=5)
Z_est_c = imputer.fit_transform(K_est_c)
Z_est_f = imputer.fit_transform(K_est_f)
Z_est_GASM = imputer.fit_transform(Z_est_GASM)


# Reformulated as Constrained optimization problem - ADMM
P = np.concatenate((K_est_c,K_est_f))

ep_dual = 0.001
ep_pri = 0.001
max_iterations = 5000
error_dual = ep_dual+10 # add a positive number to stop criterion
error_pri = ep_pri+10
iter = 0
beta_1 = 1
beta_2 = 1

Lambda = np.ones(K_shape)
Y = np.ones(K_shape)
W = np.ones((K_shape[0],P.shape[0]))
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    # W_old = W
    W = (Lambda @ P.transpose() + beta_1*Y @ P.transpose()) @ LA.inv(beta_2*np.eye(P.shape[0]) + beta_1*P @ P.transpose())
    W[W < 0] = 0
    Y = np.divide(2*Mask_mat*Z + beta_1*W @ P - Lambda,2*Mask_mat + beta_1*np.ones(K_shape))
    Lambda = Lambda + beta_1 * (Y - W @ P)
    error_dual = LA.norm(beta_1 *(W-W_old))
    error_pri = LA.norm(Y - W @ P)
    iter = iter + 1
Z_est = W @ P
LA.norm(Z_est-K_true)/LA.norm(K_true)

# nuclear norm on Y
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    W_old = W
    W = (Lambda @ P.transpose() - beta_1*Y @ P.transpose()) @ LA.inv(-beta_1*P @ P.transpose())
    Y = np.divide(2*Mask_mat*Z + beta_1*W @ P - Lambda,2*Mask_mat + (beta_1+beta_2)*np.ones(K_shape))
    Lambda = Lambda + beta_1 * (Y - W @ P)
    error_dual = LA.norm(beta_1 *(W-W_old))
    error_pri = LA.norm(Y - W @ P)
    iter = iter + 1
Z_est = W @ P

# formation 2
iter = 0
beta = 1
epsilon = 1
Lambda = np.ones(K_shape)
W = np.ones((K_shape[0],P.shape[0]))
S = np.ones(P.shape)

while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    R_old = W @ S
    W = np.divide((Mask_mat * Lambda + beta * Mask_mat * Z),(np.ones(K_shape) + beta * Mask_mat)) @ LA.pinv(S + LA.pinv(S.transpose()))
    S = LA.pinv(W + LA.pinv(W.transpose())) @ np.divide((Mask_mat * Lambda + beta * Mask_mat * Z),(np.ones(K_shape) + beta * Mask_mat))
    S = P + epsilon/LA.norm(S-P) * (S - P)
    R =  W @ S
    # W[W < 0] = 0
    Lambda = Lambda + beta * Mask_mat * (Z - R)
    # Lambda = Lambda + beta * (Z - R)
    error_dual = LA.norm(beta *(R-R_old))
    error_pri = LA.norm(Z - R)
    iter = iter + 1
Z_est = R

# ADMM update fix
ep_dual = 0.001
ep_pri = 0.001
max_iterations = 30000
error_dual = ep_dual+10 # add a positive number to stop criterion
error_pri = ep_pri+10
iter = 0
beta = 1
# epsilon = 0.1
epsilon = 0.1
Lambda_X = np.ones(K_shape)
Y = np.ones(K_shape)
W = np.ones((K_shape[0],P.shape[0]))
S = np.ones(P.shape)
Lambda_W = np.ones(W.shape)
Lambda_H = np.ones(S.shape)
W_plus = np.zeros(W.shape)
rela_err_list = [0]*max_iterations

while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    R_old = W @ S
    W = ((Lambda_X @ S.transpose() - Lambda_W)/beta + Y @ S.transpose() + W_plus) @ LA.inv(S @ S.transpose() + np.eye(S.shape[0]))
    S = LA.inv(W.transpose() @ W + np.eye(W.shape[1])) @ ((W.transpose() @ Lambda_X - Lambda_H)/beta + W.transpose() @ Y + P)
    S = P + epsilon / LA.norm(S - P) * (S - P) # project S to constraint
    Y = np.divide(2* Mask_mat * Z - Lambda_X + beta * W @ S, 2 * Mask_mat + beta * np.ones(K_shape))
    #Y = Y + Lambda_X/beta
    #Y[Y < 0] = 0
    W_plus = W + Lambda_W/beta
    W_plus[W_plus < 0] = 0
    R = W_plus @ S
    Lambda_X = Lambda_X + beta * (Y - W @ S)
    Lambda_W = Lambda_W + beta * (W - W_plus)
    Lambda_H = Lambda_H + beta * (S - P)
    error_dual = LA.norm(beta *(R-R_old))
    error_pri = LA.norm(Z - R)
    rela_err_list[iter] = LA.norm(R - K_true) / LA.norm(K_true)
    iter = iter + 1
Z_est = W_plus @ S

LA.norm(Z_est-K_true)/LA.norm(K_true)

# Optimizing with Hadamard product
ep_dual = 0.001
ep_pri = 0.001
max_iterations = 10000
error_dual = ep_dual+10 # add a positive number to stop criterion
error_pri = ep_pri+10
iter = 0
beta = 0.1

rela_err_list = [0]*max_iterations

W = np.ones(Z.shape)
W_plus = np.ones(W.shape)
Lambda_Z = np.ones(Z.shape)
Lambda_W = 0.5*np.ones(W.shape)
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W * K_est_c + (J - W) * K_est_f
    Z_hat = np.divide(Mask_mat*Z - Lambda_Z + beta * W * K_est_c + beta * (J - W) * K_est_f, Mask_mat + beta * J)
    W = np.divide((K_est_c - K_est_f) * (Lambda_Z + beta * (Z_hat - K_est_f))-Lambda_W + beta * W_plus, beta * (K_est_f - K_est_c) * (K_est_f - K_est_c) + beta * J)
    W_plus = W + Lambda_W / beta
    W_plus[W_plus < 0] = 0
    W_plus = np.divide(W_plus,np.max(W_plus))
    Lambda_Z = Lambda_Z + beta * (Z_hat - W * K_est_c - (J - W) * K_est_f)
    Lambda_W = Lambda_W + beta * (W - W_plus)
    Z_est = W * K_est_c + (J - W) * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Mask_mat*(Z - Z_est))
    rela_err_list[iter] = LA.norm(Z_est-K_true)/LA.norm(K_true)
    iter = iter + 1

print(LA.norm(Z_est-K_true)/LA.norm(K_true))

# Optimizing with Hadamard product: binary

W_S = np.ones(W.shape)
Lambda_S = np.ones(W_S.shape)
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W * K_est_c + (J - W) * K_est_f
    Z_hat = np.divide(Mask_mat*Z - Lambda_Z + beta * W * K_est_c + beta * (J - W) * K_est_f, Mask_mat + beta * J)
    W = np.divide((K_est_c - K_est_f) * (Lambda_Z + beta * (Z_hat - K_est_f))-Lambda_W -Lambda_S + beta * (W_plus + W_S), beta * (K_est_f - K_est_c) * (K_est_f - K_est_c) + 2* beta * J)
    W_plus = W + Lambda_W / beta
    W_plus[W_plus < 0] = 0
    W_plus = np.divide(W_plus,np.max(W_plus))
    W_S = W + Lambda_S / beta
    W_S = 0.5 * J + np.sqrt(np.prod(W_S.shape)/4) * np.divide(W - 0.5 * J,LA.norm(W - 0.5 * J))
    Lambda_Z = Lambda_Z + beta * (Z_hat - W * K_est_c - (J - W) * K_est_f)
    Lambda_W = Lambda_W + beta * (W - W_plus)
    Lambda_S = Lambda_S + beta * (W - W_S)
    Z_est = W * K_est_c + (J - W) * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Mask_mat*(Z - Z_est))
    iter = iter + 1

LA.norm(Z_est-K_true)/LA.norm(K_true)

# Optimizing with Hadamard product: min rank constraint, but no constraint on W
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 1000;iter = 0
rela_err_list = [0]*max_iterations
error_dual = ep_dual+10;error_pri = ep_pri+10
Z_hat = np.ones(Z.shape)
W_1 = 0.5*np.ones(Z.shape);W_2 = 0.5*np.ones(Z.shape)
Lambda_Z = np.ones(Z.shape);Lambda_1 = np.ones(Z.shape);Lambda_2 = np.ones(Z.shape)
gamma = 1;beta = 2
dim = 10
np.random.seed(1)
L = 100*np.random.rand(Z.shape[0],dim);R = 100*np.random.rand(dim,Z.shape[1])
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    L = (beta*(Z_hat + W_1 * K_est_c + W_2 * K_est_f) - Lambda_1 - Lambda_2) @ R.T @ LA.inv(np.eye(dim) + 2*beta* R @ R.T)
    R = LA.inv(np.eye(dim) + 2*beta* L.T @ L) @ L.T @ (beta*(Z_hat + W_1 * K_est_c + W_2 * K_est_f) - Lambda_1 - Lambda_2)
    Z_hat = np.divide(beta* Mask_mat * Z + beta * L @ R + Lambda_1 - Mask_mat * Lambda_Z, beta * (Mask_mat + J))
    W_1 = np.divide((Lambda_2 + beta * L @ R - beta * W_2 * K_est_f) * K_est_c, gamma * J + beta * K_est_c * K_est_c)
    W_2 = np.divide((Lambda_2 + beta * L @ R - beta * W_1 * K_est_c) * K_est_f, gamma * J + beta * K_est_f * K_est_f)
    Lambda_Z = Lambda_Z + beta * (Mask_mat * (Z_hat - Z))
    Lambda_1 = Lambda_1 + beta * (L @ R - Z_hat)
    Lambda_2 = Lambda_2 + beta * (L @ R - W_1 * K_est_c - W_2 * K_est_f)
    Z_est = W_1 * K_est_c + W_2 * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Mask_mat * (Z - Z_est))
    rela_err_list[iter] = LA.norm(Z_est-K_true)/LA.norm(K_true)
    iter = iter + 1

# Optimizing with Hadamard product: min rank constraint, W_1 and W_2 nonnegative
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 10000;iter = 0
error_dual = ep_dual+10;error_pri = ep_pri+10
Z_hat = np.ones(Z.shape)
W_1 = np.ones(Z.shape);W_2 = np.ones(Z.shape)
W_1p = np.ones(Z.shape);W_2p = np.ones(Z.shape)
Lambda_Z = np.ones(Z.shape);Lambda_1 = np.ones(Z.shape);Lambda_2 = np.ones(Z.shape);Lambda_1p = np.ones(Z.shape);Lambda_2p = np.ones(Z.shape)
gamma = 0.01;beta = 1
dim = 30
np.random.seed(1)
L = 10*np.random.rand(Z.shape[0],dim);R = 10*np.random.rand(dim,Z.shape[1])
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    L = (beta*(Z_hat + W_1 * K_est_c + W_2 * K_est_f) - Lambda_1 - Lambda_2) @ R.T @ LA.inv(np.eye(dim) + 2*beta* R @ R.T)
    R = LA.inv(np.eye(dim) + 2*beta* L.T @ L) @ L.T @ (beta*(Z_hat + W_1 * K_est_c + W_2 * K_est_f) - Lambda_1 - Lambda_2)
    Z_hat = np.divide(beta* Mask_mat * Z + beta * L @ R + Lambda_1 - Mask_mat * Lambda_Z, beta * (Mask_mat + J))
    W_1 = np.divide((Lambda_2 + beta * L @ R - beta * W_2 * K_est_f) * K_est_c - Lambda_1p + beta * W_1p, (gamma + beta) * J + beta * K_est_c * K_est_c)
    W_2 = np.divide((Lambda_2 + beta * L @ R - beta * W_1 * K_est_c) * K_est_f - Lambda_2p + beta * W_2p, (gamma + beta) * J + beta * K_est_f * K_est_f)
    W_1p = W_1p + Lambda_1p / beta
    W_2p = W_1p + Lambda_2p / beta
    W_1p[W_1p < 0] = 0
    W_2p[W_2p < 0] = 0
    Lambda_Z = Lambda_Z + beta * (Mask_mat * (Z_hat - Z))
    Lambda_1 = Lambda_1 + beta * (L @ R - Z_hat)
    Lambda_2 = Lambda_2 + beta * (L @ R - W_1 * K_est_c - W_2 * K_est_f)
    Lambda_1p = Lambda_1p + beta* (W_1 - W_1p)
    Lambda_2p = Lambda_2p + beta * (W_2 - W_2p)
    Z_est = W_1 * K_est_c + W_2 * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Mask_mat * (Z - Z_est))
    iter = iter + 1
Z_est = W_1p * K_est_c + W_2p * K_est_f

# Optimizing with Hadamard product: without min rank constraint, W_1 and W_2 nonnegative
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 10000;iter = 0
error_dual = ep_dual+10;error_pri = ep_pri+10
Z_hat = np.ones(Z.shape)
W_1 = np.ones(Z.shape);W_2 = np.ones(Z.shape)
W_1p = np.ones(Z.shape);W_2p = np.ones(Z.shape)
Lambda_Z = np.ones(Z.shape);Lambda_1p = np.ones(Z.shape);Lambda_2p = np.ones(Z.shape)
beta = 1
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    Z_hat = np.divide(Mask_mat * Z - Lambda_Z + beta * (W_1 * K_est_c + W_2 * K_est_f), Mask_mat + beta * J)
    W_1 = np.divide(Lambda_Z * K_est_c + beta * (Z_hat - W_2 * K_est_f) * K_est_c - Lambda_1p + beta * W_1p, beta * (J + K_est_c * K_est_c))
    W_2 = np.divide(Lambda_Z * K_est_f + beta * (Z_hat - W_1 * K_est_c) * K_est_f - Lambda_1p + beta * W_2p, beta * (J + K_est_f * K_est_f))
    W_1p = W_1p + Lambda_1p / beta
    W_2p = W_1p + Lambda_2p / beta
    W_1p[W_1p < 0] = 0
    W_2p[W_2p < 0] = 0
    Lambda_Z = Lambda_Z + beta * (Z_hat - W_1 * K_est_c - W_2 * K_est_f)
    Lambda_1p = Lambda_1p + beta* (W_1 - W_1p)
    Lambda_2p = Lambda_2p + beta * (W_2 - W_2p)
    Z_est = W_1 * K_est_c + W_2 * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Mask_mat * (Z - Z_est))
    iter = iter + 1
Z_est = W_1 * K_est_c + W_2 * K_est_f
LA.norm(Z_est-K_true)/LA.norm(K_true)

# Coordinate descent: Optimizing with Hadamard product: min rank constraint, but no constraint on W
ep = 0.001;max_iterations = 1000;iter = 0;error = ep+10
rela_err_list = [0]*max_iterations
W_1 = np.ones(Z.shape);W_2 = np.ones(Z.shape)
alpha = 10**(-10)
Lambda_1 = 2;Lambda_2 = 2
dim = 20
L = 10*np.random.rand(Z.shape[0],dim);R = 10*np.random.rand(dim,Z.shape[1])
while (error > ep) and iter < max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    L = L - alpha * (L - Lambda_1 * Mask_mat * (L @ R - Z) @ R.T - Lambda_2 * (L @ R - W_1 * K_est_c - W_2 * K_est_f) @ R.T)
    R = R - alpha * (R - Lambda_1 * L.T @ (Mask_mat * (L @ R - Z)) - Lambda_2 * L.T @ (L @ R - W_1 * K_est_c - W_2 * K_est_f))
    W_1 = W_1 - alpha * (Lambda_2 * (L @ R - W_1 * K_est_c - W_2 * K_est_f) * K_est_c)
    W_2 = W_2 - alpha * (Lambda_2 * (L @ R - W_1 * K_est_c - W_2 * K_est_f) * K_est_f)
    error = LA.norm(W_1 * K_est_c + W_2 * K_est_f - Z_old)
    rela_err_list[iter] = error
    iter = iter + 1
Z_est = W_1 * K_est_c + W_2 * K_est_f
LA.norm(Z_est-K_true)/LA.norm(K_true)

# without min rank constraint, W is binary (alternative)
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 1000;iter = 0
rela_err_list = [0]*max_iterations
error_dual = ep_dual+10;error_pri = ep_pri+10
Z_hat = np.ones(Z.shape)
W_1 = 0.5*np.ones(Z.shape);W_2 = 0.5*np.ones(Z.shape)
#W_1 = np.random.rand(Z.shape[0],Z.shape[1]);W_2 = np.random.rand(Z.shape[0],Z.shape[1]);
Z_init = W_1 * K_est_c + W_2 * K_est_f
Lambda_Z = np.ones(Z.shape);Lambda_1 = np.ones(Z.shape);Lambda_2 = np.ones(Z.shape)
beta = 1
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    Z_hat = np.divide(Mask_mat*Z-Lambda_Z+beta*(W_1*K_est_c+W_2*K_est_f), Mask_mat+beta*J)
    W_1 = np.divide(Lambda_Z*K_est_c+beta*(Z_hat-W_2*K_est_f)*K_est_c-Lambda_1*W_2-Lambda_2-beta*(W_2-J),beta*(K_est_c*K_est_c+W_2*W_2+J))
    W_2 = np.divide(Lambda_Z*K_est_f+beta*(Z_hat-W_1*K_est_c)*K_est_f-Lambda_1*W_1-Lambda_2-beta*(W_1-J),beta*(K_est_f*K_est_f+W_1*W_1+J))
    Lambda_Z = Lambda_Z + beta * (Z_hat-W_1*K_est_c-W_2*K_est_f)
    Lambda_1 = Lambda_1 + beta * (W_1*W_2)
    Lambda_2 = Lambda_2 + beta * (W_1+W_2-J)
    Z_est = W_1 * K_est_c + W_2 * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Z_hat-W_1*K_est_c-W_2*K_est_f)
    rela_err_list[iter] = LA.norm(Z_est - K_true) / LA.norm(K_true)
    iter = iter + 1
W_1[W_1 < W_2] = 0;W_1[W_1 > W_2] = 1;W_2[W_1 < W_2] = 1;W_2[W_1 > W_2] = 0;
Z_est = W_1 * K_est_c + W_2 * K_est_f
print(LA.norm(Z_est-K_true)/LA.norm(K_true))

# without min rank constraint, W is binary (alternative) with positive constraints
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 700;iter = 0
error_dual = ep_dual+10;error_pri = ep_pri+10
Z_hat = np.ones(Z.shape)
W_1 = 0.5*np.ones(Z.shape);W_2 = 0.5*np.ones(Z.shape);W_1p = 0.5*np.ones(Z.shape);W_2p = 0.5*np.ones(Z.shape);
Lambda_Z = np.ones(Z.shape);Lambda_1 = np.ones(Z.shape);Lambda_2 = np.ones(Z.shape);Lambda_1p = np.ones(Z.shape);Lambda_2p = np.ones(Z.shape)
beta = 1
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = W_1 * K_est_c + W_2 * K_est_f
    Z_hat = np.divide(Mask_mat*Z-Lambda_Z+beta*(W_1*K_est_c+W_2*K_est_f), Mask_mat+beta*J)
    W_1 = np.divide(Lambda_Z*K_est_c+beta*(Z_hat-W_2*K_est_f)*K_est_c-Lambda_1*W_2-Lambda_2-beta*(W_2-J)-Lambda_1p+beta*W_1p,beta*(K_est_c*K_est_c+W_2*W_2+2*J))
    W_2 = np.divide(Lambda_Z*K_est_f+beta*(Z_hat-W_1*K_est_c)*K_est_f-Lambda_1*W_1-Lambda_2-beta*(W_1-J)-Lambda_2p+beta*W_2p,beta*(K_est_f*K_est_f+W_1*W_1+2*J))
    W_1p = W_1p + Lambda_1p / beta
    W_2p = W_1p + Lambda_2p / beta
    W_1p[W_1p < 0] = 0
    W_2p[W_2p < 0] = 0
    Lambda_Z = Lambda_Z + beta * (Z_hat-W_1*K_est_c-W_2*K_est_f)
    Lambda_1 = Lambda_1 + beta * (W_1*W_2)
    Lambda_2 = Lambda_2 + beta * (W_1+W_2-J)
    Lambda_1p = Lambda_1p + beta * (W_1 - W_1p)
    Lambda_2p = Lambda_2p + beta * (W_2 - W_2p)
    Z_est = W_1 * K_est_c + W_2 * K_est_f
    error_dual = LA.norm(beta * (Z_est - Z_old))
    error_pri = LA.norm(Z_hat-W_1*K_est_c-W_2*K_est_f)
    iter = iter + 1
#W_1[W_1 < 0.5] = 0;W_1[W_1 > 0.5] = 1;W_2[W_2 < 0.5] = 0;W_2[W_2 > 0.5] = 1;
W_1[W_1 < W_2] = 0;W_1[W_1 > W_2] = 1;W_2[W_1 < W_2] = 1;W_2[W_1 > W_2] = 0;
Z_est = W_1 * K_est_c + W_2 * K_est_f
print(LA.norm(Z_est-K_true)/LA.norm(K_true))

# Sample c to generate multiple estimation states
c1 = -15 # km/h
c2 = 80
sigma = 0.6 # km
zeta = 1.1 # min
Phi_list = [0] * obs_num
est_num = 5
K_est_list = [0] * est_num
iter = 0
for c in np.linspace(c1, c2, est_num):
    print(iter)
    for i, x in enumerate(position_list):
        print(i)
        x_i = x[0]
        t_j = x[1]
        U = X - x_i * J
        V = T - t_j * J - U / c
        Phi_list[i] = smothing_kernel(U,V,sigma,zeta)
    K_est = [a*b for (a,b) in zip(obsValue_list,Phi_list)]
    K_est = np.divide(sum(K_est),sum(Phi_list))
    K_est_list[iter] =  K_est
    iter += 1

## a better loop
Phi_list_dict={}
for i in range(est_num):
    Phi_list_dict[i]=[0] * obs_num
for i, x in enumerate(position_list):
    print(i)
    iter_c = 0
    for c in np.linspace(c1, c2, est_num):
        x_i = x[0]
        t_j = x[1]
        U = X - x_i * J
        V = T - t_j * J - U / c
        Phi_list_dict[iter_c][i] = smothing_kernel(U,V,sigma,zeta)
        iter_c += 1
for c in range(est_num):
    K_est = [a*b for (a,b) in zip(obsValue_list,Phi_list_dict[c])]
    K_est = np.divide(sum(K_est),sum(Phi_list_dict[c]))
    K_est_list[c] =  K_est


for K_est in K_est_list:
    print(LA.norm(K_est-K_true)/LA.norm(K_true))

## a better loop: random sample all hyperparameters
est_num = 10
v1 = v2 = 0.1
sigma_r = np.random.normal(sigma,v1,est_num)
zeta_r = np.random.normal(zeta,v2,est_num)
c_list = np.random.uniform(c1, c2, est_num)
Phi_list_dict={}
for i in range(est_num):
    Phi_list_dict[i]=[0] * obs_num
for i, x in enumerate(position_list):
    print(i)
    for j in range(est_num):
        c = c_list[j]
        x_i = x[0]
        t_j = x[1]
        U = X - x_i * J
        V = T - t_j * J - U / c
        Phi_list_dict[j][i] = smothing_kernel(U,V,sigma_r[j],zeta_r[j])
for c in range(est_num):
    K_est = [a*b for (a,b) in zip(obsValue_list,Phi_list_dict[c])]
    K_est = np.divide(sum(K_est),sum(Phi_list_dict[c]))
    K_est_list[c] =  K_est


# sampling hyperparameters: coordinate descent
ep = 0.001;max_iterations = 5000;iter = 0;error = ep+10
W_list = [0]*est_num
for i in range(est_num):
    W_list[i] = 1/est_num*np.ones(Z.shape)
Z_est_list = K_est_list
alpha = 10**(-10)
Lambda_1 = np.ones(Z.shape);Lambda_Z = np.ones(Z.shape)
rela_err_list = [0]*max_iterations
while (error > ep) and iter < max_iterations:
    print(iter)
    #Z_old = W_1 * K_est_c + W_2 * K_est_f
    Z_old = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    Z_hat = Z_hat - alpha*(Mask_mat*(Z_hat-Z) + Lambda_Z + beta * (Z_hat - Z_old))
    for i in range(est_num):
        W_i = W_list[i]
        Z_i = Z_est_list[i]
        W_i = W_i - alpha * (-Lambda_Z * Z_i - beta * (Z_hat - sum([a * b for (a, b) in zip(W_list, Z_est_list)]))*Z_i + Lambda_1 + beta *(sum(W_list) - J))
        W_list[i] = W_i
    Z_est = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    Lambda_Z = Lambda_Z + beta * (Z_hat - Z_est)
    Lambda_1 = Lambda_1 + beta * (sum(W_list)-J)
    error = LA.norm(Z_est - Z_old)
    rela_err_list[iter] = LA.norm(Z_est - K_true) / LA.norm(K_true)
    iter = iter + 1
LA.norm(Z_est-K_true)/LA.norm(K_true)

# sampling hyperparameters: ADMM
ep = 0.001;max_iterations = 5000;iter = 0;error = ep+10
W_list = [0]*est_num
for i in range(est_num):
    W_list[i] = 1/est_num*np.ones(Z.shape)
Z_est_list = K_est_list
Lambda_1 = np.ones(Z.shape);Lambda_Z = np.ones(Z.shape)
beta = 1
rela_err_list = [0]*max_iterations
while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations:
    print(iter)
    Z_old = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    Z_hat = np.divide(Mask_mat*Z-Lambda_Z+beta*Z_old, Mask_mat+beta*J)
    for i in range(est_num):
        W_i = W_list[i]
        Z_i = Z_est_list[i]
        temp = np.zeros(Z.shape)
        for j in range(est_num):
            if j == i:
                continue # skip W_i and Z_i
            temp += W_list[j]*(Z_est_list[j]*Z_i + J)
        W_i =np.divide(Lambda_Z*Z_i + beta*(Z_hat*Z_i + J - temp)-Lambda_1,beta*(Z_i*Z_i+J))
        W_list[i] = W_i
    Z_est = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    Lambda_Z = Lambda_Z + beta * (Z_hat - Z_est)
    Lambda_1 = Lambda_1 + beta * (sum(W_list) - J)
    error = LA.norm(Z_est - Z_old)
    rela_err_list[iter] = LA.norm(Z_est - K_true) / LA.norm(K_true)
    iter = iter + 1
# compare multiple W
W_binary_list = [0]*est_num
for i in range(est_num):
    W_binary_list[i] = np.zeros(Z.shape)
W_binary_index = np.argmax(W_list, 0)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        temp_ind = W_binary_index[i,j]
        W_i = W_binary_list[temp_ind]
        W_i[i,j] = 1
        W_binary_list[temp_ind] = W_i

Z_est_binary = sum([a * b for (a, b) in zip(W_binary_list, Z_est_list)])
print(LA.norm(Z_est-K_true)/LA.norm(K_true))
np.min(rela_err_list)

# sampling hyperparameters: ADMM + gradient descent
def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

while (error_dual > ep_dual or error_pri > ep_pri) and iter<max_iterations: # not done yet
    print(iter)
    Z_old = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    #Z_hat = np.divide(Mask_mat*Z-Lambda_Z+beta*Z_old, Mask_mat+beta*J)
    Z_hat = gradient_descent(gradient=lambda v: np.array(Mask_mat*(v-Z) + Lambda_Z + beta * (v - Z_old)),
                             start = np.divide(Mask_mat*Z-Lambda_Z+beta*Z_old, Mask_mat+beta*J), learn_rate=0.1)
    for i in range(est_num):
        W_i = W_list[i]
        Z_i = Z_est_list[i]
        temp = np.zeros(Z.shape)
        for j in range(est_num):
            if j == i:
                continue # skip W_i and Z_i
            temp += W_list[j]*(Z_est_list[j]*Z_i + J)
        W_i =np.divide(Lambda_Z*Z_i + beta*(Z_hat*Z_i + J - temp)-Lambda_1,beta*(Z_i*Z_i+J))
        W_list[i] = W_i
    Z_est = sum([a * b for (a, b) in zip(W_list, Z_est_list)])
    Lambda_Z = Lambda_Z + beta * (Z_hat - Z_est)
    Lambda_1 = Lambda_1 + beta * (sum(W_list) - J)
    error = LA.norm(Z_est - Z_old)
    rela_err_list[iter] = LA.norm(Z_est - K_true) / LA.norm(K_true)
    iter = iter + 1

# replace the extreme value entries and re-imputation
Z_est_fix = copy.deepcopy(Z_est)
ext_ind = np.argwhere(Z_est_fix > 110)
for ind in ext_ind:
    Z_est_fix[ind[0],ind[1]] = np.nan

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10)
Z_est_new = imputer.fit_transform(Z_est_fix)

########################### scratch
temp = np.random.rand(3)
for (x, y), element in np.ndenumerate(temp):
    print(x, y, element)

for i, x in enumerate(temp):
    print(i,x)

for (i, j), x in np.ndenumerate(U):
    print(i, j)

for i, x in enumerate(position_list):
    print(i, x)

# plot
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(K_true, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('True data')
plt.subplot(122)
plt.imshow(Z_est, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('estimation')
plt.suptitle('comparison: 10% coverage')
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(K_true, cmap='jet_r', origin='lower', aspect='auto')
axs[0, 0].set_title('True data')
axs[0, 1].imshow(Z_est_GASM, cmap='jet_r', origin='lower', aspect='auto')
axs[0, 1].set_title('GASM estimation')
axs[1, 0].imshow(K_est_c, cmap='jet_r', origin='lower', aspect='auto')
axs[1, 0].set_title('congested component')
axs[1, 1].imshow(K_est_f, cmap='jet_r', origin='lower', aspect='auto')
axs[1, 1].set_title('free flow componenet')

for ax in axs.flat:
    ax.set(xlabel='time(s)', ylabel='Road length(m)')

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(K_true, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('True data')
plt.subplot(132)
masked_array = np.ma.array (K_true, mask=1-Mask_mat)
cmap = matplotlib.cm.jet_r
cmap.set_bad('white')
plt.imshow(masked_array, cmap=cmap, origin='lower', aspect='auto', interpolation='none')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('observed')
plt.subplot(133)
plt.imshow(Z_est, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('ADMM estimation: 3400th iteration')
plt.suptitle('comparison: 3% coverage')
plt.show()

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

LA.norm(Z_init-Z_est)
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(Z_init, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('Initial estimation')
plt.subplot(122)
plt.imshow(Z_est, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('Final estimation (50000th iteration)')
plt.suptitle('comparison: 3% coverage')
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(y_map, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('True data')
plt.subplot(122)
plt.imshow(x_map1, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('estimation')
plt.suptitle('comparison: 10% coverage')
plt.show()

plt.yscale('log')
plt.plot(range(max_iterations), rela_err_list)
plt.scatter(np.argmin(rela_err_list), np.min(rela_err_list),c='r', label="min location = 3311, min err = 4.97673")
plt.legend()
plt.xlabel('time(s)')
plt.ylabel('relative estimation error')
plt.title('3% coverage, ADMM, 20000 iterations')

fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(Z_est_list[0], cmap='jet_r', origin='lower', aspect='auto')
axs[0, 0].set_title('c = -15, err = 0.28473')
axs[0, 1].imshow(Z_est_list[1], cmap='jet_r', origin='lower', aspect='auto')
axs[0, 1].set_title('c = 8.75, err = 0.28858')
axs[1, 0].imshow(Z_est_list[2], cmap='jet_r', origin='lower', aspect='auto')
axs[1, 0].set_title('c = 32.5, err = 0.2866')
axs[1, 1].imshow(Z_est_list[3], cmap='jet_r', origin='lower', aspect='auto')
axs[1, 1].set_title('c = 56.25, err = 0.28641')
axs[2, 0].imshow(Z_est_list[4], cmap='jet_r', origin='lower', aspect='auto')
axs[2, 0].set_title('c = 80, err = 0.28632')
axs[2, 1].imshow(Z_est_new, cmap='jet_r', origin='lower', aspect='auto')
axs[2, 1].set_title('ADMM+KNN outlier removal, err = 0.30595')
plt.suptitle('componenet estimation vs. ADMM estimation')


plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(-1*(abs(temp)>2), cmap='Greys_r',origin='lower', aspect='auto',interpolation='bilinear')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('W_sum absolute-value entry > 2')
plt.subplot(132)
plt.imshow(-1*(abs(temp)>5), cmap='Greys_r', origin='lower', aspect='auto', interpolation='bilinear')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('W_sum absolute-value entry > 5')
plt.subplot(133)
plt.imshow(-1*(Z_est>200), cmap='Greys_r',origin='lower', aspect='auto',interpolation='bilinear')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('Z absolute-value entry > 200')
plt.suptitle('W & Z estimation : outliers')

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(Z_est_binary, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('ADMM + Binary Convertion (10 components)')
plt.subplot(122)
plt.imshow(Z_est_new, cmap='jet_r', origin='lower', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length(km)')
plt.title('ADMM+KNN (10 components)')
plt.suptitle('comparison: 3% coverage')
plt.show()

plt.imshow(np.isnan(K_est_f), cmap='Greys_r', origin='lower', aspect='auto', interpolation='bilinear')
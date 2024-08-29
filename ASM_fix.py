# Adaptive Smoothing Method as MC: unit fixed
import numpy as np
import random
import numpy.linalg as LA
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn.impute import KNNImputer

### trajectory data
# parameters
h_act=67
max_speed=100
unit_change = 1000/(60*60)

# output data
output_Y = np.load('data/out_data_Ngsim_us101_lane2.npy')
y_test = output_Y[:, :h_act, :, :]
y_map = (y_test.copy().squeeze(axis=-1))*max_speed
K_true = y_map[0]
K_true = K_true[:,100:700]
#K_true = K_true[:,1400:2000]
#K_true = K_true[:,1500:]
K_true = unit_change*K_true
K_shape = K_true.shape

# input data
input_X = np.load('data/inp_data_Ngsim_us101_lane2_3p.npy')
x_test = input_X[:, :h_act, :, :]
x_bin = (x_test.sum(axis=3) != 0)
x_map = (x_bin * y_map)*max_speed
Z = x_map[0]
#Z = Z[:,:600]
Z = Z[:,1400:2000]
#Z = Z[:,1000:]
Mask_mat = (Z!=0).astype(int)

### detector data
#np.random.seed(5) # updated exp 1
np.random.seed(1)
Mask_col = np.random.choice([True, False], size=K_shape[0], p=[0.05, 0.95])
Mask_col = np.linspace(1,67,4)
Mask_col = Mask_col.astype(int)
Mask_mat = np.zeros(K_shape)
for i in range(K_shape[1]):
    Mask_mat[:,i] = Mask_col
Z = K_true * Mask_mat

# evenly spaced detector data
Mask_col = np.linspace(5,60,4)
Mask_col = Mask_col.astype(int)
Mask_mat = np.zeros(K_shape)
for i in Mask_col:
    Mask_mat[i, :] = np.ones((1,K_shape[1]))
Z = K_true * Mask_mat

# Data preparation
J = np.ones(K_shape)
X = np.zeros(K_shape)
for j in range(K_shape[0]):
    X[j,:] = (K_shape[0]-j)*np.ones((1,K_shape[1]))
X = 10*X
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

# both case in the same loop
c1 = -15*unit_change # km/h to m/s
c = 80*unit_change
sigma = 0.6*1000 # km to m
zeta = 1.1*60 # min to sec
#sigma = 110
#zeta = 1
Phi_list_f = [0] * obs_num
for i, x in enumerate(position_list):
    print(i)
    x_i = 10*x[0]
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
v_thr = 60*unit_change #km/h to m/s
V_thr = v_thr*np.ones(K_shape)
v_tw = 20*unit_change
W = (np.ones(K_shape)+np.tanh((V_thr-np.minimum(K_est_c, K_est_f))/v_tw))/2
Z_est_GASM = W * K_est_c + (1 - W) * K_est_f



# calculate error
print(LA.norm(K_est_f-K_true)/LA.norm(K_true))
print(LA.norm(np.divide(K_est_f-K_true,K_true)))

# figure plotting
masked_array = np.ma.array (K_true, mask=1-Mask_mat)
cmap = matplotlib.cm.jet_r
cmap.set_bad('white')
plt.imshow(masked_array, cmap=cmap, origin='upper', aspect='auto', interpolation='none')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('observed')

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(K_true, cmap='jet_r', aspect='auto')
axs[0, 0].set_title('True data')
axs[0, 1].imshow(Z_est_GASM, cmap='jet_r', aspect='auto')
axs[0, 1].set_title('GASM estimation: err = 0.1641')
axs[1, 0].imshow(K_est_c, cmap='jet_r', aspect='auto')
axs[1, 0].set_title('congested component: err = 0.1988')
axs[1, 1].imshow(K_est_f, cmap='jet_r', aspect='auto')
axs[1, 1].set_title('free flow componenet: err = 0.1267')
plt.suptitle('sigma = 110m,tau = 1s')

# ADMM: sample multiple c
sigma = 70 # formal
zeta = 0.5 #formal


est_num = 8
K_est_list = [0] * est_num
Phi_list_dict={}
for i in range(est_num):
    Phi_list_dict[i]=[0] * obs_num
for i, x in enumerate(position_list):
    print(i)
    iter_c = 0
    #for c in unit_change * np.array([-15, 80]):
    for c in unit_change*np.array([-20,90,-17.5,80,-15,70,-12.5,65]):
    #for c in unit_change * np.linspace(-15,110,5):
        x_i = 10*x[0]
        t_j = x[1]
        U = x_i * J - X
        V = t_j * J - T - U / c
        Phi_list_dict[iter_c][i] = smothing_kernel(U,V,sigma,zeta)
        iter_c += 1
for c in range(est_num):
    K_est = [a*b for (a,b) in zip(obsValue_list,Phi_list_dict[c])]
    K_est = np.divide(sum(K_est),sum(Phi_list_dict[c]))
    K_est_list[c] =  K_est

K_est_c = K_est_list[0];K_est_f = K_est_list[1]
#K_est_c = K_est_list[1];K_est_f = K_est_list[4]


# ADMM estimation
ep_dual = 0.001;ep_pri = 0.001;max_iterations = 214;iter = 0
error_dual = ep_dual+10;error_pri = ep_pri+10
W_list = [0]*est_num
for i in range(est_num):
    W_list[i] = 1/est_num*np.ones(Z.shape)
#W_list[0] = 0.7*np.ones(Z.shape)
#W_list[1] = 0.3*np.ones(Z.shape)
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

# replace the extreme value entries and re-imputation
Z_est_fix = copy.deepcopy(Z_est)
ext_ind = np.argwhere(Z_est_fix > 110*unit_change)
for ind in ext_ind:
    Z_est_fix[ind[0],ind[1]] = np.nan

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
Z_est_new = imputer.fit_transform(Z_est_fix)

# calculate error
print(LA.norm(Z_est_GASM-K_true)/LA.norm(K_true))
print(LA.norm(np.divide(K_est_c-K_true,K_true)))
print(LA.norm(np.divide(K_est_c-K_true, K_true, out=np.zeros_like(K_est_c-K_true), where=K_true!=0))) # deal with divide by zero error
print(LA.norm(np.divide(K_est_c-K_true, K_true, out=np.zeros_like(K_true), where=K_true!=0))/np.sqrt(np.sum(K_true!=0)))

# plt.figure(figsize=(8,4))
fig, ax = plt.subplots(2,2,figsize=(4,5))
#fig.tight_layout()
plt.subplots_adjust(hspace = 0.4)
plt.subplot(211)
plt.imshow(K_true, cmap='jet', aspect='auto')
plt.xticks(np.arange(0, 600, step=100))
plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.xlabel('time (s)',fontsize=8)
plt.ylabel('Road length (m)',fontsize=8)
plt.title('Ground truth',fontsize=8)
plt.colorbar()
plt.subplot(212)
masked_array = np.ma.array (K_true, mask=1-Mask_mat)
cmap = matplotlib.cm.jet
cmap.set_bad('white')
plt.imshow(masked_array, cmap=cmap, aspect='auto', interpolation='none')
plt.xticks(np.arange(0, 600, step=100))
plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.xlabel('time (s)',fontsize=8)
plt.ylabel('Road length (m)',fontsize=8)
plt.title('Observed detector Input',fontsize=8)
#plt.suptitle('True Data And Observed Data')
plt.colorbar()
plt.show()
fig.savefig('TrueAndObs.svg', format='svg', dpi=300)

fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(Z_est_list[0], cmap='jet_r', aspect='auto')
axs[0, 0].set_title('c = -30, err = 0.1680')
axs[0, 1].imshow(Z_est_list[1], cmap='jet_r', aspect='auto')
axs[0, 1].set_title('c = -15, err = 0.1827')
axs[1, 0].imshow(Z_est_list[2], cmap='jet_r', aspect='auto')
axs[1, 0].set_title('c = 65, err = 0.1399')
axs[1, 1].imshow(Z_est_list[3], cmap='jet_r', aspect='auto')
axs[1, 1].set_title('c = 80, err = 0.1411')
axs[2, 0].imshow(Z_est_list[4], cmap='jet_r', aspect='auto')
axs[2, 0].set_title('c = 95, err = 0.1423')
axs[2, 1].imshow(Z_est_GASM, cmap='jet_r', aspect='auto')
axs[2, 1].set_title('standard GASM (2 component), err = 0.1534')
plt.suptitle('componenet estimation vs. ADMM estimation')

masked_array = np.ma.array (Density_true, mask=1-Mask_mat)
cmap = matplotlib.cm.jet_r
cmap.set_bad('white')
plt.imshow(masked_array, cmap=cmap, aspect='auto', interpolation='none')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('observed')


### evaluate on density data
def velocity_to_density(v,rou_jam,c_cong,c_free):
    d = rou_jam * (1 + c_free/c_cong * np.log(1-v/c_free))**(-1)
    return d

def density_to_velocity(d,rou_jam,c_cong,c_free):
    v = c_free * (1-np.exp(c_cong/c_free*(rou_jam*d**(-1)-1)))
    return v

rou_jam = 1/7.5
# c_cong = -15*unit_change/c_free = 80*unit_change doesn't work
c_cong = -15*unit_change
c_free = 110*unit_change
Density_true = velocity_to_density(K_true,rou_jam,c_cong,c_free)

Z = Density_true * Mask_mat

counter = 0
for (i, j), element in np.ndenumerate(Density_true):
    if Mask_mat[i, j]:
        obsValue_list[counter] = element
        index_list[counter] = [i, j]
        position_list[counter] = [K_shape[0] - i, j + 1]
        counter += 1

print(LA.norm(Z_est_new-K_true)/LA.norm(K_true))
print(LA.norm(np.divide(Z_est_GASM-Density_true,Density_true)))

K_est_c = density_to_velocity(K_est_c,rou_jam,c_cong,c_free)
K_est_f = density_to_velocity(K_est_f,rou_jam,c_cong,c_free)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(Z_est, cmap='jet', aspect='auto')
axs[0, 0].set_title('ADMM estimation')
axs[0, 1].imshow(Z_est_GASM, cmap='jet', aspect='auto')
axs[0, 1].set_title('GASM estimation')
axs[1, 0].imshow(K_est_c, cmap='jet', aspect='auto')
axs[1, 0].set_title('congested component')
axs[1, 1].imshow(K_est_f, cmap='jet', aspect='auto')
axs[1, 1].set_title('free flow componenet')
fig.colorbar()

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(K_true, cmap='jet', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('True data')
plt.subplot(122)
plt.imshow(K_true_2, cmap='jet_r', aspect='auto')
plt.xlabel('time(s)')
plt.ylabel('Road length')
plt.title('estimation')
plt.suptitle('comparison: 10% coverage')
plt.show()

## multiple components test

# ADMM estimation
mul_err_list = [0]*4
mul_err_list2 = [0]*4
mul_min_loc = [0]*4
iter_mul = 0
for j in [2,4,6,8]:
    est_num = j
    ep_dual = 0.001;ep_pri = 0.001;max_iterations = 1000;iter = 0
    error_dual = ep_dual+10;error_pri = ep_pri+10
    W_list = [0]*est_num
    for i in range(est_num):
        W_list[i] = 1/est_num*np.ones(Z.shape)
    Z_est_list = K_est_list[:j]
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
    mul_min_loc[iter_mul] = np.argmin(rela_err_list)
    mul_err_list[iter_mul] = np.min(rela_err_list)
    mul_err_list2[iter_mul] = (LA.norm(Z_est-K_true)/LA.norm(K_true))
    iter_mul += 1


# formal plots
fig, ax = plt.subplots(2,2,figsize=(5,2.5))
#fig.tight_layout()
plt.subplots_adjust(wspace = 0.1)
plt.subplot(121)
plt.imshow(K_true, cmap='jet_r', aspect='auto',vmin=0,vmax=110*unit_change)
plt.xticks(np.arange(0, 600, step=100),rotation=25)
#plt.xticks(np.arange(0, 1260, step=200),rotation=25)
plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Road length (m)',fontsize=8)
plt.title('Ground truth',fontsize=8)
plt.subplot(122)
masked_array = np.ma.array (K_true, mask=1-Mask_mat)
cmap = matplotlib.cm.jet_r
cmap.set_bad('white')
plt.imshow(masked_array, cmap=cmap, aspect='auto', interpolation='none',vmin=0,vmax=110*unit_change)
#plt.xticks(np.arange(0, 1260, step=200),rotation=25)
plt.xticks(np.arange(0, 600, step=100),rotation=25)
#plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.yticks([])
plt.xlabel('Time (s)',fontsize=8)
#plt.ylabel('Road length (m)',fontsize=8)
plt.title('Observed detector Input',fontsize=8)
#plt.suptitle('True Data And Observed Data')
plt.colorbar(label = 'velocity (m/s)')
plt.show()
fig.savefig('TrueAndObs_horizon.svg', format='svg', dpi=300)


fig, ax = plt.subplots(2,2,figsize=(5,2.5))
#fig.tight_layout()
plt.subplots_adjust(wspace = 0.1)
plt.subplot(121)
plt.imshow(Z_est, cmap='jet_r', aspect='auto',vmin=0,vmax=110*unit_change)
plt.xticks(np.arange(0, 600, step=100),rotation=25)
plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Road length (m)',fontsize=8)
plt.title('ADMM estimation',fontsize=8)
plt.subplot(122)
plt.imshow(Z_est_GASM, cmap='jet_r', aspect='auto',vmin=0,vmax=110*unit_change)
plt.xticks(np.arange(0, 600, step=100),rotation=25)
#plt.yticks(np.arange(0, 67, step=5),np.arange(670, 0, step=-50))
plt.yticks([])
plt.xlabel('Time (s)',fontsize=8)
#plt.ylabel('Road length (m)',fontsize=8)
plt.title('ASM estimation',fontsize=8)
#plt.suptitle('True Data And Observed Data')
plt.colorbar(label = 'velocity (m/s)')
plt.show()
fig.savefig('ADMM_and_ASM_horizon.svg', format='svg', dpi=300)


fig = plt.figure(figsize=(4,4))
plt.plot([2,4,6,8], mul_err_list,'b-o')
#plt.axhline(0.12417, linestyle='--')
plt.plot(2, 0.12417,'g*',markersize=10)
plt.xlim(1,9)
plt.ylim(0.1185,0.125)
plt.legend(['ADMM','ASM'])
plt.xlabel('number of priori estimates',fontsize=8)
plt.ylabel('relative error',fontsize=8)
plt.grid()
plt.show()
fig.savefig('Multiple_Comp.svg', format='svg', dpi=300)

fig, ax = plt.subplots(5,5,figsize=(4,4))
#fig.tight_layout()
plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
plt.subplot(221)
plt.plot(range(K_shape[0]), np.flip(Z_est[:,100]))
plt.plot(range(K_shape[0]), np.flip(Z_est_GASM[:,100]))
plt.plot(range(K_shape[0]), np.flip(K_true[:,100]))
plt.xticks([])
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 100',fontsize=8)
plt.subplot(222)
plt.plot(range(K_shape[0]), np.flip(Z_est[:,300]))
plt.plot(range(K_shape[0]), np.flip(Z_est_GASM[:,300]))
plt.plot(range(K_shape[0]), np.flip(K_true[:,300]))
plt.xticks([])
plt.title('Timestamp = 300',fontsize=8)
plt.subplot(223)
line1, = plt.plot(range(K_shape[0]), np.flip(Z_est[:,500]))
line2, = plt.plot(range(K_shape[0]), np.flip(Z_est_GASM[:,500]))
line3, = plt.plot(range(K_shape[0]), np.flip(K_true[:,500]))
plt.xlabel('Location',fontsize=8)
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 500',fontsize=8)
ax = plt.subplot(224)
plt.legend([line1, line2, line3], ['ADMM','ASM','Truth'],loc='center')
plt.xticks([])
plt.yticks([])
plt.title('Line description',fontsize=8)
#plt.suptitle('Case 2 velocity estimation snapshot')
plt.show()
fig.savefig('Case1_snap.svg', format='svg', dpi=300)


fig, ax = plt.subplots(5,5,figsize=(4,4))
#fig.tight_layout()
plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
plt.subplot(221)
plt.semilogy(range(K_shape[0]), np.flip(Z_est[:,50]))
plt.semilogy(range(K_shape[0]), np.flip(Z_est_GASM[:,50]))
plt.semilogy(range(K_shape[0]), np.flip(K_true[:,50]))
plt.xticks([])
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 100',fontsize=8)
plt.subplot(222)
plt.semilogy(range(K_shape[0]), np.flip(Z_est[:,250]))
plt.semilogy(range(K_shape[0]), np.flip(Z_est_GASM[:,250]))
plt.semilogy(range(K_shape[0]), np.flip(K_true[:,250]))
plt.xticks([])
plt.title('Timestamp = 300',fontsize=8)
plt.subplot(223)
line1, = plt.semilogy(range(K_shape[0]), np.flip(Z_est[:,550]))
line2, = plt.semilogy(range(K_shape[0]), np.flip(Z_est_GASM[:,550]))
line3, = plt.semilogy(range(K_shape[0]), np.flip(K_true[:,550]))
plt.xlabel('Location',fontsize=8)
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 500',fontsize=8)
ax = plt.subplot(224)
plt.legend([line1, line2, line3], ['ADMM','ASM','Truth'],loc='center')
plt.xticks([])
plt.yticks([])
plt.title('Line description',fontsize=8)
#plt.suptitle('Case 2 velocity estimation snapshot')
plt.show()
fig.savefig('Case1_snap.svg', format='svg', dpi=300)

temp_all = np.concatenate((temp_ADMM.reshape(-1,1),temp_ASM.reshape(-1,1),K_true.reshape(-1,1)),axis=-1)
temp_all_sorted = temp_all[temp_all[:, 2].argsort()]

plt.plot(temp_all_sorted[::100,2],temp_all_sorted[::100,0])
plt.plot(temp_all_sorted[::100,2],temp_all_sorted[::100,1])
plt.xlim([10,16])
plt.show()

plt.boxplot(temp_all_sorted[:,2],temp_all_sorted[:,0])
plt.boxplot(temp_all_sorted[:,2],temp_all_sorted[:,1])
plt.show()

np.save('K_est_list1.npy', K_est_list)
np.save('Phi_list_dict1.npy', Phi_list_dict)

Z_est_error = np.abs(Z_est-K_true)
Z_GASM_error = np.abs(Z_est_GASM-K_true)

fig, ax = plt.subplots(5,5,figsize=(4,4))
#fig.tight_layout()
plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
plt.subplot(221)
plt.plot(range(K_shape[0]), np.flip(Z_est_error[:,50]))
plt.plot(range(K_shape[0]), np.flip(Z_GASM_error[:,50]))
plt.xticks([])
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 100',fontsize=8)
plt.subplot(222)
plt.plot(range(K_shape[0]), np.flip(Z_est_error[:,250]))
plt.plot(range(K_shape[0]), np.flip(Z_GASM_error[:,250]))
plt.xticks([])
plt.title('Timestamp = 300',fontsize=8)
plt.subplot(223)
line1, = plt.plot(range(K_shape[0]), np.flip(Z_est_error[:,550]))
line2, = plt.plot(range(K_shape[0]), np.flip(Z_GASM_error[:,550]))
plt.xlabel('Location',fontsize=8)
plt.ylabel('Velocity (m/s)',fontsize=8)
plt.title('Timestamp = 500',fontsize=8)
ax = plt.subplot(224)
plt.legend([line1, line2], ['ADMM','ASM'],loc='center')
plt.xticks([])
plt.yticks([])
plt.title('Line description',fontsize=8)
#plt.suptitle('Case 2 velocity estimation snapshot')
plt.show()
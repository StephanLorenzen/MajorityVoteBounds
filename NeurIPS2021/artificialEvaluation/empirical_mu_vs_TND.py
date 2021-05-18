import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import ticker
from math import log

title = 'empirical_'

npoints = 1000

# Define grid for empirical gibbs and tandem loss
emp_gibbs_loss = np.linspace(0., 0.495, npoints)
emp_tandem_loss = np.linspace(0., 0.495, npoints)
axis_gibbs_loss, axis_tandem_loss = np.meshgrid(emp_gibbs_loss, emp_tandem_loss) # axis

# Define the eps term
KL = 1
lg = log(1/0.05)
n1 = 10000.
eps1 = (KL+lg)/n1 # for gibbs
n2 = n1
eps2 = (2.*KL+lg)/n2 # for tnd

# helper functions for kl-1
def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([ q*log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])

def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])
    

def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    #print(q, right_hand_side)
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0-1e-9) <= 0.0:
        return 1.0-1e-9
    else:
        return optimize.brentq(f, q, 1.0-1e-11)


def solve_kl_inf(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x < q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1e-9) <= 0.0:
        return 1e-9
    else:
        return optimize.brentq(f, 1e-11, q)

# calculate kl-1
up_tandem_loss = np.array([solve_kl_sup(i, eps2) for i in emp_tandem_loss])
low_gibbs_loss = np.array([solve_kl_inf(i, eps1) for i in emp_gibbs_loss])
grid_gibbs_loss, grid_tandem_loss = np.meshgrid(low_gibbs_loss, up_tandem_loss)

mu_star = (0.5 * grid_gibbs_loss - grid_tandem_loss) / (0.5 - grid_gibbs_loss)
secondOrderBound = 4 * grid_tandem_loss
c2Bound = (grid_tandem_loss - 2 * mu_star * grid_gibbs_loss + mu_star ** 2) / (0.5 - mu_star) ** 2

for i in range(npoints):
    for j in range(npoints):
        # if Tandem>Gibbs
        if (grid_tandem_loss[i, j]>grid_gibbs_loss[i, j]):
            mu_star[i,j]=np.nan
            secondOrderBound[i,j]=np.nan
            c2Bound[i,j]=np.nan

        # if Gibbs^^>Tandem
        if (grid_gibbs_loss[i, j]**2>grid_tandem_loss[i, j]):
            mu_star[i,j]=np.nan
            secondOrderBound[i,j]=np.nan
            c2Bound[i,j]=np.nan

gibbs_test = 0.2
tnd_test = 0.08
gibbs_test = solve_kl_inf(gibbs_test, eps1)
tnd_test = solve_kl_sup(tnd_test, eps2)
mu_test = (0.5 * gibbs_test - tnd_test) / (0.5 - gibbs_test)
print(mu_test)

fig = plt.figure()
plt.contourf(axis_gibbs_loss, axis_tandem_loss, mu_star, levels = 30, cmap='rainbow')
#plt.plot(emp_gibbs_loss, 0.5 * emp_gibbs_loss - 0.006, c='black', label='our data') #rfc
plt.plot(emp_gibbs_loss, 1.2355 * emp_gibbs_loss**2 - 0.01 * emp_gibbs_loss + 0.026, c='black', label='our data')  #abc
plt.plot(emp_gibbs_loss, 0.5 * emp_gibbs_loss, 'b-', label='tnd=0.5gibbs')

plt.colorbar()
plt.title('muStar, n1='+str(n1))
plt.xlabel('Empirical Gibbs Loss')
plt.ylabel('Empirical Tandem Loss')
plt.legend()
plt.savefig(title+'muStar.png')

fig = plt.figure()
plt.contourf(axis_gibbs_loss, axis_tandem_loss, secondOrderBound, levels = 30, cmap = "rainbow")
plt.colorbar()
plt.title('TndBound, n1='+str(n1))
plt.xlabel('Empirical Gibbs Loss')
plt.ylabel('Empirical Tandem Loss')
plt.savefig(title+'TNDBound.png')

fig = plt.figure()
plt.contourf(axis_gibbs_loss, axis_tandem_loss, c2Bound, levels = 30, cmap = "rainbow")
plt.colorbar()
plt.title('mu-Bound, n1='+str(n1))
plt.xlabel('Empirical Gibbs Loss')
plt.ylabel('Empirical Tandem Loss')
plt.savefig(title+'muBound.png')


fig = plt.figure()
#levels = np.linspace(0,1,11)
plt.contourf(axis_gibbs_loss, axis_tandem_loss, c2Bound / secondOrderBound, levels = 30, cmap = "rainbow")
plt.plot(emp_gibbs_loss, 1.2355 * emp_gibbs_loss**2 - 0.01 * emp_gibbs_loss + 0.026, c='black', label='our data')
plt.plot(emp_gibbs_loss, 0.5 * emp_gibbs_loss, 'b-', label='tnd=0.5gibbs')
#plt.plot(expected_gibbs_loss, 0.35 * expected_gibbs_loss, 'b-', label='tnd=0.35gibbs')
plt.colorbar()
plt.title('mu-Bound/TNDBound, n1='+str(n1))
plt.xlabel('Empirical Gibbs Loss')
plt.ylabel('Empirical Tandem Loss')
plt.legend()
plt.savefig(title+'muBound_vs_TNDBound.png')
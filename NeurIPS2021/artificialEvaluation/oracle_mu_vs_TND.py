import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

title = 'oracle_'

npoints = 1000

expected_gibbs_loss = np.linspace(0.005, 0.495, npoints)
expected_tandem_loss = np.linspace(0.005, 0.495, npoints)

grid_gibbs_loss, grid_tandem_loss = np.meshgrid(expected_gibbs_loss, expected_tandem_loss)

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


# The following is not strictly essential, but it will eliminate
# a warning.  Comment it out to see the warning.
#mu_star = ma.masked_where(z <= 0, z)

fig = plt.figure()
plt.contourf(grid_gibbs_loss, grid_tandem_loss, mu_star, levels = 30, cmap='rainbow')
plt.plot(expected_gibbs_loss, 0.5 * expected_gibbs_loss, c='black', label='$\mu^*=0$')
plt.colorbar()
plt.title('muStar')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.legend()
plt.savefig(title+'muStar.png')

fig = plt.figure()
plt.contourf(grid_gibbs_loss, grid_tandem_loss, secondOrderBound, levels = 30, cmap = "rainbow")
plt.colorbar()
plt.title('TndBound')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.savefig(title+'TNDBound.png')

fig = plt.figure()
plt.contourf(grid_gibbs_loss, grid_tandem_loss, c2Bound, levels = 30, cmap = "rainbow")
plt.colorbar()
plt.title('mu-Bound')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.savefig(title+'muBound.png')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

fig = plt.figure()
#levels = np.linspace(0,1,11)
plt.contourf(grid_gibbs_loss, grid_tandem_loss, c2Bound / secondOrderBound, levels = 30, cmap = "rainbow")
plt.plot(expected_gibbs_loss, 0.5 * expected_gibbs_loss, c='black', label='$\mu$')
#plt.plot(expected_gibbs_loss, 0.35 * expected_gibbs_loss, 'b-', label='tnd=0.35gibbs')
plt.colorbar()
plt.title('C$\mu$T/TND')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.legend()
plt.savefig(title+'CmuT_vs_TND.png')




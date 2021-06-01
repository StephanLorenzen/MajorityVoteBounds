import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib

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
    'font.size': 30,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

fig = plt.figure(figsize=(12,9.6))
#levels = np.linspace(0,1,11)
plt.contourf(grid_gibbs_loss, grid_tandem_loss, c2Bound / secondOrderBound, levels = 30, cmap = "rainbow")
plt.plot(expected_gibbs_loss, 0.5 * expected_gibbs_loss, c='black', linewidth=4, label=r"$\mathbb{E}_{\rho^2}[L(h,h')]=0.5\mathbb{E}_{\rho}[L(h)]$")
#plt.plot(expected_gibbs_loss, 0.35 * expected_gibbs_loss, 'b-', label='tnd=0.35gibbs')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=40)
cbar.ax.locator_params(nbins=8)
plt.title('RHS-Thm7 / RHS-Thm3', fontsize=45)
plt.xlabel(r"$\mathbb{E}_{\rho}[L(h)]$", fontsize=45)
plt.xticks([0.1, 0.2, 0.3, 0.4], fontsize=42)
plt.ylabel(r"$\mathbb{E}_{\rho^2}[L(h,h')]$", fontsize=45)
plt.yticks([0.1, 0.2, 0.3, 0.4], fontsize=42)
plt.legend(loc='upper left', framealpha=0., handlelength=1, fontsize=40)
plt.tight_layout()
plt.savefig(title+'CmuT_vs_TND.png')




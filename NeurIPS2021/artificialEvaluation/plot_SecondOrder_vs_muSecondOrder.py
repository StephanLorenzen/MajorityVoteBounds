import numpy as np
import matplotlib.pyplot as plt

npoints = 1000

expected_gibbs_loss = np.linspace(0, 0.49, npoints)
expected_tandem_loss = np.linspace(0, 0.49, npoints)

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



plt.contourf(grid_gibbs_loss, grid_tandem_loss, mu_star)
plt.colorbar()
plt.title('muStar')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.show()


plt.contourf(grid_gibbs_loss, grid_tandem_loss, secondOrderBound)
plt.colorbar()
plt.title('secondOrderBound')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.show()


plt.contourf(grid_gibbs_loss, grid_tandem_loss, c2Bound)
plt.colorbar()
plt.title('c2Bound')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.show()


levels = np.linspace(0,1,11)
plt.contourf(grid_gibbs_loss, grid_tandem_loss, c2Bound / secondOrderBound, levels)
plt.plot(expected_gibbs_loss, 0.5 * expected_gibbs_loss, c='black', label='C2/SO=1')
plt.colorbar()
plt.title('c2Bound/secondOrderBound')
plt.xlabel('Expected Gibbs Loss')
plt.ylabel('Expected Tandem Loss')
plt.legend()
plt.show()




# Make a color map perceptually uniform
# Inspired from https://bids.github.io/colormap/
# Depends: numpy, matplotlib, scipy
# Usage:
#   python unify_cm.py hot.csv
# Then you may run it multiple times to get a better result
#   python unify_cm.py tmp_cmap.csv
#   python unify_cm.py tmp_cmap.csv
#   python unify_cm.py tmp_cmap.csv
#   python unify_cm.py tmp_cmap.csv

import sys
import time
import csv
import numpy as np
from colorspacious import cspace_converter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

# Function to represent colors in perceptually uniform space
sRGB_to_uniform = cspace_converter('sRGB1', 'CAM02-UCS')

if len(sys.argv) > 1:
    c_name = sys.argv[1]
else:
    print("Usage: ", __FILE__, " _COLORMAP.csv_ [N_OUTPUT_POINT]")
    exit()

# Load cmap
with open(c_name, 'rb') as f:
    reader = csv.reader(f)
    cmap = np.array(list(reader)).astype(np.float)

# number of output colors
if len(sys.argv) > 2:
    n_output = int(sys.argv[2])
else:
    n_output = len(cmap);

# Initial guess of transform for uniformize
n = len(cmap)
s_t_init = np.linspace(1.0/n, 1 - 1.0/n, n-2)

# Return a new interpolated color map by setting abscissas of original
def colorMapReInterp(s_t_p, cmap):
    s_t = np.concatenate([[0], s_t_p, [1]])
    s_w = np.linspace(0, 1, n_output)    # for output
    return np.column_stack([np.interp(s_w, s_t, cmap[:,j]) for j in [0,1,2]])

# Compute color distance (speed of color change)
def colorDiff(cmap):
    cm_u = sRGB_to_uniform(cmap)
    return np.sqrt(np.sum((cm_u[:-1, :] - cm_u[1:, :]) ** 2, axis=1)) * (len(cmap)+1)

# Divergence of color changes after apply the new coordinate
def rms_color_diff(s_t_p):
    cmap_int = colorMapReInterp(s_t_p, cmap)
    return np.std(colorDiff(cmap_int))

# Object function for minimizing divergence of color changes
# Input is the increment of abscissas
# Other forms are possible, but:
#   * directly use abscissas: main disadvantage is that the optimization
#     functions in scipy do not convergent, probably due to stiffness.
#   * Add penality to make sure monotonicity: slow convergence, seems 
#     better to just use the constrained optimization.
# The extra penality term here is for relexing the constrain on increments,
# i.e. they should add up to one.
def cost_func2(s_t_inc):
    s_t_p = np.cumsum(s_t_inc)
    d = s_t_p[-1]
    s_t_p /= d           # normalize
    rms = rms_color_diff(s_t_p[:-1])
    return rms*rms + 100 * (d-1) ** 2

# Almost the same above, but optimize both the divergence of color changes 
# and divergence of lightness changes.
def cost_func2bw(s_t_inc):
    s_t_p = np.cumsum(s_t_inc)
    d = s_t_p[-1]
    s_t_p /= d           # normalize
    cmap_int = colorMapReInterp(s_t_p[:-1], cmap)
    cm_u = sRGB_to_uniform(cmap_int)
    rms    = np.std(np.sqrt(np.sum((cm_u[:-1, :] - cm_u[1:, :]) ** 2, axis=1)) * (len(cm_u)+1))
    rms_bw = np.std((cm_u[:-1, 0] - cm_u[1:, 0]) * (len(cm_u)+1))
    return rms*rms + rms_bw*rms_bw + 100 * (d-1) ** 2

start_tm = time.time()

x_init = np.diff(np.concatenate([[0], s_t_init, [1]]))
res = minimize(cost_func2, x_init, method='L-BFGS-B',
        bounds=[(0,1) for i in range(n-1)],
        options={'disp':True})

#t_res = np.cumsum(x_init)
t_res = np.cumsum(res.x)
t_res /= t_res[-1]
t_res = t_res[:-1]     # solotion of colormap abscissa

print "time = %.3f sec\n" % (time.time() - start_tm)
print "divergence: %.2f (before)" % (rms_color_diff(s_t_init))
print "divergence: %.2f (after)" % (rms_color_diff(t_res))

cmap_test = colorMapReInterp(t_res, cmap)

# Output cmap
with open('tmp_cmap.csv', 'wb') as f:
    f.write('\n'.join(["%.16f, %.16f, %.16f" % (c[0],c[1],c[2]) for c in cmap_test]))

# plot
s_w = np.linspace(0,1,len(cmap_test))

grid = GridSpec(4, 1)
ax = {};
ax['cmap'] = plt.subplot(grid[0,0])
ax['delta'] = plt.subplot(grid[1,0])
ax['delta_bw'] = plt.subplot(grid[2,0])
ax['maping'] = plt.subplot(grid[3,0])

# The color map
ax['cmap'].imshow(cmap_test[np.newaxis, ...], aspect='auto')

# color changes
ax['delta'].plot(s_w[:-1], colorDiff(cmap_test))
ax['delta'].set_xlim([0,1])

# lightness changes
ax['delta_bw'].plot(s_w[:-1], np.diff(sRGB_to_uniform(cmap_test)[:,0])*(len(cmap_test)-1))
ax['delta_bw'].set_xlim([0,1])

# abscissa map
ax['maping'].plot(np.linspace(0,1,len(t_res)+2), np.concatenate([[0], t_res, [1]]))
ax['maping'].set_xlim([0,1])

plt.show()


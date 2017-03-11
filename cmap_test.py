#!/usr/bin/env python

# Ref:
# Origin (mpl colormaps):
#   https://bids.github.io/colormap/

import csv
import sys
import numpy as np
from os.path import basename

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Usage: ",__FILE__," COLORMAP.csv"
    exit()

with open(fname, 'rb') as f:
    reader = csv.reader(f)
    cmap = np.array(list(reader)).astype(np.float)
#cm = np.floor(cmap[::-1] * 255.99);
cm_data = cmap

from matplotlib.colors import LinearSegmentedColormap
from numpy import nan, inf
test_cm = LinearSegmentedColormap.from_list(basename(fname), cm_data)

import matplotlib.pyplot as plt
import numpy as np

from viscm import viscm
v = viscm(test_cm)
v.fig.set_size_inches(20, 12)
v.fig.savefig(fname+".png")

sys.exit()

#plt.show()


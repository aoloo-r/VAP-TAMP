import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
 
# create a figure
fig = plt.figure()
# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(8)
 
# set width of each subplot as 8
fig.set_figwidth(8)
 
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[1, 2], wspace=0.5)

 # initializing x,y axis value
x = np.arange(0, 10, 0.1)
y = np.cos(x)

ax0 = fig.add_subplot(spec[0])
ax0.plot(x, y)
 
# ax1 will take 0th position in
# geometry(Grid we created for subplots),
# as we defined the position as "spec[1]"
ax1 = fig.add_subplot(spec[1])
ax1.plot(x, y)

plt.show()
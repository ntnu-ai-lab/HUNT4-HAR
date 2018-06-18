import matplotlib.pyplot as plt

# start with one
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3])

# now later you get a new subplot; change the geometry of the existing
n = len(fig.axes)
for i in range(n):
    fig.axes[i].change_geometry(n + 1, 1, i + 1)

# add the new
ax = fig.add_subplot(n + 1, 1, n + 1)
ax.plot([4, 5, 6])

plt.show()


import matplotlib.pyplot as plt

# ================= CONFIGURATION =================

GRAPH_TITLE = "Dynamic Graph Update Performance"

X_LABEL = "Percentage of Edges Modified (%)"
Y_LABEL = "Time (ms)"

INSERTION_NAME = "wiki-Topcats Insertion"
DELETION_NAME = "wiki-Topcats Deletion"

# ==================================================

percentages = [2, 5, 10, 12.5, 20, 25]

# wiki-Topcats measurements (ms)
insertion_times = [56.1367, 57.7475, 66.6767, 61.6366, 77.6878, 84.9633]
deletion_times  = [84.268, 83.1437, 83.668, 87.4721, 90.5789, 92.2399]

# ================= PLOTTING =======================

plt.figure(figsize=(9, 5.5))

plt.plot(
    percentages, insertion_times,
    marker='o', linewidth=2, markersize=7,
    label=INSERTION_NAME
)

plt.plot(
    percentages, deletion_times,
    marker='s', linewidth=2, markersize=7,
    label=DELETION_NAME
)

plt.xlabel(X_LABEL, fontsize=12)
plt.ylabel(Y_LABEL, fontsize=12)
plt.title(GRAPH_TITLE, fontsize=14)

plt.xticks(percentages)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('update_performance.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

time_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
y_values = [0, 2, 7.5, 8, 8.5, 8.3, 8.1, 8.0, 8.2, 8.3, 8.4, 7]
highlight_time = 2.1
highlight_value = 7.8

fig, ax = plt.subplots(figsize=(10, 6))

# Main plot
ax.plot(time_values, y_values, 'bo-', label='Angular velocity (rad/s)')
ax.scatter([highlight_time], [highlight_value], color='red', s=100, label=f'Highlighted value at t = {highlight_time:.2f}s')

# Zoom-in plot (inset) positioned at the center bottom
axins = zoomed_inset_axes(ax, 3, loc='center', borderpad=6, bbox_to_anchor=(21.06 / max(time_values), 4.27 / max(y_values)), bbox_transform=ax.transAxes)
axins.plot(time_values, y_values, 'bo-')
axins.scatter([highlight_time], [highlight_value], color='red', s=100)

# Adjust the limits for the zoomed area to focus on the highlighted point
x1, x2 = highlight_time - 1, highlight_time + 1  # Narrower x-axis range centered on the highlighted point
y1, y2 = highlight_value - 0.5, highlight_value + 0.5  # Narrower y-axis range centered on the highlighted point
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Add a grid for better readability
axins.grid(True)

# Draw a box and lines connecting the zoom area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular velocity (rad/s)')
ax.set_title('Angular velocity (rad/s) vs Time with Zoomed Area')
ax.legend()

plt.show()

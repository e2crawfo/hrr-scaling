from random import *

import argparse
parser = argparse.ArgumentParser(description="Draw a bar graph")
parser.add_argument('-s', action='store_true', help='Supply this argument to save the graph')
argvals = parser.parse_args()

print argvals

save = argvals.s

try:
  import matplotlib as mpl
  if save:
    mpl.use('Agg')

  import matplotlib.pyplot as plt
except:
  pass

#mpl.rc('font', family='serif', serif='Times New Roman')
#mpl.rcParams['lines.linewidth'] = 0.3

import numpy
from operator import *

fig = plt.figure(figsize=(5,2.2.55))

mpl.rcParams.update({'font.size': 9})

#this is the number of different bar locations...typically a single measure
num_vals = 3

#this is the number of bars at each location...typically  series of conditions that we want to compare on multiple measures
bars_per_val = 3

y_vals = [[100.0, 95.5, 99.6], [17.9, 74.9, 51.8], [81.0, 96.4, 93.2]]
error_lo = [[100.0, 94.3, 99.3], [16.2, 71.5, 50.3], [79.4, 95.4, 91.9]]
error_hi = [[100.0, 96.9, 99.8], [19.7, 78.1, 53.3], [82.6, 97.5, 94.5]]

error_lo = [[y - el for y, el in zip(yvl, ell)] for yvl, ell in zip(y_vals, error_lo)]
error_hi = [[eh - y for y, eh in zip(yvl, ehl)] for yvl, ehl in zip(y_vals, error_hi)]


hatch = ['/', '|||||', '\\', 'xxx', '||', '--', '+', 'OO', '...', '**']
labels = ["Core", "Consistent", "Unitary"]
color = [[0.33] * 3, [0.5] * 3, [0.66] * 3]

cross_measurement_spacing = 0.3 
within_measurement_spacing = 0.0 
bar_width = 0.4

#plt.figure(figsize=(4,2))
plt.title("Neural Model Performance")
plt.ylabel("% Correct")

bar_left_positions = [[] for b in range(bars_per_val)]
val_offset = 0
middles = []
for i in range(num_vals):
  val_offset += cross_measurement_spacing

  left_side = val_offset

  for j in range(bars_per_val):
    if j > 0:
      val_offset += within_measurement_spacing
      val_offset += bar_width

    bar_left_positions[j].append(val_offset)

  val_offset += bar_width
    
  right_side = val_offset

  middles.append(float(left_side + right_side) / 2.0)
print bar_left_positions

for blp, yv, cl, el, eh, lb in zip(bar_left_positions, y_vals, color, error_lo, error_hi, labels):
  plt.bar(blp, yv, color=cl, width = bar_width, linewidth=0.2, yerr=[el, eh], ecolor="black", label=lb, error_kw = {"linewidth":0.5, "capsize":1.5})

legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':9}, handlelength=.75, handletextpad=.5, shadow=False, frameon=False)

ax = fig.axes[0]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles, labels = ax.get_legend_handles_labels()

ticks = middles
tick_labels = ["Single Edge", "Hierarchal", "Sentence"]

plt.xticks(ticks, tick_labels)
plt.ylim([0.0, 105.0])
plt.xlim([0.0, val_offset + cross_measurement_spacing])
plt.axhline(100.0, linestyle='--', color='black')

plt.show()

if save:
  plt.savefig('prgraph.pdf')



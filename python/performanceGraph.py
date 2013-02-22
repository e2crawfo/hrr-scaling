from random import *
import matplotlib.pyplot as plt
import numpy
from operator import *


num_vals = 7

bars_per_val = 2
data = [range(1,num_vals+1) for i in range(bars_per_val)]

#hatch = ['xxx', 'ooo', '\\\\', '///', '||', '--', '+', 'OO', '...', '**']
hatch = ['xxx', '...']
labels = ["Dot-product with correct", "Next largest dot-product"]
#hatch = hatch[0:min(bars_per_val, len(hatch) + 1)]
color = ['white'] * bars_per_val


#plt.figure(figsize=(4,2))
plt.title("Dot Products")

y_vals = zip(*data)
y_vals = [i for y in y_vals for i in [0] + list(y) ]

for i in range(bars_per_val):
  r = range(i + 1, (bars_per_val + 1) * (num_vals-1) + i + 2 , bars_per_val + 1)
  plt.bar( r, data[i], hatch=hatch[i], color=color[i], label=labels[i])

for i in range(num_vals * bars_per_val):
  a = mod(i, bars_per_val)
  b = i / bars_per_val
  print a, b
  plt.errorbar( (1 + i / bars_per_val) + i + 0.4, data[a][b], yerr=[[.2], [.2]], color='black')

#data = [0, 1, 2, 0, 3, 4, 0, 
#data1 = [5, 6, 0, 7, 8, 0]

#xs = range
#xs1

#hatch = ['x', 'x', 'o', 'x', 'x', 'o', 'x', 'x', 'o', 'x', 'x', 'o', 'x']

#plt.bar([1,2,3,4,5,6,7,8,9,10,11,12, 13], data, 1, bottom=0, hatch=['x'])
#plt.subplot(212, axisbg='y')

#plt.legend()
plt.show()



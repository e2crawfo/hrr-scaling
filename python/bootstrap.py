import random

#bootstrap.py!
def draw_bootstrap_samples(data, num, rng=random):
  samples = []
  for i in range(num):

    sample = []
    for j in range(len(data)):
      val = rng.sample(data, 1)
      sample.extend(val)

    samples.append(sample)

  return samples

def get_bootstrap_stats(stat_func, data, num):
  samples = draw_bootstrap_samples(data, num)

  stats = []
  for s in samples:
    stats.append(stat_func(s))

  return stats

def bootstrap_CI(alpha, stat_func, data, num) :
  stats = get_bootstrap_stats(stat_func, data, num)
  stats.sort()
  lower_CI_bound = stats[int(round((num + 1) * alpha / 2.0 ))]
  upper_CI_bound = stats[int(round((num + 1) * (1 - alpha/2.0)))]

  return (lower_CI_bound, upper_CI_bound)

class Bootstrapper:

  def __init__(self):
    self.data = {}

  def add_data(self, index, data):
    if not (index in self.data):
      self.data[index] = []

    self.data[index].append(data)

  def print_summary(self, output_file):
    mean = lambda x: float(sum(x)) / float(len(x))

    data_keys = self.data.keys()
    data_keys.sort()

    for n in data_keys:
      s = self.data[n]
      CI = bootstrap_CI(0.05, mean, s, 999)
      largest = max(s)
      smallest = min(s)

      output_file.write("\nmean " + n + ": " + str(mean(s)) + "\n")
      output_file.write("lower 95% CI bound: " + str(CI[0]) + "\n")
      output_file.write("upper 95% CI bound: " + str(CI[1]) + "\n")
      output_file.write("max: " + str(largest) + "\n")
      output_file.write("min: " + str(smallest) + "\n")
      output_file.write("num_samples: " + str(len(s)) + "\n")
      output_file.write("raw data: " + str(s) + "\n")


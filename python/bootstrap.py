import random

#bootstrap.py!

class Bootstrapper:

  def __init__(self, verbose=False, write_raw_data=False, seed=1):
    self.data = {}
    self.verbose = verbose
    self.write_raw_data = write_raw_data
    self.seed = seed 
    self.rng = random.Random(seed)

  def add_data(self, index, data):
    if not (index in self.data):
      self.data[index] = []

    self.data[index].append(data)

    if self.verbose:
      print "Bootstrapper adding data ... name: ", index, ", data: ", data

  def draw_bootstrap_samples(self, data, num):
    samples = []
    for i in range(num):

      sample = []
      for j in range(len(data)):
        val = self.rng.sample(data, 1)
        sample.extend(val)

      samples.append(sample)

    return samples

  def get_bootstrap_stats(self, stat_func, data, num):
    samples = self.draw_bootstrap_samples(data, num)

    stats = []
    for s in samples:
      stats.append(stat_func(s))

    return stats

  def bootstrap_CI(self, alpha, stat_func, data, num) :
    stats = self.get_bootstrap_stats(stat_func, data, num)
    stats.sort()
    lower_CI_bound = stats[int(round((num + 1) * alpha / 2.0 ))]
    upper_CI_bound = stats[int(round((num + 1) * (1 - alpha/2.0)))]

    return (lower_CI_bound, upper_CI_bound)

  def print_summary(self, output_file):
    mean = lambda x: float(sum(x)) / float(len(x))

    data_keys = self.data.keys()
    data_keys.sort()

    for n in data_keys:
      s = self.data[n]
      CI = self.bootstrap_CI(0.05, mean, s, 999)
      largest = max(s)
      smallest = min(s)

      output_file.write("\nmean " + str(n) + ": " + str(mean(s)) + "\n")
      output_file.write("lower 95% CI bound: " + str(CI[0]) + "\n")
      output_file.write("upper 95% CI bound: " + str(CI[1]) + "\n")
      output_file.write("max: " + str(largest) + "\n")
      output_file.write("min: " + str(smallest) + "\n")
      output_file.write("num_samples: " + str(len(s)) + "\n")

      if self.write_raw_data:
        output_file.write("raw data: " + str(s) + "\n")


import random
import re

import utilities as util


def draw_bootstrap_samples(data, num, rng=random):
  samples = []
  for i in range(num):

    sample = []
    for j in range(len(data)):
      val = rng.sample(data, 1)
      sample.extend(val)

    samples.append(sample)

  return samples


def get_bootstrap_stats(stat_func, data, num, rng=random):
  samples = draw_bootstrap_samples(data, num, rng)

  stats = []
  for s in samples:
    stats.append(stat_func(s))

  return stats


def bootstrap_CI(alpha, stat_func, data, num, rng=random):
  stats = get_bootstrap_stats(stat_func, data, num, rng)
  stats.sort()
  lower_CI_bound = stats[int(round((num + 1) * alpha / 2.0))]
  upper_CI_bound = stats[int(round((num + 1) * (1 - alpha / 2.0)))]

  return lower_CI_bound, upper_CI_bound


class Bootstrapper:

  def __init__(self, verbose=False, write_raw_data=False, seed=1):
    self.data = {}
    self.verbose = verbose
    self.write_raw_data = write_raw_data
    self.seed = seed
    self.rng = random.Random(seed)
    self.float_re = re.compile(r"""-*\d +  # the integral part
                   \.    # the decimal point
                   \d *  # some fractional digits""", re.X)

  def read_bootstrap_file(self, filename, match_regex=r".*", ignore_regex=r"a^"):
    """Collects data from a file previously created from an instance of the Bootstrap class
    using the print_summary function. Adds that data to the current Bootstrapper instance.
    Only collects data from the last "Bootstrap Summary" in the file.
    Also requires that the Bootstrapper objects that wrote the file had write_raw_data=True.

    :param filename: The name of the file to load bootstrap data from.
    :type string

    :param match_regex: A string specifying a regular expression. The function will only read
    data fields which match this regex.
    :type string

    :param ignore_regex: A string specifying a regular expression. The function will ignore
    all data fields that match this regex.
    :type string
    """

    match_regex = re.compile(match_regex)
    ignore_regex = re.compile(ignore_regex)

    num_summaries = 0
    with open(filename) as bs_file:
      for line in bs_file:
        if "Bootstrap Summary" in line and not "End" in line:
          num_summaries += 1

    if not num_summaries:
      return

    i = 0
    with open(filename) as bs_file:
      for line in bs_file:
        if "Bootstrap Summary" in line and not "End" in line:
          i += 1

          if i == num_summaries:
            break

      line = bs_file.next()
      line = bs_file.next()

      while not "End Bootstrap Summary" in line:
        name = bs_file.next()
        name = re.split('\W+', name)[1]

        lCI = bs_file.next()
        uCI = bs_file.next()
        mx = bs_file.next()
        mn = bs_file.next()
        num_samples = bs_file.next()

        raw_data = bs_file.next()

        if not "raw data" in raw_data:
          return

        raw_data = self.float_re.findall(raw_data)

        if match_regex.search(name) and not ignore_regex.search(name):
          for rd in raw_data:
            self.add_data(name, rd)

        line = bs_file.next()

  def add_data(self, index, data):
    """
    Add data to the bootstrapper. data gets appended to the end of the list
    referred to by index. If such a list doesn't yet exist in the current
    bootstrapper, one is created.

    :param index: The index of the list that data is to be appended to.
    :type hashable:

    :param data: The data to add to the list reffered to by index
    :type data:

    """

    if not (index in self.data):
      self.data[index] = []

    self.data[index].append(float(data))

    if self.verbose:
      print "Bootstrapper adding data ... name: ", index, ", data: ", data

  def get_stats(self, index):
    """
    Retrieve a set of stats about the numbers in the list referred to by
    index. The stats are returned in a tuple, whose order is:
      (raw data, mean, (low_CI, hi_CI), largest, smallest).
    If index is not in the current bootstrapper, None is returned.

    :param index: The index of the list whose stats are to be reported
    :type hashable:

    """

    if index not in self.data:
      return None

    mean = lambda x: float(sum(x)) / float(len(x))

    s = self.data[index]
    m = mean(s)
    CI = bootstrap_CI(0.05, mean, s, 999)
    largest = max(s)
    smallest = min(s)
    return (s, m, CI, largest, smallest)

  def print_summary(self, output_file):
    """
    Prints a summary of the data currently stored in the bootstrapper.
    Basically, we call get_stats on each index in the bootstrapper.

    :param outfile: Place to send the summary data
    :type fileobject or string (filename):

    """

    close = False
    if isinstance(output_file, str):
      output_file = open(output_file,'w')
      close = True

    title = "Bootstrap Summary"
    util.print_header(output_file, title)

    data_keys = self.data.keys()
    data_keys.sort()

    for n in data_keys:
      s, m, CI, largest, smallest = self.get_stats(n)

      output_file.write("\nmean " + str(n) + ": " + str(m) + "\n")
      output_file.write("lower 95% CI bound: " + str(CI[0]) + "\n")
      output_file.write("upper 95% CI bound: " + str(CI[1]) + "\n")
      output_file.write("max: " + str(largest) + "\n")
      output_file.write("min: " + str(smallest) + "\n")
      output_file.write("num_samples: " + str(len(s)) + "\n")

      if self.write_raw_data:
        output_file.write("raw data: " + str(s) + "\n")

    util.print_footer(output_file, title)

    if close:
      output_file.close()

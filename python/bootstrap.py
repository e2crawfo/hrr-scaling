import random

#bootstrap.py!
def draw_bootstrap_samples(data, num):
  samples = []
  for i in range(num):

    sample = []
    for j in range(len(data)):
      val = random.sample(data, 1)
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


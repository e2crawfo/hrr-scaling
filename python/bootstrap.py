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


#test

data = [random.randint(0, 10) for i in range(20)]
print bootstrap_CI(0.05, lambda x: float(sum(x)) / float(len(x)), data, 999)
print sum(data) / len(data)

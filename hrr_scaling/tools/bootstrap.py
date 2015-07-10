import random
import re
import os
from collections import defaultdict, deque
import inspect

import file_helpers


def draw_bootstrap_samples(data, num, rng=random):
    """
    Draw num bootstrap samples. Each bootstrap sample consists
    of sampling from data with replacement len(data) times
    """
    L = len(data)
    samples = [[rng.choice(data) for i in range(L)]
               for n in range(num)]

    return samples


def get_bootstrap_stats(stat_func, data, num, rng=random):
    """
    Draw num bootstrap samples, and then apply stat_func to those samples
    to generate bootstrapped statistics.
    """
    samples = draw_bootstrap_samples(data, num, rng)

    stats = []
    for s in samples:
        stats.append(stat_func(s))

    return stats


def bootstrap_CI(alpha, stat_func, data, num, rng=random):
    """
    Generates bootstrapped confidence interval statistics.
    Calls get_bootstrap_stats with the appropriate functions.
    """
    stats = get_bootstrap_stats(stat_func, data, num, rng)
    stats.sort()
    lower_CI_bound = stats[int(round((num + 1) * alpha / 2.0))]
    upper_CI_bound = stats[int(round((num + 1) * (1 - alpha / 2.0)))]

    return lower_CI_bound, upper_CI_bound


def add_data(index, data):
    """
    Module-level add_data. Adds data to the Bootstrapper object managing the
    current context. Raises an exception if no Bootstapper context is active.
    """

    if len(Bootstrapper.context) == 0:
        raise RuntimeError(
            "module-level add_data must be called in a context managed "
            "by an instance of the Bootstrapper class.")

    bs = Bootstrapper.context[-1]

    if not isinstance(bs, Bootstrapper):
        raise RuntimeError(
            "module-level add_data must be called in a context managed "
            "by an instance of the Bootstrapper class.")

    bs.add_data(index, data)


class Bootstrapper:

    context = deque(maxlen=100)  # static stack of bootstrapper objects

    def __enter__(self):
        Bootstrapper.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Bootstrapper.context) == 0:
            raise RuntimeError(
                "Bootstrapper.context in bad state; was empty when "
                "exiting from a 'with' block.")

        bs = Bootstrapper.context.pop()
        if bs is not self:
            raise RuntimeError(
                "Bootstrapper.context in bad state; was expecting "
                "current context to be '%s' but instead got "
                "'%s'." % (self, bs))

    def __init__(self, verbose=False, write_raw_data=False, seed=1):
        self.order = []
        self.data = defaultdict(list)
        self.verbose = verbose
        self.write_raw_data = write_raw_data
        self.seed = seed
        self.rng = random.Random(seed)
        self.float_re = re.compile(r'-*\d+.\d+(?:e-?\d+)?', re.X)

    def read_bootstrap_file(self, filename,
                            match_regex=r".*", ignore_regex=r"a^"):
        """
        Collects data from a file previously created from an instance of the
        Bootstrap class using the print_summary function. Adds that data to the
        current Bootstrapper instance. Only collects data from the last
        "Bootstrap Summary" in the file. Also requires that the Bootstrapper
        objects that wrote the file had write_raw_data=True.

        :param filename: The name of the file to load bootstrap data from.
        :type string

        :param match_regex: A string specifying a regular expression. The
                            function will only read data fields which
                            match this regex.
        :type string

        :param ignore_regex: A string specifying a regular expression. The
                             function will ignore all data fields that
                             match this regex.
        :type string
        """

        match_regex = re.compile(match_regex)
        ignore_regex = re.compile(ignore_regex)

        if not os.path.isfile(filename):
            raise IOError(
                "read_bootstrap_file: %s is not a valid file" % filename)

        num_summaries = 0
        with open(filename) as bs_file:
            for line in bs_file:
                if "Bootstrap Summary" in line and "End" not in line:
                    num_summaries += 1

        if not num_summaries:
            raise IOError(
                "read_bootstrap_file: %s is not a "
                " valid bootstrap file" % filename)

        if self.verbose:
            print ("Bootstrapper reading from file: %s" % filename)

        i = 0
        with open(filename) as bs_file:
            for line in bs_file:
                if "Bootstrap Summary" in line and "End" not in line:
                    i += 1

                    if i == num_summaries:
                        break

            line = bs_file.next()

            while "End Bootstrap Summary" not in line:
                name = bs_file.next()
                name = re.split('\W+', name)[1]

                bs_file.next()  # lower CI bound
                bs_file.next()  # upper CI bound
                bs_file.next()  # max
                bs_file.next()  # min
                bs_file.next()  # num samples

                raw_data = bs_file.next()

                if "raw data" not in raw_data:
                    raise IOError("Error reading bootstrap file."
                                  " No raw data in file")

                if match_regex.search(name) and not ignore_regex.search(name):
                    raw_data = self.float_re.findall(raw_data)

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
        data = float(data)

        self.order.append((index, data))
        self.data[index].append(data)

        if self.verbose:
            print ("Bootstrapper adding data ..."
                   "name: %s, data: %s" % (index, data))

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

    def print_summary(self, output_file, flush=False):
        """
        Prints a summary of the data currently stored in the bootstrapper.
        Basically, we call get_stats on each index in the bootstrapper.

        :param outfile: Place to send the summary data
        :type fileobject or string (filename):
        """

        close = False
        if isinstance(output_file, str):
            output_file = open(output_file, 'w')
            close = True

        if self.verbose:
            print ("Bootstrapper writing summary to file...")

        title = "Bootstrap Summary"
        print_header(output_file, title)

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

        print_footer(output_file, title)

        if flush:
            output_file.flush()

        if close:
            output_file.close()

        if self.verbose:
            print ("Bootstrapper done writing to file...")


def print_header(output_file, string, char='*', width=15, left_newline=True):
    line = char * width
    string = line + " " + string + " " + line + "\n"

    if left_newline:
        string = "\n" + string

    output_file.write(string)


def print_footer(output_file, string, char='*', width=15):
    print_header(output_file, "End " + string, char=char,
                 width=width, left_newline=False)


def filestrap(prefix, memoized_args=None, can_delete=True):
    """
    The goal is basically to provide persistent memoization using the
    Bootstrapper class.  Useful for quickly memoizing small exploratory
    functions, where we only want to collect the data once, but might have
    to rerun analysis or plotting many times in order to make adjustments.

    prefix is the prefix of the files that get written. Can include a directory
    and that directory will get created if it doesn't already exist.

    memoized_args is the set of args that will be memoized if they are provided
    (i.e. if the function is called with different values of these args, then a
    new file is created.)

    Returns a decorator that can be applied to functions such that whenever
    that function is called, a number of things happen:

    1. An instance of the Bootstrapper class is created.

    2. The Bootstrapper object tries to load data from a file whose name is
    determined by prefix and the arguments supplied to the function.

    3. If the file is found to be a valid summary file created by an instance
    of the Bootstrapper class, then the data is loaded into the current
    Bootstrapper, and the function returns without calling the
    decorated function.

    4. Otherwise, the decorated function is called in a context managed by the
    Bootstrapper object (so calls to add_data inside that function go to that
    Bootstrapper object).

    5. Finally, the Bootstrapper, populated by the add_data calls inside the
    function, is written to a file with the same name as before.

    6. Whether or not the decorated function is called, the result of calling
    the function is the Bootstrapper object. If there was a return value from
    calling the decorated function, then it is stored in Bootstrapper object
    under the index 'return_value'.
    """

    def decorator(func):

        if memoized_args is not None:
            argspec = inspect.getargspec(func)
            default_index = len(argspec.args) - len(argspec.defaults)
            valid_args = all([x in argspec.args for x in memoized_args])

            if not (argspec.keywords or valid_args):
                raise ValueError(
                    "memoized_args must be names of possible "
                    "arguments to decorated function")

            arg_indices = []
            for arg in memoized_args:
                try:
                    arg_indices.append(argspec.args.index(arg))
                except:
                    arg_indices.append(None)

        def f(*args, **kwargs):

            prefix_dir, prefix_title = os.path.split(prefix)

            if memoized_args is not None:
                memoized_argvals = {}

                for arg, idx in zip(memoized_args, arg_indices):
                    if idx is not None and idx < len(args):
                        memoized_argvals[arg] = args[idx]
                    elif arg in kwargs:
                        memoized_argvals[arg] = kwargs[arg]
                    elif (idx is not None and
                          idx >= default_index and idx < len(argspec.args)):
                        memoized_argvals[arg] = \
                            argspec.defaults[idx - default_index]

                filename = file_helpers.make_filename(
                    prefix_title, prefix_dir, config_dict=memoized_argvals,
                    use_time=False)
            else:
                filename = file_helpers.make_filename(
                    prefix_title, prefix_dir, use_time=False)

            bootstrapper = Bootstrapper(
                verbose=True, write_raw_data=True)

            loaded = False

            if os.path.isfile(filename):
                need_delete = False

                try:
                    bootstrapper.read_bootstrap_file(filename)
                    loaded = True
                except IOError:
                    need_delete = True

                if can_delete and need_delete:
                    try:
                        os.remove()
                    except:
                        raise IOError(
                            'Filename %s points to invalid bootstrap file, '
                            'but it cannot be deleted '
                            '(maybe it is a directory?).' % filename)

            if not loaded:
                with bootstrapper:
                    ret = func(*args, **kwargs)

                bootstrapper.add_data('return_value', ret)

                bootstrapper.print_summary(filename, flush=True)

            return bootstrapper

        return f

    return decorator

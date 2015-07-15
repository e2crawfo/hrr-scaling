import ConfigParser
import argparse


def parse_args(print_args=False):
    parser = argparse.ArgumentParser(description='Test an HRR extractor.')

    parser.add_argument(
        'num_runs', type=int,
        help="Run a jump test with this number of trials")

    parser.add_argument(
        '--jump', default=0, type=int,
        help="Run a jump test with this number of trials")

    parser.add_argument(
        '--hier', default=0, type=int,
        help="Run a hierarchical test with this number of trials")

    parser.add_argument(
        '--sent', default=0, type=int,
        help="Run a sentence test with this number of trials")

    parser.add_argument(
        '--deep', default=0, type=int,
        help="Run a deep sentence test with this number of trials")

    parser.add_argument(
        '--expr', nargs=2,
        help="(num trials, expression). Run num trials tests generated "
        "from the give HRR expression).")

    # parameters of the experiment
    parser.add_argument(
        '--unitary-roles', dest='unitary_roles', action='store_true',
        help='Supply this argument to make role vectors '
             'in the sentence test unitary.')

    parser.add_argument(
        '--unitary-rels', dest='unitary_relations', action='store_true',
        help='Supply this argument to make relation vectors unitary.')

    parser.add_argument(
        '--no-ids', dest='no_ids', action='store_true',
        help='Supply this to omit ID-vectors. Define semantic pointers '
             'directly in terms of other semantic pointers')

    parser.add_argument(
        '--no-normalize', dest='no_norm', action='store_true',
        help='Supply this to skip normalizing semantic pointers')

    parser.add_argument(
        '--sp-noise', dest='sp_noise', default=0, type=int,
        help='Number of noise terms to add to semantic pointers. '
             'Defaults to 0')

    parser.add_argument(
        '-d', default=512, type=int,
        help='Specify the number of dimensions to use.')

    parser.add_argument(
        '-p', default=1.0, type=float,
        help='Specify the proportion of Wordnet synsets to use. '
             'A float between 0 and 1.')

    parser.add_argument(
        '--corpus-seed', default=-1, type=int,
        help='Seed for the random number generator that creates the corpuses '
             'and the vectors encoding it.')

    parser.add_argument(
        '--extractor-seed', default=-1, type=int,
        help='Seed for the random number generator that makes the extractors.')

    parser.add_argument(
        '--test-seed', default=-1, type=int,
        help='Seed for the random number generator that creates the tests.')

    parser.add_argument(
        '--seed', default=-1, type=int,
        help='Seed for the random number generator that controls everything. '
             'Overrides the other seeds if supplied.')

    # picking the type of extractor
    parser.add_argument(
        '--abstract', action='store_true',
        help='Supply this argument do extraction using pure '
             'linear algebra (i.e. thresholding and summing).')

    # parameters for the neural network
    parser.add_argument(
        '-t', default=0.3, type=float,
        help='Specify the cleanup threshold. A float between 0 and 1.')

    parser.add_argument(
        '--synapse', default=0.02, type=float,
        help='Post-synaptic time constant. Controls the shape of the '
             'post-synaptic current.')

    parser.add_argument(
        '--steps', default=100, type=int,
        help='Number of steps to run the neural model for.')

    # configuring gpus
    parser.add_argument(
        '--gpus', nargs='+', type=int,
        help='Specify the devices (gpus) to use. Specified as a list '
             'of integers. e.g. "python run.py j 10 100 --gpus 0 2 3" '
             'would use 3 devices, skipping the device with index 1.')

    parser.add_argument(
        '--ocl', nargs='+', type=int,
        help='Specify the devices (gpus) to use for the ocl simulator '
             'if using the new nengo code (i.e. if --new is supplied). '
             'Specified as a list of integers. (e.g. "python run.py j 10 '
             '100 --ocl 0 2 3" would use 3 devices for the ocl simulators, '
             'skipping the device with index 1). --ocl and --gpus'
             'can be combined, but their intersection should be empty.')

    parser.add_argument(
        '--identical', action='store_true',
        help='To make all association ensembles identical when using '
             'gpu assoc with new nengo')

    parser.add_argument(
        '--fast', action='store_true',
        help='Whether to use fast gpu algorithm with new nengo code')

    parser.add_argument(
        '--numwords', default=0, type=int,
        help='Number of planned words. Only has an effect on jump tests.')

    # not used very often
    parser.add_argument(
        '-b', action='store_true',
        help='Supply this argument to use bidirectional relations.')

    parser.add_argument(
        '--plot', action='store_false',
        help='Supply this argument to create and save plots of the '
             'activities of the cleanup populations (only works in '
             'neural mode). The graphs are stored in the "graphs" directory.')

    parser.add_argument(
        '--noneg', action='store_true',
        help='Supply this argument to only do positive runs on hier test.')

    parser.add_argument(
        '--shortsent', action='store_true',
        help='Supply this arg to only do a single run of a sentence test.')

    parser.add_argument(
        '--num-synsets', dest='num_synsets', default=-1, type=int,
        help='Set the number of synsets to use.')

    parser.add_argument(
        '--probe-all', action='store_true', dest="probe_all",
        help='Probe all association nodes.')

    parser.add_argument(
        '--name', dest='name', type=str, default="results",
        help='File to write results to.')

    argvals = parser.parse_args()

    if print_args:
        print argvals

    return argvals


def create_outfile_suffix(neural, unitary, id_vecs=True, bidir=False):
    suff = "_"

    if neural:
        suff += "n"

    if unitary:
        suff += "u"

    if id_vecs:
        suff += "i"

    if bidir:
        suff += "b"

    return suff


def read_config(config_name="config"):
    configParser = ConfigParser.SafeConfigParser()
    configParser.readfp(open(config_name))

    input_dir = configParser.get("Directories", "input_dir")
    output_dir = configParser.get("Directories", "output_dir")
    return (input_dir, output_dir)


def setup_relation_stats(largest_degree=1000):
    relation_stats = zip(
        range(largest_degree),
        [[[], [], [], 0] for i in range(largest_degree)])

    relation_stats = dict(relation_stats)

    return relation_stats


def print_header(output_file, string, char='*', width=15, left_newline=True):
    line = char * width
    string = line + " " + string + " " + line + "\n"

    if left_newline:
        string = "\n" + string

    output_file.write(string)


def print_footer(output_file, string, char='*', width=15):
    print_header(output_file, "End " + string, char=char,
                 width=width, left_newline=False)
    output_file.write("\n")

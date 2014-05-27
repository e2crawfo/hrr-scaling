# dodo.py

import datetime
import string
import copy
import os

import run
from mytools.bootstrap import Bootstrapper

DOIT_CONFIG = {'verbosity': 2}

num_subtasks = 3

test_names = ['jump', 'hierarchical', 'deep']

date_time_string = str(datetime.datetime.now()).split('.')[0]
date_time_string = reduce(
    lambda y, z: string.replace(y, z, "_"),
    [date_time_string, ":", " ", "-"])

# directory = '/data/e2crawfo/hrr-scaling/experiment'
directory = ('/home/e2crawfo/hrr-scaling/experiments/experiment_'
             + date_time_string)

if not os.path.exists(directory):
    os.makedirs(directory)

args = {'num_runs': 2, 'jump_trials': 3, 'hier_trials': 3, 'sent_trials': 0,
        'deep_trials': 3, 'expr': 0, 'unitary_roles': True,
        'short_sentence': False, 'do_neg': True, 'corpus_seed': 100,
        'extractor_seed': 200, 'test_seed': 300, 'seed': None,
        'dimension': 512, 'num_synsets': 100, 'proportion': 1.0,
        'unitary_relations': False, 'abstract': True,
        'synapse': 0.005, 'timesteps': 100, 'threshold': 0.3,
        'probeall': False, 'identical': True, 'fast': False, 'plot': True,
        'show': False, 'gpus': [], 'ocl': [], 'outfile_format': ""}


def consolidate_bootstraps(input_filenames, summary_filename):

    bs = Bootstrapper(write_raw_data=True)

    for fn in input_filenames:
        bs.read_bootstrap_file(fn)

    bs.print_summary(summary_filename)


def task_experiment():
    results_files = []

    for subtask in range(num_subtasks):

        filename_format = '%s/%%s_subtask_%g' % (directory, subtask)

        subtask_args = copy.deepcopy(args)
        subtask_args['outfile_format'] = filename_format
        subtask_args['gpus'] = [subtask]

        targets = [filename_format % t for t in test_names]
        results_files.extend(targets)

        yield {
            'name': 'Subtask %g' % subtask,
            'actions': [(run.run, [], subtask_args)],
            'file_dep': [],
            'targets': targets,
            }

    experiment_file = directory + '/summary'

    yield {
        'name': 'Consolidate',
        'actions': [(consolidate_bootstraps,
                     [results_files, experiment_file])],
        'file_dep': results_files,
        'targets': [experiment_file]
        }
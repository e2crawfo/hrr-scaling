# dodo.py
import datetime
import string
import copy
import os
from doit import get_var
from doit.tools import run_once

import run
import plot
from mytools.bootstrap import Bootstrapper

DOIT_CONFIG = {'verbosity': 2}

if __name__ != "__main__":
    DOIT_CONFIG['datetime'] = get_var('datetime', '')

num_subtasks = 4

test_names = ['jump', 'hierarchical', 'deep']

date_time_string = DOIT_CONFIG.get('datetime', '')

if not date_time_string:
    date_time_string = str(datetime.datetime.now()).split('.')[0]
    date_time_string = reduce(
        lambda y, z: string.replace(y, z, "_"),
        [date_time_string, ":", " ", "-"])

# experiment_directory = '/home/e2crawfo/hrr-scaling/experiments'
experiment_directory = '/data/e2crawfo/hrr-scaling/experiments'

directory = (experiment_directory+'/experiments_'
             + date_time_string)

if not os.path.exists(directory):
    os.makedirs(directory)

args = {'num_runs': 5, 'jump_trials': 100, 'hier_trials': 20, 'sent_trials': 0,
        'deep_trials': 30, 'expr': 0, 'unitary_roles': True,
        'short_sentence': False, 'do_neg': True, 'corpus_seed': -1,
        'extractor_seed': -1, 'test_seed': -1, 'seed': 1000,
        'dimension': 512, 'num_synsets': -1, 'proportion': 1.0,
        'unitary_relations': False, 'abstract': False,
        'synapse': 0.005, 'timesteps': 75, 'threshold': 0.3,
        'probeall': False, 'identical': True, 'fast': False, 'plot': True,
        'show': False, 'gpus': [], 'ocl': [], 'outfile_format': ""}

neural_summary = directory + '/neural_summary'
abstract_summary = directory + '/abstract_summary'

results_files = []
for s in ['abstract', 'neural']:
    for st in range(num_subtasks):
        for tn in test_names:
            fn = '%s/%s_%s_subtask_%g' % (directory, s, tn, st)
            results_files.append(fn)


def consolidate_bootstraps(input_filenames, summary_filename):

    bs = Bootstrapper(write_raw_data=True)

    for fn in input_filenames:
        bs.read_bootstrap_file(fn)

    bs.print_summary(summary_filename)


def task_neural_experiments():

    for subtask in range(num_subtasks):

        filename_format = '%s/neural_%%s_subtask_%g' % (directory, subtask)

        subtask_args = copy.deepcopy(args)
        subtask_args['outfile_format'] = filename_format
        subtask_args['seed'] += subtask * 1000
        subtask_args['gpus'] = [subtask]
        subtask_args['abstract'] = False

        targets = [filename_format % t for t in test_names]

        print targets

        yield {
            'name': 'Subtask %g' % subtask,
            'actions': [(run.run, [], subtask_args)],
            'file_dep': [],
            'targets': targets,
            'uptodate': [run_once]
            }

    neural_results_files = filter(lambda x: 'neural' in x, results_files)

    yield {
        'name': 'Consolidate',
        'actions': [(consolidate_bootstraps,
                     [neural_results_files, neural_summary])],
        'file_dep': neural_results_files,
        'targets': [neural_summary]
        }


def task_abstract_experiments():
    for subtask in range(num_subtasks):

        filename_format = '%s/abstract_%%s_subtask_%g' % (directory, subtask)

        subtask_args = copy.deepcopy(args)
        subtask_args['outfile_format'] = filename_format
        subtask_args['seed'] += subtask * 1000
        subtask_args['abstract'] = True

        targets = [filename_format % t for t in test_names]

        yield {
            'name': 'Subtask %g' % subtask,
            'actions': [(run.run, [], subtask_args)],
            'file_dep': [],
            'targets': targets,
            'uptodate': [run_once]
            }

    abstract_results_files = filter(lambda x: 'abstract' in x, results_files)

    yield {
        'name': 'Consolidate',
        'actions': [(consolidate_bootstraps,
                     [abstract_results_files, abstract_summary])],
        'file_dep': abstract_results_files,
        'targets': [abstract_summary]
        }


def plot_results(summary_filenames, keys, **kwargs):
    """
    Read the summary bootstrap files and create a plot using
    plot_performance.

    Params:

    summary_filenames: list of str
        The names of the summary bootstrap files, with the data
        from all experiments.
    keys: list of str
        The keys to extract from the summary bootstrap

    **kwargs is passed to plot_performance

    """

    bs = Bootstrapper(write_raw_data=True)

    means = []
    low_cis = []
    high_cis = []

    for fn in summary_filenames:
        bs.read_bootstrap_file(fn)

        mean = []
        low_ci = []
        high_ci = []

        for key in keys:
            stats = bs.get_stats(key)

            mean.append(stats[1] * 100)
            low_ci.append(stats[2][0] * 100)
            high_ci.append(stats[2][1] * 100)

        means.append(mean)
        low_cis.append(low_ci)
        high_cis.append(high_ci)

    plot.plot_performance(means, low_cis, high_cis, **kwargs)


def task_performance_plot():
    summary_filenames = [abstract_summary, neural_summary]
    plot_filename = directory + '/prgraph.pdf'

    kwargs = {
        'summary_filenames': [abstract_summary, neural_summary],

        'keys': ['jump_score_correct', 'hierarchical_score',
                 'sentence_score_1', 'sentence_score_2'],

        'measure_labels': ["Simple", "Hierarchical",
                           "Sentence\n(Surface)",
                           "Sentence\n(Embedded)"],

        'condition_labels': ["Abstract", "Neural"],

        'filename': plot_filename
        }

    yield {
        'name': 'performance_graph',
        'actions': [(plot_results, [], kwargs)],
        'file_dep': summary_filenames,
        'targets': [plot_filename]
        }


def task_tuning_curve_plot():

    filename = 'tuning_curves.pdf'
    yield {
        'name': 'tuning_curves',
        'actions': [(plot.plot_tuning_curves, [filename])],
        'targets': [filename]
        }


def task_hierarchical_simulation():
    yield {
        'name': 'hierarchical_simulation',
        'actions': [(plot.plot_hierarchical_simulation, [])],
        'targets': ['']
    }

# if __name__ == '__main__':
#     task_iter = task_neural_experiments()
#     for task in task_iter:
#         actions = task['actions']
#         for action in actions:
#            action[0](*action[1], **action[2])

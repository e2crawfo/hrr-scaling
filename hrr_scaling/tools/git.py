import subprocess


def get_git_revision_hash():
    try:
        h = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except subprocess.CalledProcessError:
        h = None

    return h


def get_git_revision_short_hash():
    try:
        h = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    except subprocess.CalledProcessError:
        h = None

    return h
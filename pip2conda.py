"""Converts a python requirements file to a conda environment file

Run using `python pip2conda.py [NAME] [PATH]`` where NAME is the name of the
conda environment and PATH is the path to the pip requirements file.

Requires pyyaml (`conda install pyyaml`).
"""

import logging

import argparse
import subprocess
from collections import OrderedDict

import yaml


def get_info():
    """Reads the `conda info` command into a dict"""
    output = subprocess.check_output(['conda', 'info']) \
                       .decode('utf-8')
    info = {}
    for line in output.splitlines():
        if len(line) > 24 and line[24] == ':':
            # Check if line is key: value
            key, val = [s.strip() for s in line.split(':', 1)]
            info[key] = val
        elif line:
            # Adds value not on same line as key
            val = line.strip()
            try:
                info[key].append(val)
            except AttributeError as e:
                info[key] = [info[key], val]
    return info


def parse_build(version):
    """Parses the Python version to match the format used in `conda search`"""
    return 'py{}{}'.format(*version.split('.'))


def check_conda(package, build, version=None):
    """Checks if a given package exists in conda"""
    try:
        output = subprocess.check_output(['conda', 'search', package]) \
                           .decode('utf-8')
    except subprocess.CalledProcessError as e:
        pass
    else:
        for line in output.splitlines():
            try:
                pckg, ver, bld, channel = line.split()
            except ValueError as e:
                pass
            else:
                if (build is not None and bld.startswith(build)
                    and (not version or ver == version)):
                    logging.info('Found conda package for %s', package)
                    return True
    logging.info('Did not find conda package for %s', package)
    return False


def write_env_file(name, req_file, python=None):
    """Writes an environment file formatted for conda"""
    base_python = get_info()['python version']
    if python is not None and base_python[0] != python[0]:
        name += python[0]
    conda_dependencies, pip_dependencies = check_dependencies(req_file,
                                                              python=python)
    with open('{}.yaml'.format(name), 'w') as f:
        env = create_env_dict(name,
                              python,
                              conda_dependencies,
                              pip_dependencies)
        yaml.dump(env, f)
    print('Install with `conda env create -f {}.yaml`'.format(name))


def create_env_dict(name,
                    python=None,
                    conda_dependencies=None,
                    pip_dependencies=None):
    """Creates an environment dict formatted for conda"""
    if conda_dependencies is None:
        conda_dependencies = []
    env = {
        'name': name,
        'channels': ['default'],
        'dependencies': conda_dependencies
    }
    if pip_dependencies:
        env['dependencies'].append({'pip': pip_dependencies})
    return env


def check_dependencies(fp, python=None):
    """Identifies sources for dependencies based on `conda search`"""
    if python is None:
        python = get_info()['python version']
    build = parse_build(python)
    conda_dependencies = ['python={}'.format(python)]
    pip_dependencies = []
    with open(fp, 'r') as f:
        for line in f:
            try:
                package, version = line.strip().split('==')
            except ValueError as e:
                package = line.strip()
                version = ''
            if check_conda(package, build=build, version=version):
                versioned_package = '='.join([package, version]).rstrip('=')
                conda_dependencies.append(versioned_package)
            else:
                versioned_package = '=='.join([package, version]).rstrip('=')
                pip_dependencies.append(versioned_package)
    return conda_dependencies, pip_dependencies




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of conda environment')
    parser.add_argument('requirements_file', help='path to requirements file')
    parser.add_argument('--python', help='version of python', default='3.7')
    args = parser.parse_args()
    write_env_file(args.name, args.requirements_file, python=args.python)

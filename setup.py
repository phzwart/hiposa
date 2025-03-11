#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.17.0',
    'scipy>=1.3.0',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Petrus H. Zwart",
    author_email='phzwart@lbl.gov',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Hierarchical Poisson Sampling",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hiposa',
    name='hiposa',
    packages=find_packages(include=['hiposa']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/phzwart/hiposa',
    version='0.1.0',
    zip_safe=False,
)

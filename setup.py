#!/usr/bin/env python3

import os
import io
import setuptools

setuptools.setup(
    name='nematus',
    version='0.4',
    description='Neural machine translation tools on top of Tensorflow',
    long_description=io.open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md'),encoding='UTF-8').read(),
    license='BSD 3-clause',
    url='http://github.com/EdinburghNLP/nematus',
    install_requires=['numpy',
                      'tensorflow'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
    packages = ['nematus', 'nematus.metrics'],
)

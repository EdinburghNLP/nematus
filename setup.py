#!/usr/bin/env python

import os
import setuptools

setuptools.setup(
    name='nematus',
    version='0.2dev',
    description='Neural machine translation tools on top of Theano',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='BSD 3-clause',
    url='http://github.com/EdinburghNLP/nematus',
    install_requires=['numpy',
                      'Theano',
                      'bottle',
                      'bottle-log',
                      'paste'],
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano',],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
    packages=['nematus',
              'nematus.metrics',
              'nematus.server',
              'nematus.server.api'],
)

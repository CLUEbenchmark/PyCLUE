#!/usr/bin/python3

'''
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import call

with open('README.md','r') as f:
    long_description = f.read()

class Installation(install):
    def run(self):
        call(['pip install -r requirements.txt --no-clean'], shell=True)
        install.run(self)

setuptools.setup(
    name='PyCLUE',
    version='0.1.2',
    author='Liu Shaoweihua',
    author_email='liushaoweihua@126.com',
    maintainer='CLUE',
    maintainer_email='chineseGLUE@163.com',
    description='Python toolkit for Chinese Language Understanding Evaluation benchmark.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CLUEBenchmark/PyCLUE',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    install_requires=[
        'tensorflow-gpu==2.4.0', 
        'requests==2.23.0', 
        'numpy==1.18.2', 
        'matplotlib==3.2.1',
        'hnswlib==0.3.4'],
    cmdclass={'install': Installation})

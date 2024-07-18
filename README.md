[![pytest status](https://github.com/denisecailab/minian/workflows/pytest/badge.svg?branch=master)](https://github.com/DeniseCaiLab/minian/actions?query=workflow%3Apytest)
[![codecov](https://codecov.io/gh/DeniseCaiLab/minian/branch/master/graph/badge.svg)](https://codecov.io/gh/DeniseCaiLab/minian)
[![conda version](https://img.shields.io/conda/vn/conda-forge/minian.svg)](https://anaconda.org/conda-forge/minian)
[![documentation status](https://readthedocs.org/projects/minian/badge/?version=latest)](https://minian.readthedocs.io/en/latest/?badge=latest)

[![license](https://img.shields.io/github/license/denisecailab/minian)](https://www.gnu.org/licenses/gpl-3.0)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![commit style](https://img.shields.io/badge/commit_style-conventional-orange)](https://conventionalcommits.org)


# MiniAn

MiniAn is an analysis pipeline and visualization tool inspired by both [CaImAn](https://github.com/flatironinstitute/CaImAn) and [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Quick Start Guide

1. Create a new conda env: `conda create -y --name minian`
1. Activate the environment: `conda activate minian`
1. Install MiniAn: `conda install -y -c conda-forge minian`
1. Activate conda env: `conda activate minian`
1. Fix Panel installation:
    * On Windows: Copy panel files from `.conda/envs/minian/site-packages/` to `.conda/envs/minian/Lib/site-packages/`
    * On Linux: Install panel with pip in minian conda environment: `pip install panel=0.8.0`
1. Install pywiz libraries with conda: `conda install -n minian -c pyviz pyviz_comms=2.2.1`
1. Install shot-scraper CLI tool in minian conda environment: `pip install shot-scraper`. If this is the first time you will have to install additional dependencies via `shot-scraper install` and/or `playwright install`. Follow the directions on the command line.
1. Install the pipeline notebooks: `minian-install --notebooks`
1. Optional install the demo movies: `minian-install --demo`
1. Fire up jupyter: `jupyter notebook` and open the notebook "pipeline.ipynb"

# Documentation

MiniAn documentation is hosted on ReadtheDocs at:

https://minian.readthedocs.io/

# Contributing to MiniAn

We would love feedback and contribution from the community!
See [the contribution page](https://minian.readthedocs.io/en/latest/start_guide/contribute.html) for more detail!

# License

This project is licensed under GNU GPLv3.

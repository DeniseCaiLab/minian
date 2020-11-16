language: bash

os:
  # - windows
  - linux

branches:
  only:
  - master
  - dev
  - "/^pull.$"
  - "/^hotfix-.+$/"

before_install:
  - . scripts/travis/before_install_nix_win.sh
  # linux
  - |
    wget -qO- "https://github.com/crazy-max/travis-wait-enhanced/releases/download/v1.2.0/travis-wait-enhanced_1.2.0_linux_x86_64.tar.gz" | tar -zxvf - travis-wait-enhanced
    mv travis-wait-enhanced /home/travis/bin/
    travis-wait-enhanced --version
  # # windows
  # - |
  #   curl -LfsS -o /tmp/travis-wait-enhanced.zip "https://github.com/crazy-max/travis-wait-enhanced/releases/download/v1.2.0/travis-wait-enhanced_1.2.0_windows_x86_64.zip"
  #   7z x /tmp/travis-wait-enhanced.zip -y -o/usr/bin/ travis-wait-enhanced.exe -r
  #   travis-wait-enhanced --version

install:
  # Install miniconda
  - . scripts/travis/install_nix_win.sh

  - source $MINICONDA_PATH/etc/profile.d/conda.sh;
  - hash -r

  # Setting up conda env and install deps
  - conda env create -q -n minian -f environment.yml
  - conda activate minian
  - conda list
  - conda install -c conda-forge -y jupyterlab
  - jupyter labextension install @pyviz/jupyterlab_pyviz
  - conda env export
  - conda install -y pytest
  - conda install -y pytest-cov
  - conda install -c anaconda -y black
  - pip install opencv-python-headless

script:
  # The test/check scripts go here
  - travis_fold start "Black-check code quality"
  - black --check minian
  - travis_fold end "Black-check code quality"

  - travis_fold start "pytest"
  - travis_wait pytest -v --color=yes --cov=minian --pyargs minian
  - travis_fold end "pytest"

after_success:
  - bash <(curl -s https://codecov.io/bash)

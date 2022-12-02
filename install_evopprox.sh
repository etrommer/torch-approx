#!/bin/bash
set -x
cur_dir=$(pwd)
tmp_dir=$(mktemp -d)
cd $tmp_dir
git clone https://github.com/ehw-fit/evoapproxlib.git
cd evoapproxlib
git checkout v2022
pip install cython
python3 make_cython.py
cd cython
python3 setup.py build_ext
python3 setup.py install
cd $cur_dir
rm -rf $tmp_dir

#!/bin/bash

apt update
apt upgrade -y

apt install -y build-essential libssl-dev libffi-dev python3-dev zlib1g-dev gdb lcov libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma tk-dev uuid-dev xvfb dstat ifstat

cd ~
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz

tar -xf Python-3.9.16.tgz
cd Python-3.9.16

./configure --prefix=/usr --enable-optimizations
make -j 8
make install

# Install pip for Python 3.9
apt install -y python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py --prefix=/usr

pip3 --version
python3 --version

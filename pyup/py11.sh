#!/bin/bash

apt update
apt upgrade -y

apt install -y build-essential libssl-dev libffi-dev python3-dev zlib1g-dev gdb lcov libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma tk-dev uuid-dev xvfb dstat ifstat openssl

# cd ~
# wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz

# tar -xf Python-3.11.9.tgz
# cd Python-3.11.9

# ./configure --prefix=/usr --enable-optimizations
# make -j 8
# make install

# # Ensure pip is installed for Python 3.12
# python3.11 -m ensurepip --upgrade

# # Check versions
# pip3 --version
# python3 --version

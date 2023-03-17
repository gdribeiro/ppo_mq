#!/bin/bash

# python3 setup.py build_ext --inplace
# gcc -I/usr/include/python3.5m -c DummyMQ.c
# gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.5/PPOAgent.o -I/usr/include/python3.5m -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt

python3 setup.py build_ext --inplace
gcc -I/usr/include/python3.9 -c DummyMQ.c
# gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.9/PPOAgent.o -I/usr/include/python3.9/ -L/usr/lib/python3.9/config-3.9-x86_64-linux-gnu -L/usr/lib -lpython3.9 -lpthread -ldl  -lutil -lm -lcrypt -ltensorflow_framework
gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.9/PPOAgent.o -I/usr/include/python3.9/ -L/usr/lib/python3.9/config-3.9-x86_64-linux-gnu -L/usr/lib -lpython3.9 -lpthread -ldl  -lutil -lm -lcrypt


# LOCAL
# python3 setup.py build_ext --inplace
# gcc -I/usr/local/include/python3.9 -c DummyMQ.c
# gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.9/PPOAgent.o -I/usr/local/include/python3.9/ -L/usr/local/lib/python3.9/config-3.9-x86_64-linux-gnu -L/usr/lib -L/usr/local/lib -lpython3.9 -lpthread -ldl  -lutil -lm -lcrypt -ltensorflow_framework


#!/bin/bash

# python3 setup.py build_ext --inplace
# gcc -I/usr/include/python3.5m -c DummyMQ.c
# gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.5/PPOAgent.o -I/usr/include/python3.5m -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt

# python3 setup.py build_ext --inplace

# gcc -I/usr/include/python3.9 -c DummyMQ.c
# gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.9/PPOAgent.o -I/usr/include/python3.9/ -L/usr/lib/python3.9/config-3.9-x86_64-linux-gnu -L/usr/lib -lpython3.9 -lpthread -ldl  -lutil -lm -lcrypt

# echo "Moving PPOAgent.h ..."
# cp PPOAgent.h /home/gribeiro/ml_on_mq/ML_MODULE/include
# echo "Moving PPOAgent.c ..."
# cp PPOAgent.c /home/gribeiro/ml_on_mq/ML_MODULE/src

# for i in `cat /home/gribeiro/NFS_ERODS/Scripts/NODEFILES/nodefile `; do
#     echo "Compiling ml_on_mq"
# 	ssh root@$i "source /home/gribeiro/NFS_ERODS/Scripts/FUNCTIONS/install_ml_module.sh"
# 	ssh root@$i "source /home/gribeiro/NFS_ERODS/Scripts/APPLICATIONS/MQ/auto-deploy.sh"
# 	echo "Done! $i"
# done

## VER SE DA PRA EXECUTAR EM UMA SO MAQUINA... TIRA ROOT
# Pre verificar que ta compilando Print no meio!

nodefile="/home/gribeiro/NFS_ERODS/Scripts/NODEFILES/nodefile"
first_machine=$(head -n 1 "$nodefile")

echo "Logging into the first machine: $first_machine"
ssh $USER@"$first_machine" "source /home/gribeiro/ml_on_mq/ppo_mq/install_ppo.sh"

echo "PPO Module Installed!"
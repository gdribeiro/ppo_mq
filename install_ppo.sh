#!/bin/bash


nodefile="/home/gribeiro/NFS_ERODS/Scripts/NODEFILES/nodefile"
first_machine=$(head -n 1 "$nodefile")

cd "/home/gribeiro/ml_on_mq/ppo_mq"
python3 setup.py build_ext --inplace

echo "Moving PPOAgent.h to ML_MODULE/include..."
cp PPOAgent.h /home/gribeiro/ml_on_mq/ML_MODULE/include
echo "Moving PPOAgent.c ML_MODULE/src..."
cp PPOAgent.c /home/gribeiro/ml_on_mq/ML_MODULE/src


echo "Compiling ml_on_mq"
ssh root@$first_machine "source /home/gribeiro/NFS_ERODS/Scripts/FUNCTIONS/install_ml_module.sh"

echo "Compiling MQ module with ml library"
source "/home/gribeiro/NFS_ERODS/Scripts/APPLICATIONS/MQ/auto-deploy.sh"

echo "Logged out from the $first_machine"
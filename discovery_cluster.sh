#/!/bin/bash

#Load conf files
source ./FUNCTIONS/utils.sh
source ./CONFIG/conf.DATA

#Obtainining the current path
dir=$(pwd)

echo "Removing old files..."

rm $folder/* 

echo "creating file of nodes..."


#Creating node file through of G5K reservation
discovery-cluster $folder/$filename $dir

update-local-key  $folder/$filename $dir

for i in `cat NODEFILES/nodefile `; do

	# ssh root@$i "source /home/gribeiro/ml_on_mq/ppo_mq/pyup/py9.sh"

	ssh root$i "pip3 install tf-agents"
	# ssh $i "source /home/gribeiro//home/gribeiro/.bashrc"
	# ssh root$i "pip3 install -r /home/gribeiro/ml_on_mq/ppo_mq/requirements.txt"
	# ssh $i "pip3 install --upgrade tensorflow tf-agents"
	# ssh root@$i "source /home/gribeiro//home/gribeiro/.bashrc"
	# ssh root@$i "pip3 install -r /home/gribeiro/ml_on_mq/ppo_mq/requirements.txt"
#
	# ssh root@$i "apt -y update"
	# ssh root@$i "apt -y upgrade"
	# ssh root@$i "source /home/gribeiro/NFS_ERODS/Scripts/FUNCTIONS/install_ml_module.sh"
	# ssh root@$i "pip3 install matplotlib"
	# ssh root@$i "apt -y install dstat ifstat"
	# ssh root@$i "apt -y install python3 python3-pip python3-dev python3-venv"
	# ssh root@$i "pip3 install --upgrade pip"
	# ssh root@$i "pip3 install --upgrade tensorflow"
	# ssh root@$i "pip3 install --upgrade dm-tree==0.1.5 tf-agents==0.6.0"
	# ssh root@$i "pip3 install --upgrade dm-tree tf-agents"
	# ssh root@$i "pip3 install -r /home/gribeiro/ml_on_mq/SAQN/debian9_requirements.txt"


done


echo "DONE!"

cat $folder/$filename







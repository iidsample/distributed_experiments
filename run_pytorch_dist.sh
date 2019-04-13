NODE_RANK="$1"
NNODE="$2"
MASTER_IP="$3"
SRC_DIR=${HOME}/distributed_experiments

NCCL_SOCKET_IFNAME=ens3 /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python ${SRC_DIR}/imagenet_example.py \
-a resnet50 \
--lr 0.01 \
--dist-url=${MASTER_IP} \
--dist-backend='nccl' \
--multiprocessing-distributed \
--world-size ${NNODE} \
--rank ${NODE_RANK} \
~/data > out_node_${NODE_RANK} 2>&1
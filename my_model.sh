# python ./my_model/cls_adapter.py
python ./my_model/cls_frqattn.py
# python ./my_model/cls_double.py
# python ./my_model/loc_unet.py
# python ./my_model/loc_fc.py
# python ./my_model/loc_gene.py

#!/bin/bash

# # Distributed training configuration
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# #25678 $(shuf -i 20001-29999 -n 1)
# MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# NNODES=${WORLD_SIZE:-1}

# # Set visible devices ,2,3,4,5
# export CUDA_VISIBLE_DEVICES="0,1"
# NPROC_PER_NODE=2

# # Launch training wandb
# torchrun --nproc_per_node=${NPROC_PER_NODE} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
#          ./my_model/cls_adapter.py 

# #  sft_vit_cls.py

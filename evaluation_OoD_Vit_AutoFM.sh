#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution_OoD_Vit_AutoFM.py --gp \
--change_qk --relative_position --dist-eval --search-space ./experiments/supernet/supernet-S.yaml --supernet 'small' \
--config-dataset './configs/ImageNet-O.yaml'

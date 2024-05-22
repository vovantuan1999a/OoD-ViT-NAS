#!/bin/bash
python compute_trainning_free_for_Vit_OoD.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --search-space './experiments/supernet/supernet-B.yaml' --output_dir './OUTPUT/trainning_free_nas' --supernet 'small'



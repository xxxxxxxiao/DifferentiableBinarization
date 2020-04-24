#!/usr/bin/env bash
python ./faster_rcnn/train_net.py \
--gpu 1 \
--weights ./data/pretrain_model/Resnet50.npy \
--imdb voc_2007_trainval \
--iters 160000 \
--cfg ./experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--network Resnet50_train \
--restore 0

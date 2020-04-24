#!/usr/bin/env bash
python ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/Resnet101.npy \
--imdb voc_2007_trainval \
--iters 160000 \
--cfg ./experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--network Resnet101_train \
--restore 0

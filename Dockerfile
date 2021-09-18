#基于的基础镜像
FROM tensorflow/tensorflow:1.15.5-gpu-py3

COPY bert-qa-master /bert-qa-master
COPY ccks /ccks
COPY pretrain_model /pretrain_model

WORKDIR /bert-qa-master


# 安装支持
RUN ["chmod", "+x", "train.sh"]

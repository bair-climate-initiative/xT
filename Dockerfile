ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV PYTHONPATH=.

RUN pip install albumentations==1.0.0 timm==0.4.12 tensorboardx pandas madgrad
RUN apt update
RUN apt install wget mc -y
# Setting the working directory
WORKDIR /workspace

RUN mkdir -p logs
RUN mkdir -p weights

RUN wget -O weights/val_only_TimmUnet_nfnet_l0_99_xview https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_nfnet_l0_99_xview
RUN wget -O weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_99_last https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_99_last
RUN wget -O weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_77_xview https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_77_xview
RUN wget -O weights/val_only_TimmUnet_tf_efficientnetv2_m_in21k_99_last https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_tf_efficientnetv2_m_in21k_99_last
RUN wget -O weights/val_only_TimmUnet_tf_efficientnet_b7_ns_77_xview https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_tf_efficientnet_b7_ns_77_xview
RUN wget -O weights/val_only_TimmUnet_resnet34_77_xview https://github.com/selimsef/xview3_solution/releases/download/weights/val_only_TimmUnet_resnet34_77_xview

# Copying the required codebase
COPY . /workspace

RUN chmod 777 run_inference.sh

ENTRYPOINT [ "/workspace/run_inference.sh" ]


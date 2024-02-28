# 🌐 xT: Nested Tokenization for Larger Context in Large Images
![xT](assets/xt.png "xT")

> **xT: Nested Tokenization for Larger Context in Large Images**\
> Ritwik Gupta*, Shufan Li*, Tyler Zhu*, Jitendra Malik, Trevor Darrell, Karttikeya Mangalam\
> Paper: https://arxiv.org/abs/something

## About
_xT_ enables you to model large images, end-to-end, on contemporary, memory-limited GPUs. It is a simple framework for vision transformers which effectively aggregates global context with local details.

## Installation
* `conda env create -f environment.yml`

The code has been tested on Linux on NVIDIA A100 GPUs with PyTorch 2+. We use custom CUDA kernels as implemented by the `Mamba` and `OpenAI Triton` projects. Therefore, modifications may be required to use this repository on other operating systems or GPUs.

## Pretrained Models

Weights and configs for our experiments are available on [Hugging Face](https://huggingface.co/bair-climate-initiative/swin-xt/tree/main).

|Name                          |Top1-ACC|Top5-ACC|
|------------------------------|-------:|-------:|
|Swin-L-XL (1 layer) 512/256   |   69.25|   91.67|
|Swin-L-XL 512/256             |   72.42|   94.48|
|Swin-L-Mamba 512/256          |   73.36|   94.13|
|Swin-L-XL 256/256             |   67.84|   92.25|
|Swin-L                        |   68.78|   90.96|
|------------------------------|--------|--------|
|Swin-B-XL (1 layer) 512/256   |   65.49|   89.08|
|Swin-B-XL 512/256             |   64.08|   90.73|
|Swin-B-Mamba 512/256          |   63.73|   89.91|
|Swin-B-XL 256/256             |   55.52|   85.80|
|Swin-B                        |   58.57|   84.74|
|------------------------------|--------|--------|
|Swin-S-XL (1 layer) 512/256   |    63.5|   88.26|
|Swin-S-XL 512/256             |   63.62|   88.03|
|Swin-S-XL 256/256             |   57.04|   83.22|
|Swin-S                        |   58.45|   85.21|
|------------------------------|--------|--------|
|Swin-T-XL (1 layer) 512/256   |   59.27|   86.38|
|Swin-T-XL 512/256             |   60.56|   85.09|
|Swin-T-Mamba 512/256          |   61.97|   86.27|
|Swin-T-XL 256/256             |   52.93|   80.87|
|Swin-T                        |   53.76|   82.86|
|------------------------------|--------|--------|
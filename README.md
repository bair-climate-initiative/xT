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

## Training
Training can be launched through
```./run_submit.sh <num GPUs> <port number> config=<path to config>```

We also provide [SubmitIt](https://github.com/facebookincubator/submitit) scripts in `launch_scripts` to submit training jobs on Slurm clusters.

## Pretrained Models

Weights and configs for our experiments are available on [Hugging Face](https://huggingface.co/bair-climate-initiative/swin-xt/tree/main).

|Name                     |Top1-ACC|
|-------------------------|-------:|
|[Swin-T](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-t/weights/swin-t-nonxl-256-top1.ckpt)                   |   53.76|
|[Swin-T \<xT> XL 256/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-t/weights/swin-t-xl-256-256-top1.ckpt)        |   52.93|
|[Swin-T \<xT> XL 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-t/weights/swin-t-xl-512-256-top1.ckpt)        |   60.56|
|[Swin-T \<xT> Mamba 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-t/weights/swin-t-mamba-512-256-top1.ckpt)     |   **61.97**|
|-------------------------|--------|
|[Swin-S](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-s/weights/swin-s-nonxl-256-top1.ckpt)                   |   58.45|
|[Swin-S \<xT> XL 256/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-s/weights/swin-s-xl-256-256-top1.ckpt)        |   57.04|
|[Swin-S \<xT> XL 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-s/weights/swin-s-xl-512-256-top1.ckpt)        |   **63.62**|
|-------------------------|--------|
|[Swin-B](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-b/weights/swin-b-nonxl-256-top1.ckpt)                   |   58.57|
|[Swin-B \<xT> XL 256/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-b/weights/swin-b-xl-256-256-top1.ckpt)        |   55.52|
|[Swin-B \<xT> XL 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-b/weights/swin-b-xl-512-256-top1.ckpt)        |   **64.08**|
|[Swin-B \<xT> Mamba 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-b/weights/swin-b-mamba-512-256-top1.ckpt)     |   63.73|
|-------------------------|--------|
|[Swin-L](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-l/weights/swin-l-nonxl-256-top1.ckpt)                   |   68.78|
|[Swin-L \<xT> XL 256/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-l/weights/swin-l-xl-256-256-top1.ckpt)        |   67.84|
|[Swin-L \<xT> XL 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-l/weights/swin-l-xl-512-256-top1.ckpt)        |   72.42|
|[Swin-L \<xT> Mamba 512/256](https://huggingface.co/bair-climate-initiative/swin-xt/blob/main/swin-l/weights/swin-l-mamba-512-256-top1.ckpt)     |   **73.36**|

## Citation

```
@article{xTLargeImageModeling,
  title={xT: Nested Tokenization for Larger Context in Large Images},
  author={Gupta, Ritwik and Li, Shufan and Zhu, Tyler and Malik, Jitendra and Darrell, Trevor and Mangalam, Karttikeya},
  journal={arXiv preprint arXiv:tbd},
  year={2024}
}
```
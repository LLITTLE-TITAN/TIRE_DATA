---
## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>
If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation.

1. Install dependent packages

    pip install -r requirements.txt
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio===2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html


## :zap: Quick Inference

Download pre-trained models: [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P experiments/pretrained_models
```

**Inference!**

```Write your OpenAI API key in 'main.py'


```bash
python main.py -i inputs/whole_imgs -o results -v 1.4 -s 2
```

```console
Usage: python main.py -i inputs/whole_imgs -o results -v 1.4 -s 2 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3
  -s upscale           The final upsampling scale of the image. Default: 2
  -bg_upsampler        background upsampler. Default: realesrgan
  -bg_tile             Tile size for background sampler, 0 for no tile during testing. Default: 400
  -suffix              Suffix of the restored faces
  -only_center_face    Only restore the center face
  -aligned             Input are aligned faces
  -ext                 Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

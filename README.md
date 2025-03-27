# ![icon](assets/icon_small.png) FaceXLib

[![PyPI](https://img.shields.io/pypi/v/facexlib)](https://pypi.org/project/facexlib/)
[![download](https://img.shields.io/github/downloads/xinntao/facexlib/total.svg)](https://github.com/xinntao/facexlib/releases)
[![Open issue](https://img.shields.io/github/issues/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/facexlib.svg)](https://github.com/xinntao/facexlib/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/facexlib/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/facexlib/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/publish-pip.yml)

[English](README.md) **|** [简体中文](README_CN.md)

---

**facexlib** aims at providing ready-to-use **face-related** functions based on current SOTA open-source methods. <br>
Only PyTorch reference codes are available. For training or fine-tuning, please refer to their original repositories listed below. <br>
Note that we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use.

We have supported the detection and recognition models `antelopev2` and `buffalo_l` for [Insighface](https://github.com/deepinsight/insightface) (identical models with ignorable differences), without the need to install any onnx runtime. For users who are unable or unwilling to install the onnx runtime due to issues with glib, python, or CUDA versions, we suggest using our repository. See [migration tutorial](tutorial/migrate_from_insightface.ipynb) for migrating from Insightface to facexlib. :heart_eyes:

If facexlib is helpful in your projects, please help to :star: this repo. Thanks:blush: <br>

---

## :sparkles: Functions

| Function | Sources  | Original LICENSE |
| :--- | :---:        |     :---:      |
| [Detection](facexlib/detection/README.md) | [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MIT |
| [Alignment](facexlib/alignment/README.md) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [Recognition](facexlib/recognition/README.md) | [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) and [InsightFace (official)](https://github.com/deepinsight/insightface) | MIT |
| [Parsing](facexlib/parsing/README.md) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [Matting](facexlib/matting/README.md) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [Headpose](facexlib/headpose/README.md) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [Tracking](facexlib/tracking/README.md) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [Super Resolution](facexlib/resolution/README.md) | [HAT](https://github.com/XPixelGroup/HAT) | - |
| [Gender & Age](facexlib/genderage/README.md) | [MiVOLO](https://github.com/WildChlamydia/MiVOLO) | - |
| [Assessment](facexlib/assessment/README.md) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [Utils](facexlib/utils/README.md) | Face Restoration Helper | - |

> *TODO*: Finetune super resolution model for face tasks.

## :eyes: Demo and Tutorials

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniforge (mamba)](https://github.com/conda-forge/miniforge))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

```bash
pip install git+https://github.com/bhcao/facexlib.git
```

### Pre-trained models

If your network is not stable, you can download in advance (may with other download tools), and put them in the folder: `PACKAGE_ROOT_PATH/facexlib/weights`. `PACKAGE_ROOT_PATH` defaults to the installation path of `facexlib`, you can also change it by passing the argument `model_rootpath` during the initialization of each model.

If your network is stable, you can set the environment variable `BLOCK_DOWNLOADING` to `false` to enable automatically downloading of pre-trained models at the first inference.

## :scroll: License and Acknowledgement

This project is released under the MIT license. <br>

## :e-mail: Contact

If you have any question, open an issue or email `xintao.wang@outlook.com`.

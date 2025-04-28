# ![icon](assets/icon_small.png) FaceXLib Plus

[![PyPI](https://img.shields.io/pypi/v/facexlib)](https://pypi.org/project/facexlib/)
[![download](https://img.shields.io/github/downloads/xinntao/facexlib/total.svg)](https://github.com/xinntao/facexlib/releases)
[![Open issue](https://img.shields.io/github/issues/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/facexlib.svg)](https://github.com/xinntao/facexlib/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/facexlib/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/facexlib/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/publish-pip.yml)

[English](README.md) **|** [简体中文](README_CN.md)

---

**facexlib plus** is a enhanced version of [FaceXLib](https://github.com/xinntao/facexlib), which provides more functions while maintaining compatibility with the top level API.

**facexlib** aims at providing ready-to-use **face-related** functions based on current SOTA open-source methods. <br>
Only PyTorch reference codes are available. For training or fine-tuning, please refer to their original repositories listed below. <br>
Note that we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use.

If facexlib is helpful in your projects, please help to :star: this repo. Thanks:blush: <br>

---

## :sparkles: Functions

| Function | Sources  | Original LICENSE |
| :--- | :---:        |     :---:      |
| [Detection](inference/inference_detection.py) | [Retinaface](https://github.com/biubug6/Pytorch_Retinaface) / [YOLO](https://github.com/ultralytics/ultralytics) | MIT / AGPL 3.0 |
| [Alignment](inference/inference_alignment.py) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [Recognition](inference/inference_recognition.py) | [InsightFace](https://github.com/deepinsight/insightface) / [FaceNet](https://github.com/davidsandberg/facenet) | MIT / MIT |
| [Parsing](inference/inference_parsing.py) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [Matting](inference/inference_matting.py) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [Headpose](inference/inference_headpose.py) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [Tracking](inference/inference_tracking.py) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [Super Resolution](inference/inference_super_resolution.py) | [SwinIR](https://github.com/JingyunLiang/SwinIR) / [DRCT](https://github.com/ming053l/drct) | Apache 2.0 / MIT |
| [Anti-Spoofing](inference/inference_anti_spoofing.py) | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Apache 2.0 |
| [Expression](inference/inference_expression.py) | [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | Apache 2.0 |
| [Gender & Age](inference/inference_gender_age.py) | [MiVOLO](https://github.com/WildChlamydia/MiVOLO) | - |
| [Assessment](inference/inference_hyperiqa.py) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [Utils](inference/inference_crop_standard_faces.py) | Face Restoration Helper | - |

## :eyes: Demo and Tutorials

**Migrate from insightface.** We have supported the detection and recognition models `antelopev2` and `buffalo_l` for [Insighface](https://github.com/deepinsight/insightface) (identical models with a few differences), without the need to install any onnx runtime. For users who are unable to install the onnx runtime due to issues with glib, python, or CUDA versions, or who need to calculate losses on the results of the generated model, we suggest using our repository. See [migration tutorial](tutorial/migrate_from_insightface.ipynb) for migrating from Insightface to facexlib. :heart_eyes:

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniforge (mamba)](https://github.com/conda-forge/miniforge))
- [PyTorch >= 1.10](https://pytorch.org/) (Recommend NOT using torch 1.12! It would cause abnormal performance.)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

```bash
pip install git+https://github.com/bhcao/facexlib.git
```

### Pre-trained models

If your network is not stable, you can download in advance (may with other download tools), and put them in the folder: `PACKAGE_ROOT_PATH/facexlib/weights`. `PACKAGE_ROOT_PATH` defaults to the installation path of `facexlib`, you can also change it by passing the argument `model_rootpath` during the initialization of each model.

If your network is stable, you can set the argument `auto_download` of `build_model` function to `True` to enable automatically downloading of pre-trained models at the first inference.

## :scroll: License and Acknowledgement

This project is released under the MIT license. <br>

## :e-mail: Contact

If you have any question, open an issue or email `xintao.wang@outlook.com`.

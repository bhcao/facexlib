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

**facexlib plus** 是 [FaceXLib](https://github.com/xinntao/facexlib) 的增强版本，在保持顶级 API 的兼容性的同时提供更多功能。

**facexlib** 的目标是提供基于当前 SOTA 开源方法的现成的 **人脸相关** 功能。<br>
仅提供 PyTorch 参考代码。有关训练或微调，请参考下面列出的原始代码库。<br>
请注意，我们仅提供这些算法的集合。对于您的预期用途，请参考它们的原始许可证。

如果 facexlib 对您的项目有所帮助，请帮助 :star: 这个仓库。谢谢 :blush:<br>

---

## :sparkles: 模块

| 模块 | 源代码  | 原始许可证 |
| :--- | :---:        |     :---:      |
| [检测](inference/inference_detection.py) | [Retinaface](https://github.com/biubug6/Pytorch_Retinaface) / [YOLO](https://github.com/ultralytics/ultralytics) | MIT / AGPL 3.0 |
| [对齐](inference/inference_alignment.py) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [识别](inference/inference_recognition.py) | [InsightFace](https://github.com/deepinsight/insightface) / [FaceNet](https://github.com/davidsandberg/facenet) | MIT / MIT |
| [解析](inference/inference_parsing.py) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [抠图](inference/inference_matting.py) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [头部姿态](inference/inference_headpose.py) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [跟踪](inference/inference_tracking.py) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [超分](inference/inference_super_resolution.py) | [SwinIR](https://github.com/JingyunLiang/SwinIR) / [DRCT](https://github.com/ming053l/drct) | Apache 2.0 / MIT |
| [活体检测](inference/inference_anti_spoofing.py) | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Apache 2.0 |
| [性别与年龄](inference/inference_gender_age.py) | [MiVOLO](https://github.com/WildChlamydia/MiVOLO) | - |
| [评估](inference/inference_hyperiqa.py) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [工具](inference/inference_crop_standard_faces.py) | Face Restoration Helper | - |

## :eyes: 演示和教程

**从 Insightface 迁移。** 我们支持了 [Insighface](https://github.com/deepinsight/insightface) 的检测和识别模型 `antelopev2` 以及 `buffalo_l`（相同模型，有一定误差），无需安装任何 onnx 运行时。对于由于 glib、python 或 CUDA 版本的问题而无法安装 onnx 运行时的用户，或者需要对生成模型的结果计算损失的用户，我们建议使用我们的存储库。从 Insightface 迁移到 facexlib，请见 [迁移教程](tutorial/migrate_from_insightface.ipynb)。 :heart_eyes:

## :wrench: 依赖和安装

- Python >= 3.7 （推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniforge (mamba)](https://github.com/conda-forge/miniforge)）
- [PyTorch >= 1.10](https://pytorch.org/) （建议不要使用 torch 1.12！因为它会导致性能异常）
- 可选：NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### 安装

```bash
pip install git+https://github.com/bhcao/facexlib.git
```

### 预训练模型

如果您网络不稳定，可以提前下载（可以使用其他下载工具），并将它们放在文件夹 `PACKAGE_ROOT_PATH/facexlib/weights` 中。`PACKAGE_ROOT_PATH` 默认为 `facexlib` 的安装路径，你也可以在每个模型初始化时通过传递参数 `model_rootpath` 来改变它。

如果您的网络稳定，您可以将 `build_model` 函数的 `auto_download` 参数设置为 `True`，以在第一次推理时自动下载预训练模型。

## :scroll: 许可证和致谢

本项目在 MIT 许可证下发布。<br>

## :e-mail: 联系

如果您有任何问题，请创建一个 issue 或发送邮件至 `xintao.wang@outlook.com`。
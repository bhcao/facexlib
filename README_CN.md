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

**facexlib** 的目标是提供基于当前 SOTA 开源方法的现成的 **人脸相关** 功能。<br>
仅提供 PyTorch 参考代码。有关训练或微调，请参考下面列出的原始代码库。<br>
请注意，我们仅提供这些算法的集合。对于您的预期用途，请参考它们的原始许可证。

我们支持了 [Insighface](https://github.com/deepinsight/insightface) 的检测和识别模型 `antelopev2` 以及 `buffalo_l`（相同模型，误差可忽略），无需安装任何 onnx 运行时。对于由于 glib、python 或 CUDA 版本的问题而无法或不愿意安装 onnx 运行时的用户，我们建议使用我们的存储库。欢迎 :heart_eyes:

如果 facexlib 对您的项目有所帮助，请帮助 :star: 这个仓库。谢谢 :blush:<br>

---

## :sparkles: 模块

| 模块 | 源代码  | 原始许可证 |
| :--- | :---:        |     :---:      |
| [检测](facexlib/detection/README.md) | [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MIT |
| [对齐](facexlib/alignment/README.md) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [识别](facexlib/recognition/README.md) | [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) 以及 [InsightFace (official)](https://github.com/deepinsight/insightface) | MIT |
| [解析](facexlib/parsing/README.md) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [抠图](facexlib/matting/README.md) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [头部姿态](facexlib/headpose/README.md) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [跟踪](facexlib/tracking/README.md) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [超分](facexlib/resolution/README.md) | [HAT](https://github.com/XPixelGroup/HAT) | - |
| [性别与年龄](facexlib/genderage/README.md) | [MiVOLO](https://github.com/WildChlamydia/MiVOLO) | - |
| [评估](facexlib/assessment/README.md) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [工具](facexlib/utils/README.md) | Face Restoration Helper | - |

> *TODO*: 微调超分模型以适应人脸任务。

## :eyes: 演示和教程

## :wrench: 依赖和安装

- Python >= 3.7 （推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniforge (mamba)](https://github.com/conda-forge/miniforge)）
- [PyTorch >= 1.7](https://pytorch.org/)
- 可选：NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### 安装

```bash
pip install git+https://github.com/bhcao/facexlib.git
```

### 预训练模型

在第一次推理时会**自动**下载预训练模型。<br> 如果您网络不稳定，可以提前下载（可以使用其他下载工具），并将它们放在文件夹 `PACKAGE_ROOT_PATH/facexlib/weights` 中。

## :scroll: 许可证和致谢

本项目在 MIT 许可证下发布。<br>

## :e-mail: 联系

如果您有任何问题，请创建一个 issue 或发送邮件至 `xintao.wang@outlook.com`。
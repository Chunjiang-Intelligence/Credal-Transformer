# Credal Transformer

[![License: arXiv.org](https://img.shields.io/badge/License-arXiv.org-b31b1b.svg)](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

这是一个基于 PyTorch 的项目，旨在实现论文 **"Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models"** (arXiv:2510.12137) 中提出的新型 Transformer 架构。

## 项目简介

大型语言模型 (LLM) 的一个核心挑战是“幻觉”——即模型会以高置信度生成事实错误的内容。该论文认为，这个问题部分源于 Transformer 架构本身，特别是注意力机制中的 Softmax 函数，它会强制模型在不同选项间做出明确选择，从而产生“人为确定性” (Artificial Certainty)。

为了应对这一挑战，**Credal Transformer** 应运而生。它用一种新颖的 **信任注意力机制 (Credal Attention Mechanism, CAM)** 取代了标准的注意力机制。CAM 基于证据理论，它不产生单一的注意力分布，而是计算一个“信任集”，从而能够直接、可微地量化模型在每个注意力头中的 **认知不确定性 (Epistemic Uncertainty)**。

本项目提供了 `CredalAttention` 和 `CredalTransformerEncoderLayer` 的完整 PyTorch 实现，旨在为研究和应用这种能够感知自身不确定性的模型提供一个坚实的基础。

## 快速开始

项目中的 `main.py` 文件提供了一个如何构建和运行 `CredalTransformerEncoder` 的简单示例。

## 如何使用

你可以轻松地将 `CredalAttention` 或 `CredalTransformerEncoderLayer` 集成到你自己的模型中。

### 替换 Transformer 层

如果你有一个基于标准 `TransformerEncoderLayer` 的模型，你可以直接将其替换为 `CredalTransformerEncoderLayer`。

```python
from credal_transformer.layers.transformer import CredalTransformerEncoderLayer
from credal_transformer.models.encoder import CredalTransformerEncoder

# 1. 定义一个 Credal Transformer 编码器层
encoder_layer = CredalTransformerEncoderLayer(
    d_model=512, 
    nhead=8, 
    dim_feedforward=2048, 
    dropout=0.1
)

# 2. 堆叠多层构建完整的编码器
credal_encoder = CredalTransformerEncoder(encoder_layer, num_layers=6)

# 3. 使用模型
# `uncertainties` 是一个包含了每一层不确定性张量的列表
src = torch.rand(10, 32, 512) # (SeqLen, Batch, Dim)
output, uncertainties = credal_encoder(src)
```

### 利用不确定性信号

获取到的 `uncertainties` 张量是该架构的核心优势。你可以在模型的训练和推理过程中利用它：

- **作为正则化项**: 在损失函数中加入不确定性项，鼓励模型在缺乏足够证据时产生更高的不确定性。
- **指导解码过程**: 在生成任务中，如果某一步的不确定性过高，可以调整解码策略，例如引入外部知识或生成更保守的文本。
- **可靠性评估**: 在分类或问答任务中，设置一个不确定性阈值。当模型的不确定性高于该阈值时，系统可以拒绝回答，从而避免给出不可靠的答案。

## 未来工作

- [ ] 在标准的自然语言处理基准测试（如 GLUE）上进行评估。
- [ ] 探索不确定性信号在指导文本生成解码过程中的应用。
- [ ] 将 Credal Attention 机制扩展到 Transformer 解码器中。
- [ ] 研究不同方法来聚合和利用多层、多头的不确定性信号。

## 引用

如果你在研究中使用了本项目的代码或思想，请引用原始论文：

```
@article{ji2025credal,
  title={Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models},
  author={Ji, Shihao and Song, Zihui and Huang, Jiajie},
  journal={arXiv preprint arXiv:2510.12137},
  year={2025},
  eprint={2510.12137},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

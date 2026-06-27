# XSliceVul: Enhancing Trustworthy Code Governance via LLM-Driven Bi-modal Vulnerability Detection with Residual Cross-Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-3820/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0-orange.svg)](https://pytorch.org/)

This repository contains the official implementation for **XSliceVul**, a knowledge-augmented, bi-modal framework designed for precise, automated vulnerability detection. By integrating fine-grained structural topology from Code Property Graphs (CPG) with high-level semantic risk insights from Large Language Models (LLMs), XSliceVul leverages a novel **Residual Cross-Attention Fusion (RCAF)** paradigm to bridge the gap between physical code structure and cognitive semantics. Empirical evaluations on *ReVeal*, *Devign*, and *TRVD* benchmarks demonstrate that XSliceVul significantly outpaces six state-of-the-art baselines, achieving substantial performance gains with relative improvements of **15.41% in accuracy** and **35.54% in F1-score**.

---


## 📁 Repository Structure

```
XSliceVul/
├── CPGSlice/
│   ├── cpg_to_context.py    # Standardizes Code Property Graphs (CPG) into JSONL format
│   ├── llm_slicer.py        # Graph-guided LLM code slicing and refinement engine
│   └── stats_analyzer.py
├── Normalization/
│   ├── __init__.py
│   ├── normalization.py  
│   └── lang_processors/
├── model/
│   ├── comment.py
│   ├── model.py             # Deep cross-attention neural network architecture
│   ├── crossattention.py    # Feature fusion module based on Residual Cross-Attention (RCAF)
│   └── run.py               # Model training, checkpointing, and evaluation script
├── .gitignore
├── environment.yml
├── LICENSE
└── README.md
```
---


## 🔧 Environment Setup
Prerequisites
NVIDIA GPU with CUDA support

Ollama running locally (configured with qwen3-coder:30b or equivalent)

Installation
1. Create the environment and install all precise dependencies via Conda:
```
conda env create -f environment.yml
conda activate XSliceVul
```
2. Download Pre-trained Models:
Manually download the following two pre-trained models from Hugging Face and place their weights/files directly into the `model/` directory:
   * [codesage-small-v2](https://huggingface.co/codesage/codesage-small-v2)
   * [roberta-base](https://huggingface.co/FacebookAI/roberta-base)
---


## 📊 Dataset Availability
The preprocessed datasets (including graph contexts and LLM-refined code slices for ReVeal, Devign, and TRVD) are compressed and archived for public verification.

How to Download: Download the dataset artifact dataset.zip directly from our latest GitHub Releases.

---


## 🏃 Quick Start

Follow these sequential steps to process the data and run the pipeline:

### 1. Source Code Normalization
Clean the raw source code gadgets and normalize syntax across languages:
```
python Normalization/normalization.py
```
### 2. CPG Extraction & Context Transformation
Generate the Code Property Graphs (CPG) and standardize the structures into contextual JSONL files:
```
python CPGSlice/cpg_to_context.py
```
### 3. LLM-Assisted Code Slicing
Leverage the graph contexts to guide local LLM refinement and slice generation:
```
python CPGSlice/llm_slicer.py
```
### 4. Code Comment & Explanation Generation
Generate high-level, vulnerability-oriented risk explanations from the sliced code:
```
python model/comment.py
```
### 5. Model Training & Evaluation
Train and evaluate the RCAF feature fusion module:
```
python model/run.py
```
---


## 🤝 References

Parts of the data preprocessing or normalization code in this repository reference the FuSEVul framework. Please cite the following paper if you use this codebase:

```latex
@article{TIAN2026103450,
  title   = {Enhancing vulnerability detection by fusing code semantic features with LLM-generated explanations},
  author  = {Tian, Zhenzhou and Li, Minghao and Sun, Jiaze and Chen, Yanping and Chen, Lingwei},
  journal = {Information Fusion},
  volume  = {125},
  pages   = {103450},
  year    = {2026},
  issn    = {1566-2535},
  doi     = {[https://doi.org/10.1016/j.inffus.2025.103450](https://doi.org/10.1016/j.inffus.2025.103450)}
}
```
---
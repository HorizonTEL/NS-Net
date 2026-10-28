<div align="center">
 <br>
<h1>NS-Net: Decoupling CLIP Semantic Information through NULL-Space for Generalizable AI-Generated Image Detection
</h1>
 
[Jiazhen Yan](https://scholar.google.com/citations?user=QkURh8EAAAAJ&hl=zh-CN)<sup>1*</sup>, [Fan Wang](https://scholar.google.com/citations?user=zT1Ad0gAAAAJ&hl=zh-CN)<sup>1*</sup>, [Ziwen He](https://scholar.google.com/citations?user=PjkDK9cAAAAJ&hl=zh-CN)<sup>1</sup>, [Ziqiang Li](https://scholar.google.com/citations?user=mj5a8WgAAAAJ&hl=zh-CN)<sup>1‡</sup>, [Zhangjie Fu](https://scholar.google.com/citations?user=fO9NmagAAAAJ&hl=zh-CN)<sup>1</sup>


<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>*</sup></span>Co-first author</p>
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>School of Computer Science, Nanjing University of Information Science and Technology
<!-- <p align="center">
  <a href='https://horizontel.github.io/DGS-Net/'>
    <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://arxiv.org/abs/2511.13108'>
    <img src='https://img.shields.io/badge/Arxiv-2511.13108-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2511.13108'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
</p> -->
</div>
<img width="10243" height="4080" alt="backbone" src="https://github.com/user-attachments/assets/514b8261-10aa-491a-9d87-eabe4b01e7ee" />



## 📰 News
* [2026-05-01]🎉🎉🎉 NS-Net is accepted by IEEE Transactions on Multimedia
## 🔥 Todo List
- [x] **Inference code**
- [x] **Pretrained models**
- [x] **Training code**

## 🚀 Quick Start
### 1. Installation
```
conda create -n NS-Net -y python=3.10

conda activate NS-Net
pip3 install torch torchvision
pip install -r requirements.txt 
```
### 2.Getting datasets
| Datasets          |    Paper                                                                                                             |    Url    |
|:------:           |:---------:                                                                                                             |:---------:|
| UniversalFakeDetect| Towards Universal Fake Image Detectors that Generalize Across Generative Models (CVPR 2023)                            | [Google Drive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-) |
| AIGCDetectBench   | PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection                                         | [ModelScope](https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/tree/master/AIGCDetectionBenchMark) |
| AIGIBench         | Is Artificial Intelligence Generated Image Detection a Solved Problem? (NeurIPS 2025)                                  | [Huggingface](https://huggingface.co/datasets/HorizonTEL/AIGIBench)/[Baidu Netdisk](https://pan.baidu.com/s/1XTwfXlfqkGxAwYLxXuZbfA?pwd=sm6v) |
| ARForensics       | D^3QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection (ICCV 2025)| [Huggingface](https://huggingface.co/HorizonTEL/AIGIBench/tree/main) |
### 3.Download Weights
Place them under ```checkpoints/```:
### 4.Inference
Of course, you need to change [DetectionTests] in test.py when testing.

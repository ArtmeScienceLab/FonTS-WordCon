<h2 align="center"> FonTS: Text Rendering with Typography and Style Controls [ICCV 2025]
</h2>

<h4 align="center">

[![ArXiv](https://img.shields.io/badge/ArXiv-2412.00136-b31b1b.svg)](https://arxiv.org/abs/2412.00136) <a href="https://openaccess.thecvf.com/content/ICCV2025/html/Shi_FonTS_Text_Rendering_With_Typography_and_Style_Controls_ICCV_2025_paper.html"><img src="https://img.shields.io/static/v1?label=CVF&message=paper&color=blue"> [![Project Website](https://img.shields.io/badge/Project-Website-green.svg)](https://wendashi.github.io/FonTS-Page/)  [![HuggingFace Model](https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg)](https://huggingface.co/SSS/FonTS-SCA) [![HuggingFace Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg)](https://huggingface.co/datasets/SSS/style_fonts_img)


  <div class="is-size-5 publication-authors">
    <span class="author-block">
      <a href="https://wendashi.github.io/">Wenda Shi</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=L2YS0jgAAAAJ">Yiren Song</a><sup>2</sup>,</span>
    <span class="author-block">
      <a href="https://littleor.github.io/">Dengming Zhang</a><sup>3</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en">Jiaming Liu</a><sup>4</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?user=UhnQA3UAAAAJ&hl=zh-CN">Xingxing Zou</a><sup>1,*</sup></span>
  </div>
  
  <div class="is-size-5 publication-authors">
    <span class="author-block"><sup>1</sup>The Hong Kong Polytechnic University, </span>
    <span class="author-block"><sup>2</sup>National University of Singapore, </span>
    <span class="author-block"><sup>3</sup>Zhejiang University, </span>
    <span class="author-block"><sup>4</sup>Tiamat AI</span>
    <span class="author-block"><sup>*</sup>Corresponding author</span>
  </div>
</h4>

This repository is the official implementation of FonTS, a two-stage DiT-based pipeline to achieve word-level typographic control, font consistency, and artistic style consistency in text rendering tasks.

![FonTS teaser image](https://raw.githubusercontent.com/ArtmeScienceLab/FonTS/main/teaser/teaser.png)

### Environment

```bash
# Create conda environment
conda create -n fonts python=3.12

# Activate environment
conda activate fonts

# Install dependencies
pip install -r requirements.txt
```

or：

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate fonts
```

### Inference
```bash
python /path/to/FonTS/flux+SCA-only/infer_flux+SCA-only.py
```

### Evaluation
Benchmark download: [🤗 ATR-Bench](https://huggingface.co/datasets/SSS/ATR-bench/tree/main) 

### Dataset
Trainset download: [🤗 SC-artext](https://huggingface.co/datasets/SSS/SC-artext) 

### Serial Work
[WordCon: Word-level Typography Control in Scene Text Rendering](https://wendashi.github.io/WordCon-Page/)

### Citation
If you find this work helpful, please consider citing our paper or give a star🌟:

```
@InProceedings{Shi_2025_ICCV,
    author    = {Shi, Wenda and Song, Yiren and Zhang, Dengming and Liu, Jiaming and Zou, Xingxing},
    title     = {FonTS: Text Rendering With Typography and Style Controls},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {18463-18474}
}

@article{WordCon,
    author   = {Shi, Wenda and Song, Yiren and Rao, Zihan and Zhang, Dengming and Liu, Jiaming and Zou, Xingxing},
    journal  = {IEEE Transactions on Circuits and Systems for Video Technology},
    title    = {WordCon: Word-level Typography Control in Visual Text Rendering},
    year     = {2026},
    volume   = {},
    number   = {},
    pages    = {1-1},
    keywords = {Visual text rendering; parameter-efficient fine-tuning; image synthesis; text-image alignment},
    doi      = {10.1109/TCSVT.2026.3686871}
}
```

### Acknowledgments

This implementation is built based on [xflux](https://github.com/XLabs-AI/x-flux), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [Flux](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers).

This work was substantially supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project No. PolyU/RGC Project PolyU 25211424) and partially supported by a grant from PolyU university start-up fund (Project No. P0047675).

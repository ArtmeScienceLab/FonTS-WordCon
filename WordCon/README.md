# WordCon: Word-level Typography Control in Visual Text Rendering

<h4 align="center">

[![IEEE Xplore](https://img.shields.io/static/v1?label=IEEE&message=paper&color=blue)](https://ieeexplore.ieee.org/document/11494032) [![Project Website](https://img.shields.io/badge/Project-Website-green.svg)](https://wendashi.github.io/WordCon-Page/) [![HuggingFace Model](https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg)](https://huggingface.co/ArtmeScienceLab/WordCon) [![HuggingFace Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg)](https://huggingface.co/datasets/ArtmeScienceLab/WordCon-WC-dataset)


  <div class="is-size-5 publication-authors">
    <span class="author-block">
      <a href="https://wendashi.github.io/">Wenda Shi</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=L2YS0jgAAAAJ">Yiren Song</a><sup>2</sup>,</span>
    <span class="author-block">
      <a href=>Zihan Rao</a><sup>3</sup>,</span>
    <span class="author-block">
      <a href="https://littleor.github.io/">Dengming Zhang</a><sup>4</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en">Jiaming Liu</a><sup>5</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?user=UhnQA3UAAAAJ&hl=zh-CN">Xingxing Zou</a><sup>1,*</sup></span>
  </div>
  
  <div class="is-size-5 publication-authors">
    <span class="author-block"><sup>1</sup>The Hong Kong Polytechnic University, </span>
    <span class="author-block"><sup>2</sup>National University of Singapore, </span>
    <span class="author-block"><sup>3</sup>Chongqing University, </span>
    <span class="author-block"><sup>4</sup>Zhejiang University, </span>
    <span class="author-block"><sup>5</sup>Alibaba Group</span>
    <span class="author-block"><sup>*</sup>Corresponding author</span>
  </div>
</h4>

![WordCon teaser image](https://raw.githubusercontent.com/ArtmeScienceLab/FonTS/main/WordCon/teaser/teaser.png)

This folder contains the official implementation of **WordCon**, a framework for word-level typographic control in text rendering tasks.

## Environment Setup

To avoid dependency conflicts with FonTS, we provide a standalone environment for WordCon:

```bash
# Create environment from yml file
conda env create -f wordcon_environment.yml

# Activate environment
conda activate wordcon
```

## Inference Code

You can run the inference script using the following command:

```bash
python examples/inference.py
```

## Future Work Suggestion 🤔

By leveraging tools like the Recraft Mockup Generator (see the [official blog](https://www.recraft.ai/mockup-generator) and the [community-reproduced version](https://github.com/LAOGOU-666/Comfyui-Transform)), you can project word-level controlled synthetic text onto diverse curved surfaces, expanding beyond flat rendering.


## Citation

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

This implementation builds upon several open-source projects: [Flux](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), and [OminiControl](https://github.com/Yuanshi9815/OminiControl) for the generation pipeline; [Break-A-Scene](https://github.com/google/break-a-scene) and [ConceptAttention](https://github.com/helblazer811/ConceptAttention) for attention analysis; and [Hi-SAM](https://github.com/ymy-k/Hi-SAM) and [SPTS v2](https://github.com/bytedance/SPTSv2) for text region labeling.

This work was substantially supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project No. PolyU/RGC Project PolyU 25211424) and partially supported by a grant from PolyU university start-up fund (Project No. P0047675).
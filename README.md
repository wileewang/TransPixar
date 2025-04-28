## TransPixeler: Advancing Text-to-Video Generation with Transparency (CVPR2025)
<br>
<a href="https://arxiv.org/abs/2501.03006"><img src='https://img.shields.io/badge/arXiv-2501.03006-b31b1b.svg'></a>
<a href='https://wileewang.github.io/TransPixeler'><img src='https://img.shields.io/badge/Project_Page-TransPixeler-blue'></a>
<a href='https://huggingface.co/spaces/wileewang/TransPixar'><img src='https://img.shields.io/badge/HuggingFace-TransPixeler-yellow'></a>
<a href="https://discord.gg/7Xds3Qjr"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
<a href="https://github.com/wileewang/TransPixar/blob/main/wechat_group.jpg"><img src="https://img.shields.io/badge/Wechat-Join-green?logo=wechat&amp"></a>
<br>

[Luozhou Wang*](https://wileewang.github.io/), 
[Yijun Li**](https://yijunmaverick.github.io/), 
[Zhifei Chen](), 
[Jui-Hsien Wang](http://juiwang.com/), 
[Zhifei Zhang](https://zzutk.github.io/), 
[He Zhang](https://sites.google.com/site/hezhangsprinter), 
[Zhe Lin](https://sites.google.com/site/zhelin625/home), 
[Ying-Cong Chen†](https://www.yingcong.me)

HKUST(GZ), HKUST, Adobe Research.

\* Internship Project  
\** Project Lead  
† Corresponding Author

Text-to-video generative models have made significant strides, enabling diverse applications in entertainment, advertising, and education. However, generating RGBA video, which includes alpha channels for transparency, remains a challenge due to limited datasets and the difficulty of adapting existing models. Alpha channels are crucial for visual effects (VFX), allowing transparent elements like smoke and reflections to blend seamlessly into scenes.  
We introduce TransPixar, a method to extend pretrained video models for RGBA generation while retaining the original RGB capabilities. TransPixar leverages a diffusion transformer (DiT) architecture, incorporating alpha-specific tokens and using LoRA-based fine-tuning to jointly generate RGB and alpha channels with high consistency. By optimizing attention mechanisms, TransPixeler preserves the strengths of the original RGB model and achieves strong alignment between RGB and alpha channels despite limited training data.  
Our approach effectively generates diverse and consistent RGBA videos, advancing the possibilities for VFX and interactive content creation.

<!-- insert a teaser gif -->
<!-- <img src="assets/mi.gif"  width="640" /> -->



## 📰 News

- **[2025.04.28]** We have introduced a new development branch [`wan`](https://github.com/wileewang/TransPixar/tree/wan) that integrates the [Wan2.1](https://github.com/Wan-Video/Wan2.1) video generation model to support **joint generation** tasks. This branch includes training code tailored for generating both RGB and associated modalities (e.g., segmentation maps, alpha masks) from a shared text prompt.

- **[2025.02.26]** **TransPixeler** is accepted by CVPR 2025! See you in Nashville!

- **[2025.01.19]** We've renamed our project from **TransPixar** to **TransPixeler**!!

- **[2025.01.17]** We’ve created a [Discord group](https://discord.gg/7Xds3Qjr) and a [WeChat group](https://github.com/wileewang/TransPixar/blob/main/wechat_group.jpg)! Everyone is welcome to join for discussions and collaborations.

- **[2025.01.14]** Added new tasks to the repository's roadmap, including support for Hunyuan and LTX video models, and ComfyUI integration.

- **[2025.01.07]** Released project page, arXiv paper, inference code, and Hugging Face demo.




## 🔥 New Branch for Joint Generation with Wan2.1

We have introduced a new development branch [`wan`](https://github.com/wileewang/TransPixar/tree/wan) that integrates the [Wan2.1](https://github.com/Wan-Video/Wan2.1) video generation model to support **joint generation** tasks.

In the `wan` branch, we have developed and released training code tailored for joint generation scenarios, enabling the simultaneous generation of RGB videos and associated modalities (e.g., segmentation maps, alpha masks) from a shared text prompt.

**Key features of the `wan` branch:**
- **Integration of Wan2.1**: Leverages the capabilities of the Wan2.1 video generation model for enhanced performance.
- **Joint Generation Support**: Facilitates the concurrent generation of RGB and paired modality videos.
- **Dataset Structure**: Expects each sample to include:
  - A primary video file (`001.mp4`) representing the RGB content.
  - A paired secondary video file (`001_seg.mp4`) with a fixed `_seg` suffix, representing the associated modality.
  - A caption text file (`001.txt`) with the same base name as the primary video.
- **Periodic Evaluation**: Supports periodic video sampling during training by setting `eval_every_step` or `eval_every_epoch` in the configuration.
- **Customized Pipelines**: Offers tailored training and inference pipelines designed specifically for joint generation tasks.

👉 To utilize the joint generation features, please checkout the [`wan`](https://github.com/wileewang/TransPixar/tree/wan) branch.




## Contents

* [Installation](#installation)
* [TransPixar LoRA Weights](#transpixar-lora-hub) 
* [Training](#training)
* [Inference](#inference)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)



## Installation

```bash
# For the main branch
conda create -n TransPixeler python=3.10
conda activate TransPixeler
pip install -r requirements.txt
```

**Note:**  
If you want to use the **Wan2.1 model**, please first checkout the `wan` branch:

```bash
git checkout wan
```

## TransPixeler LoRA Weights

Our pipeline is designed to support various video tasks, including Text-to-RGBA Video, Image-to-RGBA Video.

We provide the following pre-trained LoRA weights:

| Task          | Base Model                                                    | Frames | LoRA weights                                                       | Inference VRAM |
|---------------|---------------------------------------------------------------|--------|--------------------------------------------------------------------|----------------|
| T2V + RGBA    | [THUDM/CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5b)       | 49     | [link](https://huggingface.co/wileewang/TransPixar/blob/main/cogvideox_rgba_lora.safetensors) | ~24GB          |


## Training - RGB + Alpha Joint Generation
We have open-sourced the training code for **Mochi** on RGBA joint generation. Please refer to the [Mochi README](Mochi/README.md) for details.


## Inference - Gradio Demo
In addition to the [Hugging Face online demo](https://huggingface.co/spaces/wileewang/TransPixar), users can also launch a local inference demo based on CogVideoX-5B by running the following command:

```bash
python app.py
```

## Inference - Command Line Interface (CLI)
To generate RGBA videos, navigate to the corresponding directory for the video model and execute the following command:
```bash
python cli.py \
    --lora_path /path/to/lora \
    --prompt "..."
```

---

## Acknowledgement

* [finetrainers](https://github.com/a-r-r-o-w/finetrainers): We followed their implementation of Mochi training and inference.
* [CogVideoX](https://github.com/THUDM/CogVideo): We followed their implementation of CogVideoX training and inference.

We are grateful for their exceptional work and generous contribution to the open-source community.

## Citation

```bibtex
@misc{wang2025transpixeler,
      title={TransPixeler: Advancing Text-to-Video Generation with Transparency}, 
      author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Ying-Cong Chen},
      year={2025},
      eprint={2501.03006},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.03006}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wileewang/TransPixeler&type=Date)](https://star-history.com/#wileewang/TransPixar&Date)

# RGBA LoRA Training Instructions 

<!-- <table align=center>
<tr>
<th align=center> Dataset Sample </th>
<th align=center> Test Sample </th>
</tr>
<tr>
  <td align=center><video src="https://github.com/user-attachments/assets/6f906a32-b169-493f-a713-07679e87cd91"> Your browser does not support the video tag. </video></td>
  <td align=center><video src="https://github.com/user-attachments/assets/d356e70f-ccf4-47f7-be1d-8d21108d8a84"> Your browser does not support the video tag. </video></td>
</tr>
</table> -->
<!-- 
Now you can make Mochi-1 your own with `diffusers`, too 🤗 🧨

We provide a minimal and faithful reimplementation of the [Mochi-1 original fine-tuner](https://github.com/genmoai/mochi/tree/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/demos/fine_tuner). As usual, we leverage `peft` for things LoRA in our implementation. 

**Updates**

December 1 2024: Support for checkpoint saving and loading. -->

We follow the same steps as the original [finetrainers](https://github.com/a-r-r-o-w/finetrainers/blob/main/training/mochi-1/README.md) to prepare the [RGBA dataset](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).
For RGBA dataset, you can follow the instructions above to preprocess the dataset yourself. 

Here are some detailed steps to prepare the dataset for Mochi-1 fine-tuning:

1. Download our preprocessed [Video RGBA dataset](https://hkustgz-my.sharepoint.com/:u:/g/personal/lwang592_connect_hkust-gz_edu_cn/EezKQoum3IVJiJ9c8GebNfYBe-xN0OS5mVUvAwyL_rQLuw?e=1obdbA), which has undergone preprocessing operations such as color decontamination and background blur.
2. Use `trim_and_crop_videos.py` to crop and trim the RGB and Alpha videos as needed.  
3. Use `embed.py` to encode the RGB videos into latent representations and embed the video captions into embeddings.
4. Use `embed.py` to encode the Alpha videos into latent representations.
5. Concatenate the RGB and Alpha latent representations along the frames dimension.

Finally, the dataset should be in the following format:
```
<video_1_concatenated>.latent.pt
<video_1_captions>.embed.pt
<video_2_concatenated>.latent.pt
<video_2_captions>.embed.pt
```


Now, we're ready to fine-tune. To launch, run:

```bash
bash train.sh
```
**Note:**  

The arg `--num_frames` is used to specify the number of frames of generated **RGB** video. During generation, we will actually double the number of frames to generate the **RGB** video and **Alpha** video jointly. This double operation is automatically handled by our implementation. 

For an 80GB GPU, we support processing RGB videos with dimensions of 480 × 848 × 79 (Height × Width × Frames) at a batch size of 1 using bfloat16 precision for training. However, the training is relatively slow (over one minute per iteration) because the model processes a total of 79 × 2 frames as input. Generally, good results can be achieved after approximately 2000 iterations.





~~We haven't rigorously tested but without validation enabled, this script should run under 40GBs of GPU VRAM.~~

## Inference

To generate the RGBA video, run:

```bash
python cli.py \
    --lora_path /path/to/lora \
    --prompt "..." \
```

This command generates the RGB and Alpha videos simultaneously and saves them. Specifically, the RGB video is saved in its premultiplied form. To blend this video with any background image, you can simply use the following formula:

```python
com = rgb + (1 - alpha) * bgr
```

## Known limitations

(Contributions are welcome 🤗)

Our script currently doesn't leverage `accelerate` and some of its consequences are detailed below:

* No support for distributed training. 
* `train_batch_size > 1` are supported but can potentially lead to OOMs because we currently don't have gradient accumulation support.
* No support for 8bit optimizers (but should be relatively easy to add).

**Misc**: 

* We're aware of the quality issues in the `diffusers` implementation of Mochi-1. This is being fixed in [this PR](https://github.com/huggingface/diffusers/pull/10033). 
* `embed.py` script is non-batched. 

<div align="center">
<h1><img src='/assets/x.svg' width="40px"> MoSt-DSA <img src='/assets/Dove.gif' width="40px"></h1>
<h3>Modeling Motion and Structural Interactions for Direct Multi-Frame Interpolation in DSA Images</h3>

[Ziyang Xu](https://ziyangxu.top/)<sup>1</sup>, Huangxuan Zhao<sup>2</sup>, [Ziwei Cui](https://github.com/ziwei-cui)<sup>1</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, Chuansheng Zheng<sup>2</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1 :email:</sup>

(<sup>:email:</sup>) corresponding author.

<sup>1</sup> Institute of AI, School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, China.
<sup>2</sup> Department of Radiology, Union Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, China.

[![arxiv](https://img.shields.io/badge/Paper-Arxiv_preprint_(click_here)-orange)](https://arxiv.org/abs/2407.07078) [![license](https://img.shields.io/badge/License-Apache_2.0_(click_here)-blue)](LICENSE)

</div>

#

<div align="center">
<img src="assets/method_overall_pipeline.png" />
</div>

## 🔔 News

* **`July 10, 2024`:**  We released our code. If you found it helpful, please give us a star 🌟 and cite our paper! 🤗

* **`July 4, 2024`:**   MoSt-DSA is accepted to ECAI2024! 🍻 The [preprint paper 📔](https://arxiv.org/abs/2407.07078) can be found on arxiv.

## 🔖 Abstract
Artificial intelligence has become a crucial tool for medical image analysis. As an advanced cerebral angiography technique, Digital Subtraction Angiography (DSA) poses a challenge where the radiation dose to humans is proportional to the image count. By reducing images and using AI interpolation instead, the radiation can be cut significantly. However, DSA images present more complex motion and structural features than natural scenes, making interpolation more challenging. We propose MoSt-DSA, the first work that uses deep learning for DSA frame interpolation. Unlike natural scene Video Frame Interpolation (VFI) methods that extract unclear or coarse-grained features, we devise a general module that models motion and structural context interactions between frames in an efficient full convolution manner by adjusting optimal context range and transforming contexts into linear functions. Benefiting from this, MoSt-DSA is also the first method that directly achieves any number of interpolations at any time steps with just one forward pass during both training and testing. We conduct extensive comparisons with 7 representative VFI models for interpolating 1 to 3 frames, MoSt-DSA demonstrates robust results across 470 DSA image sequences (each typically 152 images), with average SSIM over 0.93, average PSNR over 38 (standard deviations of less than 0.030 and 3.6, respectively), comprehensively achieving state-of-the-art performance in accuracy, speed, visual effect, and memory usage.

## 🎇 Highlights

* MoSt-DSA is the first work that uses deep learning for DSA frame interpolation, and also the first method that directly achieves any number of interpolations at any time steps with just one forward pass during both training and testing.

* MoSt-DSA demonstrates robust results across 470 DSA image sequences (each typically 152 images), with average SSIM over 0.93, average PSNR over 38 (standard deviations of less than 0.030 and 3.6, respectively), comprehensively achieving SOTA performance in accuracy, speed, visual effect, and memory usage.

* MoSt-DSA can significantly reduce the DSA radiation dose received by doctors and patients when applied clinically, lowering it by 50\%, 67\%, and 75\% when interpolating 1 to 3 frames, respectively.


## 📦 Environment Setups

* python 3.8
* cudatoolkit 11.2.1
* cudnn 8.1.0.77
* See 'MoSt-DSA_env.txt' for Python libraries required

```shell
conda create -n MoSt_DSA python=3.8
conda activate MoSt_DSA
conda install cudatoolkit=11.2.1 cudnn=8.1.0.77
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# cd /xx/xx/MoSt-DSA
pip install -r MoSt-DSA_env.txt
```

## 🗃️ Model Checkpoints
Download the checkpoints of [single-frame interpolation](https://drive.google.com/file/d/1Nb3iputHMVnr4n7b8dS_-BPFcoZ4JQl9/view?usp=sharing), [two-frame interpolation](https://drive.google.com/file/d/1_S3KVy0GVQP5_oq_z-1eXEJdiAJglkVA/view?usp=sharing), [three-frame interpolation](https://drive.google.com/file/d/1iyqtxOlPAaxQSjNv7rnSNw_BpOb8os4x/view?usp=sharing), and put all pkl files into ../MoSt-DSA/weights/checkpoints. You can use these checkpoints as pre-training weights or for inference.

## 🎞️ Inference Demo
Run the following commands to generate single/multi-frame interpolation:

* Single-frame interpolation
```shell
python Simple_Interpolator.py \
--model_path ./weights/checkpoints/Inf1.pkl \
--frame1 ./demo_images/DSA_a.png \
--frame2 ./demo_images/DSA_b.png \
--inter_frames 1
```

* Two-frame interpolation
```shell
python Simple_Interpolator.py \
--model_path ./weights/checkpoints/Inf2.pkl \
--frame1 ./demo_images/DSA_a.png \
--frame2 ./demo_images/DSA_b.png \
--inter_frames 2
```

* Three-frame interpolation
```shell
python Simple_Interpolator.py \
--model_path ./weights/checkpoints/Inf3.pkl \
--frame1 ./demo_images/DSA_a.png \
--frame2 ./demo_images/DSA_b.png \
--inter_frames 3
```

## 🚂 Training on your medical datasets
### Data Preparation
You should extract consecutive frames of data and then organize your dataset as follows:

* For single-frame interpolation
```
├── <Your_datasets_root>
│    ├── Data_part_1
│        ├── Sequence_1
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3.png
│        ├── Sequence_2
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3.png
│        ├── ...
│    ├── Data_part_2
│        ├── Sequence_1
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3.png
│        ├── Sequence_2
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3.png
│        ├── ...
│    | ...
```

* For multi-frame interpolation(for example, three-frame)
```
├── <Your_datasets_root>
│    ├── Data_part_1
│        ├── Sequence_1
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3_gt.png
│            ├── frame_4_gt.png
│            ├── frame_5.png
│        ├── Sequence_2
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3_gt.png
│            ├── frame_4_gt.png
│            ├── frame_5.png
│        ├── ...
│    ├── Data_part_2
│        ├── Sequence_1
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3_gt.png
│            ├── frame_4_gt.png
│            ├── frame_5.png
│        ├── Sequence_2
│            ├── frame_1.png
│            ├── frame_2_gt.png
│            ├── frame_3_gt.png
│            ├── frame_4_gt.png
│            ├── frame_5.png
│        ├── ...
│    | ...
```

Also, you should list the paths of all the sequences for train-set and test-set by making TrainList.txt and TestList.txt, as follows:
* Make TrainList.txt for train-set. Each line in txt corresponds to a sequence path, like:
```
Data_part_1/Sequence_1
Data_part_1/Sequence_2
Data_part_1/Sequence_3
Data_part_1/Sequence_4
Data_part_2/Sequence_1
Data_part_2/Sequence_2
Data_part_2/Sequence_3
Data_part_2/Sequence_4
...
```
* Make TestList.txt for test-set. Each line in txt corresponds to a sequence path, like:
```
Data_part_301/Sequence_1
Data_part_301/Sequence_2
Data_part_301/Sequence_3
Data_part_301/Sequence_4
Data_part_302/Sequence_1
Data_part_302/Sequence_2
Data_part_302/Sequence_3
Data_part_302/Sequence_4
...
```

### Download weight for Multi-loss Calculation
For calculating style loss and perceptual loss, download [weight](https://drive.google.com/file/d/1NnMnRCGHOGjXHZ7Se2_q5c86L0XkMTBq/view?usp=sharing) and put it into ../MoSt-DSA/weights/vgg_weight.


### Training Commands
Taking the training single-frame interpolation model as an example. You can also modify the "--inter_frames" value to specify other number of interpolation frames. If you try to do so, please make sure that "--data_path" and "--txt_path" correspond to the "--inter_frames".

#### Single-machine Single-GPU Training
```shell
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 37501 Train_Test_Pipe.py \
--world_size 1 \
--batch_size 8 \
--data_path /xx/xx/Your_datasets_root \
--txt_path /xx/xx/txts_root/that/has/TrainList/and/TestList/txt \
--inter_frames 1 \
--model_save_name xxx_model \
--note name_of_log_folder \
--note_print xxx_experiment \
--max_epochs 50
```

#### Single-machine Multi-GPU Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port 37502 Train_Test_Pipe.py \
--world_size 4 \
--batch_size 8 \
--data_path /xx/xx/Your_datasets_root \
--txt_path /xx/xx/txts_root/that/has/TrainList/and/TestList/txt \
--inter_frames 1 \
--model_save_name xxx_model \
--note name_of_log_folder \
--note_print xxx_experiment \
--max_epochs 50
```

## 📈 Evaluate on your medical datasets
Taking the evaluating single-frame interpolation model as an example. You can also modify the "--inter_frames" value to specify other number of interpolation frames. If you try to do so, please make sure that "--data_path" and "--txt_path" correspond to the "--inter_frames".

```shell
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 37503 Train_Test_Pipe.py \
--world_size 1 \
--data_path /xx/xx/Your_datasets_root \
--txt_path /xx/xx/txts_root/that/has/TrainList/and/TestList/txt \
--inter_frames 1 \
--note Eval-Model-xxx \
--note_print My-Eval-Inf2 \
--pretrain_weight /xxx/xxx/xxx/modelxxx.pkl \
--only_eval True
```

## 🏵️ Acknowledgements
This project is based on [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI), [Lambda Networks](https://github.com/lucidrains/lambda-networks), [FILM](https://github.com/google-research/frame-interpolation), [Perceptual Loss](https://arxiv.org/abs/1603.08155), [Style Loss](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). Thanks for their wonderful works.

## 💖 Citation
If you find MoSt-DSA is useful in your research or applications, please feel free to give us a star 🌟 and cite our paper:
```shell
@misc{Xu2024MoSt-DSA,
      title={MoSt-DSA: Modeling Motion and Structural Interactions for Direct Multi-Frame Interpolation in DSA Images},
      author={Ziyang Xu and Huangxuan Zhao and Ziwei Cui and Wenyu Liu and Chuansheng Zheng and Xinggang Wang},
      year={2024},
      eprint={2407.07078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.07078},
}
```

# AGNN
Code for ICCV 2019 paper: Zero-shot Video Object Segmentation via Attentive Graph Neural Networks
#
![](../master/framework.png)
### Quick Start

#### Training
1. Download all the training datasets, including MARA10K (split the RGB images and masks into two files) and DUT saliency datasets. Create a folder called images and put these datasets into the folder (data augmentation is suggested for these static images). Download the davis2016 dataset. 

2. Download the deeplabv3 model from [GoogleDrive](https://drive.google.com/open?id=1hy0-BAEestT9H4a3Sv78xrHrzmZga9mj). Put it into the folder pretrained/deep_labv3.

3. Change the video path, image path and deeplabv3 path in train_iteration_conf_agnn.py.  Create two txt files which store the saliency dataset name and DAVIS16 training sequences name. Change the txt path in TripletImg_video1.py.

4. Run command: python train_iteration_conf_agnn.py --dataset davis --gpus 0,1

#### Testing
**For Object level zero-shot VOS:**

1. Install pytorch (version:1.0.1).

2. Download the pretrained model, put in the snapshots folder. Run 'test_iteration_conf_gnn.py' and change the davis dataset path, pretrainde model path and result path.

3. Run command:  python test_iteration_conf_gnn.py --dataset davis --gpus 0

4. Post CRF processing code: https://github.com/lucasb-eyer/pydensecrf (scale=1 for unary,sdims = 1, compat=5 for pairwise Gaussian, sdims=30, schan=5, compat=9
)
The pretrained weight can be download from [GoogleDrive](https://drive.google.com/open?id=1w4hWVC7ZTTVDJCQN6-vOVLY9JLJCru7G).

**For instance-level zero-shot VOS (multiple instances):**
the authors claim that  **We hope the reviewer will not punish us because we have introduced the GPM module in order to improve DeAOT's efficiency. We just believe this will contribute to the VOS community further. **

Wow, firstly, I have never punished any ones. I just give my comments objectively.
Secondly, why do you just believe this will contribute to the VOS community further? Is this an overclaim? This work still follows the AOT [58] framework. The generalization ability of the proposed method has not been proved. Also, the performance promotion compared to AOT is moderate. 
1. Download DAVIS-2017 dataset and run the object level zero-shot VOS for each video. In this way, we can obtain the object-level mask for each frame. 

2. Download the code of PWCNet from [here](https://github.com/sniklaus/pytorch-pwc) and compute the optical flow for each video.

3. Download the code of PReMVOS from [here](https://github.com/JonathonLuiten/PReMVOS). Run the proposal generation and combination code with the provided network. In this way, we can obtain the instance level proposals for each frame. 

4. Run the command proposal_selection_un.py to select the foreground instances from the first frame for each video and generate related json and jpeg file. Copy this file to PReMVOS and make a new file called my_data.

5. Run the code of refinement_net in PReMVOS and generate the mask for each instance.

6. Change the path of first frame as well as annotation in MergeTrack/merge.py. Run the mergetrack code to associate the instance mask across the subsequent frames.


The segmentation results on DAVIS-2016, Youtube-objects and DAVIS-2017 datasets can be download from [GoogleDiver](https://drive.google.com/open?id=1w5nRgUdUz-OxUhEYjytYDXB_xa2r983_).


### Citation
If you find the code and dataset useful in your research, please consider citing:
```
@InProceedings{Wang_2019_ICCV,  
author = {Wang, Wenguan and Lu, Xiankai and Shen, Jianbing and Crandall, David J. and Shao, Ling},  
title = {Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
year = {2019}  
}
```
### Other related projects/papers:
[See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks(CVPR19)](https://github.com/carrierlxk/COSNet)

[Saliency-Aware Geodesic Video Object Segmentation (CVPR15)](https://github.com/wenguanwang/saliencysegment)

[Learning Unsupervised Video Primary Object Segmentation through Visual Attention (CVPR19)](https://github.com/wenguanwang/AGS)

Any comments, please email: carrierlxk@gmail.com

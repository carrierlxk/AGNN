# AGNN
Code for ICCV 2019 paper: Zero-shot Video Object Segmentation via Attentive Graph Neural Networks
#
![](../master/framework.png)
### Quick Start

#### Training
1. Download all the training datasets, including MARA10K (split the RGB images and masks into two files) and DUT saliency datasets. Create a folder called images and put these two datasets into the folder (data augmentation is suggested for these static images). Download the davis2016 dataset. 

2. Download the deeplabv3 model from [GoogleDrive](https://drive.google.com/open?id=1hy0-BAEestT9H4a3Sv78xrHrzmZga9mj). Put it into the folder pretrained/deep_labv3.

3. Change the video path, image path and deeplabv3 path in train_iteration_conf.py.  Create two txt files which store the saliency dataset name and DAVIS16 training sequences name. Change the txt path in PairwiseImg_video.py.
4. Run command: python train_iteration_conf.py --dataset davis --gpus 0,1
#### Testing

1. Install pytorch (version:1.0.1).

2. Download the pretrained model, put in the snapshots folder. Run 'test_iteration_conf_gnn.py' and change the davis dataset path, pretrainde model path and result path.

3. Run command:  python test_iteration_conf_gnn.py --dataset davis --gpus 0

4. Post CRF processing code: https://github.com/lucasb-eyer/pydensecrf

The pretrained weight can be download from [GoogleDrive](https://drive.google.com/open?id=1w4hWVC7ZTTVDJCQN6-vOVLY9JLJCru7G).

The segmentation results on DAVIS-2016, Youtube-objects and DAVIS-2017 datasets can be download from [GoogleDiver](https://drive.google.com/open?id=1w5nRgUdUz-OxUhEYjytYDXB_xa2r983_).

### Citation
If you find the code and dataset useful in your research, please consider citing:

@InProceedings{Wang_2019_ICCV,

author = {Wang, Wenguan and Lu, Xiankai and Shen, Jianbing and Crandall, David J. and Shao, Ling},

title = {Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks},

booktitle = {The IEEE International Conference on Computer Vision (ICCV)},

year = {2019}
}

### Other related projects/papers:
[See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks(CVPR19)](https://github.com/carrierlxk/COSNet)

[Saliency-Aware Geodesic Video Object Segmentation (CVPR15)](https://github.com/wenguanwang/saliencysegment)

[Learning Unsupervised Video Primary Object Segmentation through Visual Attention (CVPR19)](https://github.com/wenguanwang/AGS)

Any comments, please email: carrierlxk@gmail.com

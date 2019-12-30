# AGNN
Code for ICCV 2019 paper: Zero-shot Video Object Segmentation via Attentive Graph Neural Networks
#
![](../master/framework.png)
### Quick Start

#### Testing

1. Install pytorch (version:1.0.1).

2. Download the pretrained model, put in the snapshots folder. Run 'test_iteration_conf_gnn.py' and change the davis dataset path, pretrainde model path and result path.

3. Run command: python test_iteration_conf_gnn.py --dataset davis --gpus 0

4. Post CRF processing code: https://github.com/lucasb-eyer/pydensecrf

The pretrained weight can be download from [GoogleDrive](https://drive.google.com/open?id=14ya3ZkneeHsegCgDrvkuFtGoAfVRgErz) or [BaiduPan](https://pan.baidu.com/s/16oFzRmn4Meuq83fCYr4boQ), pass code: xwup.

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

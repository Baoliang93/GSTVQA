# GSTVQA
Code for 'Learning Generalized Spatial-Temporal Deep Feature  Representation for No-Reference Video Quality Assessment'. The code are mostly based on [VSFA](https://github.com/lidq92/VSFA).
![image](https://user-images.githubusercontent.com/75255236/121126057-1fbca280-c85a-11eb-9b6d-2d221a83b263.png)


# Environment
* Python 3.6.7
* Pytorch 1.6.0  Cuda V9.0.176 Cudnn 7.4.1

# Running
* Using the files in "./GSTVQA/TCSVT_Release/GVQA_Release/VGG16_mean_std_features/" to genrate the multi-scale VGG feature of each frame of each video.

* Train:
“python  ./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/main.py --TrainIndex=1”  （TrainIndex=1：using the CVD2014 datase as source dataset; 2: LIVE-Qua; 3: LIVE-VQC; 4: KoNviD）

* Test:
“python  ./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py --TrainIndex=1”  （TrainIndex=1：using the CVD2014 datase as source dataset; 2: LIVE-Qua; 3: LIVE-VQC; 4: KoNviD）

* The model trained on each above four dataset have been provided in "./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/"

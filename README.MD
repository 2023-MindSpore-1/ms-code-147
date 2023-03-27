# CIRNet_MindSpore_TIP2022

Runmin Cong, Qinwei Lin, Chen Zhang, Chongyi Li, Xiaochun Cao, Qingming Huang, and Yao Zhao, CIR-Net: Cross-modality interaction and refinement for RGB-D salient object detection, IEEE Transactions on Image Processing, 2022. In Press.

# Results of CIR-Net-MindSpore
* Results:
 - We provide the resutls of our CIR-Net on six popular RGB-D SOD benchmark datasets, including STEREO797, NLPR, NJUD, DUT, LFSD and SIP.
 - The results can be download from: [Baidu Cloud](https://pan.baidu.com/s/1bwaGfSZF86FyEIH4B4kanA) (password:1234)

# MindSpore Code of CIR-Net:
* MindSpore implementation of CIR-Net
* Pretrained model:
  - We provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the checkpoint `CIRNet.pth` to `CIRNet_cpts/` folder 
  - Pretrained model using ResNet50 backbone:[Baidu Cloud](https://pan.baidu.com/s/1Ytfv7ztTwjZ3QJ4YVxsw8w) (password:1234)



## Requirements

* Python 3.7
* cudatoolkit  11.1.1  
* mindspore-gpu  1.8.1 
```
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1.1 -c mindspore -c conda-forge
```

## Data Preprocessing
* Please download and put the train data to `data` folder.
* train data can be download from: [Baidu Cloud](https://pan.baidu.com/s/1NFt3eSpdNA99DuP9O5zpHA) (password:1234)
* test data can be download from: [Baidu Cloud](https://pan.baidu.com/s/1KVCLaXLrMZDUZDpYBd_SJA) (password:1234)

## Test
```
python3 CIRNet_test.py --backbone R50 --test_model CIRNet_R50.pth
```

## Train
```
python3 CIRNet_train.py --backbone R50
```

* You can find the results in the `test_maps` folder

# If you use our CIR-Net, please cite our paper:

# Contact Us
If you have any questions, please contact Runmin Cong (rmcong@bjtu.edu.cn) or Qinwei Lin (lqw22@mails.tsinghua.edu.cn).

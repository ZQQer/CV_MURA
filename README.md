# 1020244024 张瑞璇 高级计算机视觉大作业

# 代码说明文档
代码改自 "[StructureFlow: Image Inpainting via Structure-aware Appearance Flow](https://arxiv.org/abs/1908.03852)" (ICCV 2019)

### 环境

1. Pytorch >= 1.0
2. Python 3
3. NVIDIA GPU + CUDA 9.0
4. Tensorboard
 
### 准备

Build Gaussian Sampling CUDA package 

   ```bash
   cd ./CV_MURA/resample2d_package
   python setup.py install --user
   ```
### 文件说明
```
main.py                 主函数
train.py                训练
test.py                 测试与可视化
model_config.yaml       参数设置

./src                   模型与数据
./scrips                数据预处理
./results               
    /train_inpaint_MURA 本作业模型
        /checkpoints    模型文件
        /images         训练中的验证集结果
./resample2d_package    安装包
./MURA_E_txt            训练、验证、测试集
./load_result           可视化结果
./data
    /org                以病例为单位的原始图像数据
    /RTV                原始图像的RTV预处理结果
```

### Running

####1.	数据

斯坦福大学发布的肌肉骨骼X光数据集——[MURA](stanfordmlgroup.github.io/competitions/mura/)

**数据较多，但为了让程序运行，只上传了少部分数据，如果需要请自行从官网下载**

本作业只用到了肘部数据

且以将图像进行预处理（./data/RTV），若下载了其他数据，需要运行./scripts/RTV_python.py 


####2.	训练

由于数据大小，训练集与测试集没有全部上传，需要自行下载。

但已将训练集、验证集、测试集划分出来，在./MURA_E_txt中：

```
gt_for_visual、structure_for_visual  可视化测试图像的原图与结构图    （已上传）
gt_train、structure_train            训练集的原图与结构图           （未上传）
gt_val、structure_val                验证集的原图与结构图           （未上传）
gt_test、structure_test              测试集的原图与结构图           （未上传）
gt_testn、structure_testn            阴性测试集的原图与结构图        （未上传）
gt_testp、structure_testp            阳性测试集的原图与结构图        （未上传）
```
开始训练之前，需要在[model_config.yaml](model_config.yaml)中设置好数据以及超参数。

网络的名字、路径等，在[main](main.py)中设置。

最后，运行:

```bash
python train.py 
```

####3.	测试与可视化

开始训练之前，需要在[model_config.yaml](model_config.yaml)中设置好测试数据以及超参数。

加载网络的名字、数据保存路径等，在[main](main.py)中设置。

最后，运行:

```bash
python test.py 
```


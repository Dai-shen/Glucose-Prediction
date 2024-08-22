## 项目简介

该项目旨在基于扩张 CNN 的可穿戴设备实现血糖浓度监测。我们使用扩张 CNN 对模型进行改进，增强了时序性捕捉和泛化能力，选取 HuberLoss 作为损失函数，加入先验知识处理加速度计和食物特征，降低特征维度，实现实时血糖监测。扩张
CNN 块对不同时间步的数据进行卷积，最后融合多尺度特征，增强特征表达能力。我们的模型训练 300 轮后，在预测高糖次数上获得良好结果。


## 运行

### 1. 准备环境
通过conda创建环境，然后pip安装需要的包，需自行解决依赖问题

`pip install -r requirements.txt`

### 2 数据预处理
[数据源文件](https://physionet.org/static/published-projects/big-ideas-glycemic-wearable/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0.zip)

进入项目根目录: 
```
cd glucose_prediction
```
运行bash文件：
```
bash scripts/generate_data.sh
```

- load_path: 原始数据文件存储的路径

- person_list：指定进行数据预处理的若干个用户
- save_path: 保存处理后的结果，默认路径为 `{abs_path_to_glucose}/data/newdata`

所有进行数据预处理后的文件存放于默认路径 `glucose-prediction/newdata/`目录下面


### 3. 模型训练
运行bash文件：
```
bash scripts/train.sh
```

- data_path: 数据预处理后的数据文件绝对路径，如
`{abs_path_to_glucose-prediction}/data/newdata`
- train_sample：用于训练的数据集，如`'02,03,04,05,06,07,08,09'`
- model_name: 保存的模型命名
- save_model: 模型保存路径

### 4 模型测试
运行bash文件:
```
bash scripts/test.sh
```

- data_path: 数据预处理后的数据文件绝对路径，同上
- test-sample: 测试样本集
- load_model: 训练的最优模型的存储路径, `{abs_path_to_glucose-prediction}/src/our_result/*.pth`

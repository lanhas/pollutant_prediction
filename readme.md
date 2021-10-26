# 污染物浓度预测



This is a python project that predict the pollutant concentration prediction from WRF-CMAQ data.

空气质量预报二次建模——2021年中国研究生数学建模竞赛B题



![image-20211024095324284](C:\Users\72454\AppData\Roaming\Typora\typora-user-images\image-20211024095324284.png)

### 安装Pytorch和其他依赖

```bash
# Python 3.8.5

pip install pandas seaborn scikit-learn joblib visdom

```

# 使用

```bash
# 训练模型
python train.py
# 模型预测
python predict.py
```

# 操作流程

① 使用jupyter notebook文件生成训练数据；

② 训练模型（可选模型为“bp”和“lstm”）；

③ 污染物浓度预测；


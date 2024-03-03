#只是尝试model是否可行

import numpy as np
import pandas as pd
import joblib
import lightgbm

model = joblib.load("lightgbm_model_130_0.3_639_1.0.joblib")

# 特征列表
features = ['白蛋白', '年龄', '低密度脂蛋白', '活化部分凝血活酶时间', '甘油三酯', '纤维蛋白原',
            '嗜酸性粒细胞数', '尿素氮', '淋巴细胞数', '高密度脂蛋白', '红细胞平均体积',
            '中性粒细胞数', '钠', '总胆红素', '球蛋白']

# 生成随机测试数据
#test_data = pd.DataFrame(np.random.uniform(-1, 1, size=(1, len(features))), columns=features)
test_data = [1]*15
test_data = np.array(test_data).reshape(1,len(features))

print(test_data)

# 预测
probability = model.predict(test_data)
predictions = np.round(probability)

print(probability)
print(predictions)
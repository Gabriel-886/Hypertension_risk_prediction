from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 加载模型
model = joblib.load("Models/lightgbm_model_130_0.3_639_1.0.joblib")
scaler = joblib.load('Models/scaler.pkl')

# 特征列表
features = ['白蛋白', '年龄', '低密度脂蛋白', '活化部分凝血活酶时间', '甘油三酯', '纤维蛋白原',
            '嗜酸性粒细胞数', '尿素氮', '淋巴细胞数', '高密度脂蛋白', '红细胞平均体积',
            '中性粒细胞数', '钠', '总胆红素', '球蛋白']


@app.route('/')
def index():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取 POST 请求中的参数值
        input_data = [float(request.form[feature]) for feature in features]
        # 检查年龄是否为正整数
        if not input_data[1].is_integer() or input_data[1] <= 0:
            raise ValueError("年龄必须为正整数")
        # 检查其他特征是否为正数
        if any(value <= 0 for value in input_data[:1] + input_data[2:]):
            raise ValueError("所有特征均为正数")

        # 转换成 NumPy 数组
        input_data = np.array(input_data).reshape(1, -1)
        scaler_data = scaler.transform(input_data)

        # 预测
        probability = model.predict(scaler_data)
        prediction = np.round(probability)

        # 返回预测结果
        return render_template('index.html', prediction=probability[0], features=features)
    except Exception as e:
        error_message = str(e)
        # 传递错误消息到模板
        return render_template('index.html', error=error_message, features=features)

if __name__ == '__main__':
    app.run(debug=True)

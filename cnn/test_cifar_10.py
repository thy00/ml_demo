import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# 1. 加载数据集
plt.rcParams["font.sans-serif"] = ["SimHei"] # 显示中文
plt.rcParams["axes.unicode_minus"] = False # 显示负号
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

# 2. 加载模型
loaded_model = tf.keras.models.load_model("cifar10_model.h5")

# 3. 推理
new_images = test_images[11:16]

predictions = loaded_model.predict(new_images)
predicted_classes = tf.argmax(predictions, axis= 1).numpy()

# 4. 可视化结果
plt.figure(figsize = (10, 5))
for i in range(len(new_images)):
    plt.subplot(1, 5, i +1) # 显示子图
    plt.imshow(new_images[i]) # 显示图片
    plt.title(f"预测：{class_names[predicted_classes[i]]}") # 图片类别
    plt.axis("off") # 不显示坐标轴
plt.tight_layout()
plt.show()

# print(predicted_classes)
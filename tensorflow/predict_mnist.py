import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载保存的模型
model = tf.keras.models.load_model('my_mnist_model.h5')

# 加载测试数据（仅用于示例）
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

# 预测随机测试样本
random_index = np.random.randint(0, len(x_test))
x_new = x_test[random_index:random_index+1]
prediction = model.predict(x_new)
predicted_digit = np.argmax(prediction)

# x_new = x_test[0:1]
# prediction = model.predict(x_new)
# predicted_digit = np.argmax(prediction)

# 可视化预测结果
plt.imshow(x_new[0], cmap='gray')
plt.title(f"predicted digit: {predicted_digit}")
plt.axis('off')
plt.show()

print(f"随机选择的样本索引是: {random_index}")
print(f"预测的数字是：{predicted_digit}")
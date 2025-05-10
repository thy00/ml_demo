import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import numpy as np

# 1. 收集和准备数据
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. 设计模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 训练模型
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
print("训练完成！")

# 4. 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"测试集准确率：{test_accuracy*100:.2f}%")

# 5. 进行推理
x_new = x_test[0:1]  # 取测试集的第一张图片
prediction = model.predict(x_new)
predicted_digit = np.argmax(prediction)
print(f"预测的数字是：{predicted_digit}")
print(f"预测概率分布：{prediction}")

# 6. 保存模型（可选）
model.save('my_mnist_model.h5')
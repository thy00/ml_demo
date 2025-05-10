import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore

# 1. 加载数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. 构建CNN，预处理
model = models.Sequential(
[
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(10)
]
)

# 3. 编译训练
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))

# 4. 保存模型
model.save("cifar10_model.h5")


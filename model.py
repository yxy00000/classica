from tensorflow.keras import layers, models, Model, Sequential
#导入必要的库

def AlexNet_v1(i m_height=224, im_width=224, num_classes=1000):
#定义模型，接收参数：图像高度，图像宽度，类别数） ，tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")# output(None, 224, 224, 3)
#输入层，输入图像形状（高度，宽度，通道数（RGB图像为3）），以及数据类型
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
#添加零填充层，确保卷积时卷积核不会越界，（（1,2），（1,2））表示在高度和宽度两个维度上分别添加一行零边，(input_image)表示将上一层的输出作为该层的输入
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
#卷积层，48为卷积核数量，kernel_size为卷积核大小11×11,4为卷积操作的步长，relu为所采用的激活函数，（x）表示将上一层的输出x作为该层的输入
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
#最大池化层，池化窗口大小为3，池化操作的步长为2
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
#padding="same" ，保持输入和输出特征图的尺寸相同，使用 padding="same"，会在特征图的周围自动添加适当数量的零填充，以保持尺寸不变，使得卷积后的特征图仍然是27×27
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    x = layers.Flatten()(x)                         # output(None, 6*6*128)
#展平层，将之前卷积的部分展平为一个一维向量
    x = layers.Dropout(0.2)(x)
#dropout层，减少过拟合，这里丢弃概率设为0.2
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
#全连接层，2048为神经元个数，relu为激活函数
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dense(num_classes)(x)                  # output(None, 5)
#全连接层，也为输出层，输出类别数
    predict = layers.Softmax()(x)
#Softmax层，将网络输出转化为每种类别的概率分布，且概率之和为1
    model = models.Model(inputs=input_image, outputs=predict)
#将输入输出组合为一个模型，输入为input_image,输出为predict
    return model
#返回构建好的模型


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)])                              # output(None, 6, 6, 128)

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),                                   # output(None, 2048)
            layers.Dense(num_classes),                                                # output(None, 5)
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

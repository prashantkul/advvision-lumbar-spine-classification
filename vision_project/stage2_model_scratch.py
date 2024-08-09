import tensorflow as tf
from tensorflow.keras import layers, Model

class SE(layers.Layer):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SE, self).__init__()
        se_channels = int(in_channels * se_ratio)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Conv2D(se_channels, 1, padding='same')
        self.fc2 = layers.Conv2D(in_channels, 1, padding='same')

    def call(self, x):
        y = self.avg_pool(x)
        y = tf.expand_dims(tf.expand_dims(y, 1), 1)  # To match the dimensions of x
        y = tf.nn.relu(self.fc1(y))
        y = tf.nn.sigmoid(self.fc2(y))
        return x * y

class Bottleneck(layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, group_width=1, se_ratio=0.25):
        super(Bottleneck, self).__init__()
        groups = out_channels // group_width

        self.conv1 = layers.Conv2D(out_channels, 1, strides=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, 3, strides=stride, padding='same', groups=groups, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels, 1, strides=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.se = SE(out_channels, se_ratio)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(out_channels, 1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = tf.identity

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        return tf.nn.relu(out)

class RegNetBase(Model):
    def __init__(self, num_classes=1000):
        super(RegNetBase, self).__init__()

        self.stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.stage1 = self._make_stage(32, 232, 2, 1, 232)
        self.stage2 = self._make_stage(232, 696, 7, 2, 232)
        self.stage3 = self._make_stage(696, 1392, 13, 2, 232)
        self.stage4 = self._make_stage(1392, 3712, 1, 2, 232)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, group_width):
        layers_list = [Bottleneck(in_channels, out_channels, stride, group_width)]
        for _ in range(1, num_blocks):
            layers_list.append(Bottleneck(out_channels, out_channels, 1, group_width))
        return tf.keras.Sequential(layers_list)

    def call(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)
        return x

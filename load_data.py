# import torchio, torch
import tensorflow as tf
import numpy as np
import random
import glob
import pickle
import os
from utils import resize_images, distort_imgs, distort_imgs_v2
# import pydevd
import matplotlib.pyplot as plt

def load_pickle(path):
    # load pickle data
    with open(path, 'rb') as f:
        temp = pickle.load(f)
    return temp

class tf_dataset():
    def __init__(self,
                 HGG_tfr,
                 LGG_tfr,
                 scaling_path,
                 pre_scaling=None,
                 transform_mean_std_path=None,
                 batch=1,
                 size=(160, 160),
                 pad=0,
                 target_depth=128,
                 resize_method='resize',
                 decode_size=(240, 240, 155),
                 buffer_size=32,
                 prefetch_size=32,
                 preprocessing=True,
                 repeat=None,
                 shuffle=True):

        self.HGG_tfr = glob.glob(HGG_tfr) if HGG_tfr !=None else []
        self.LGG_tfr = glob.glob(LGG_tfr) if LGG_tfr !=None else []
        self.scaling_path = scaling_path
        self.pre_scaling = pre_scaling
        self.transform_mean_std_path = transform_mean_std_path
        self.batch = batch
        self.size = size
        self.pad = pad
        self.target_depth = target_depth
        self.resize_method = resize_method
        self.decode_size = decode_size
        self.buffer_size = buffer_size
        self.preprocessing = preprocessing
        self.repeat = repeat
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size

        if self.resize_method == 'crop':
            pad_size = ((np.array(decode_size)-np.array(size))/2).astype(np.int)
            self.pad_height = pad_size[0]
            self.pad_width = pad_size[1]
        if self.scaling_path:
            with open(scaling_path, 'rb') as f:
                self.mean, self.std = pickle.load(f)
        if self.transform_mean_std_path:
            with open(self.transform_mean_std_path, 'rb') as f:
                self.transform_mean_std_dict = pickle.load(f)


    def random_depth_crop(self, data, label, channel_axis=0):
        # pydevd.settrace(suspend=False)
        origin_depth = data.shape[channel_axis]
        index = random.randint(0, origin_depth-self.target_depth)
        temp_data = data[index:index+self.target_depth]
        temp_label = label[index:index+self.target_depth]
        if self.preprocessing and random.uniform(0, 1)>0.5:
            # flip depth
            temp_data = temp_data[::-1]
            temp_label = temp_label[::-1]
        return temp_data, temp_label

    def parse_function(self, tfr):
        """
        because of dtype(tf.string), don't define shape in tf.io.FixedLenFeature, ie:
        features = {'data': tf.io.FixedLenFeature(shape=(240, 240, 4), dtype=tf.string),
                    'label': tf.io.FixedLenFeature(shape=(240, 240, 1), dtype=tf.string)}
        """
        features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
                    'label': tf.io.FixedLenFeature([], dtype=tf.string),
                    'tumor_type': tf.io.FixedLenFeature([], dtype=tf.string),
                    'name': tf.io.FixedLenFeature([], dtype=tf.string)}
        parsed_features = tf.io.parse_single_example(tfr, features)
        data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), self.decode_size+tuple([4]))
        label = tf.reshape(tf.io.decode_raw(parsed_features['label'], out_type=tf.float32), self.decode_size+tuple([3]))
        if self.pad!=0:
            zero_pad_data = tf.zeros(self.decode_size[:2]+(self.pad,)+(4,), dtype=data.dtype)
            zero_pad_label = tf.zeros(self.decode_size[:2]+(self.pad,)+(3,), dtype=data.dtype)
            data = tf.concat([data, zero_pad_data], axis=-2)
            label = tf.concat([label, zero_pad_label], axis=-2)

        tumor_type = parsed_features['tumor_type']
        name = parsed_features['name']
        detail = tf.stack((tumor_type, name), 0)
        #background label
        zero_label = tf.logical_not(tf.cast(tf.reduce_sum(label, axis=-1, keepdims=True), tf.bool))
        label = tf.concat([tf.cast(zero_label, dtype=label.dtype), label], axis=-1)
        #(W,H,D,C)--->(D,W,H,C)
        t_fun = lambda x: tf.transpose(x, (2,0,1,3))
        data = t_fun(data)
        label = t_fun(label)

        return data, label, detail

    def augument(self, x, y, detail):
        # pydevd.settrace(suspend=False)
        x, y = distort_imgs_v2(x, y)
        return x, y, detail

    def scaling(self, x, y, detail, *args):

        if self.transform_mean_std_path != None:
            scaling =  args[0]
            mean, std = scaling[:, 0], scaling[:, 1]
            x = (x-mean)/std
        elif self.pre_scaling:
            mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
            std = tf.math.reduce_std(x, axis=(1, 2, 3), keepdims=True)
            x = (x-mean)/std
        else:
            x = (x-self.mean)/self.std
        return x, y, detail

    def resize(self, x, y, detail):
        if self.pad == 0:
            x, y = self.random_depth_crop(x, y)
        if self.resize_method == 'crop':
            x = x[:, self.pad_height: -self.pad_height, self.pad_width: -self.pad_width]
            y = y[:, self.pad_height: -self.pad_height, self.pad_width: -self.pad_width]
        elif self.resize_method == 'resize':
            x = resize_images(x, self.size)
            y = resize_images(y, self.size)
        else:
            pass
        return x, y, detail

    def transform_scaling_params(self, x, y, detail):
        # pydevd.settrace(suspend=False)
        key = detail.numpy()[-1].decode()
        transform_scaling = self.transform_mean_std_dict[key]
        #3D轉4D
        e_transform_scaling = np.expand_dims(transform_scaling, axis=1)
        return x, y, detail, e_transform_scaling

    def generator(self, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        temp_concat_all = self.HGG_tfr+self.LGG_tfr
        if self.shuffle:
            random.shuffle(temp_concat_all)
        dataset = tf.data.TFRecordDataset(temp_concat_all)
        if self.shuffle and self.preprocessing:
            dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.map(self.parse_function,
                              num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(lambda data, label, detail:
                              tf.py_function(self.resize, [data, label, detail], [tf.float32, tf.float32, tf.string]),
                              num_parallel_calls=num_parallel_calls)
        if self.preprocessing:
            dataset = dataset.map(lambda data, label, detail:
                                  tf.py_function(self.augument, [data, label, detail], [tf.float32, tf.float32, tf.string]),
                                  num_parallel_calls=num_parallel_calls)
        if self.transform_mean_std_path:
            dataset = dataset.map(lambda data, label, detail:
                                  tf.py_function(self.transform_scaling_params, [data, label, detail], [tf.float32, tf.float32, tf.string, tf.float32]),
                                  num_parallel_calls=num_parallel_calls)

        dataset = dataset.repeat(self.repeat)
        dataset = dataset.batch(self.batch)
        if self.pre_scaling or self.transform_mean_std_path or self.scaling_path:
            dataset = dataset.map(self.scaling,
                                  num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(self.prefetch_size)

        return dataset

def tfr_test(path):

    return tf.data.TFRecordDataset(path)

if __name__ == '__main__':
    main_tfr_path = r'C:\BraTS\3d\train_2020_pre_scaling(2020_train_landmarks)'
    HGG_tfr = os.path.join(main_tfr_path, '*HGG*')
    LGG_tfr = os.path.join(main_tfr_path, '*LGG*')
    transform_mean_std_path = None
    std_path = None
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    tfd = tf_dataset(HGG_tfr,
                     LGG_tfr,
                     None,
                     pad=0,
                     transform_mean_std_path=transform_mean_std_path,
                     pre_scaling=False,
                     batch=2,
                     preprocessing=True,
                     buffer_size=2)
    generator = tfd.generator()
    for i in generator.take(1):
        print(i[0].shape, i[1].shape, i[2].shape)
        fig, axes = plt.subplots(10, 16)
        f_axes = axes.flatten()
        for j in range(128):
            f_axes[j].imshow(i[0][0, j, :, :, 0])
        plt.show()



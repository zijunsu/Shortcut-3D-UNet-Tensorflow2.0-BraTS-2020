# 3D-UNet-Tensorflow2.0-BraTS-2020
Naive 3D U-Net powered by tensorflow 2.0 and the fast preprocessing pipeline with TFRecord for BraTS 2020 challenge

## BraTS Data
If you don't have data, you can download it from [CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html ) after registration. Or you can download old BraTS version from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) in public.

## Dependencies
* Python 3
* tensorflow >= 2.0
* tensorlayer (based on your tensorflow version)
* tensorflow_addons (based on your tensorflow version) 

## Usage

### Load Data

#### Train
```python
# code in load_data.py and beginning from line 84

features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string),
            'tumor_type': tf.io.FixedLenFeature([], dtype=tf.string),
            'name': tf.io.FixedLenFeature([], dtype=tf.string)}
parsed_features = tf.io.parse_single_example(tfr, features)
data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), self.decode_size+tuple([4]))
label = tf.reshape(tf.io.decode_raw(parsed_features['label'], out_type=tf.float32), self.decode_size+tuple([3]))

# self.decode_size is (width, height, depth) >>> (240, 240, 155) in default.
# You can change width and height based on your GPU memory size.
```
You need to generate TFRecord with the above format. Variables, tumor_type and name, are redundant and used for recording the brain tumor type and sample name only. If you dont't need it, you can set the random variable or same texts when you generate TFRecord.

#### Test
```python
# code in load_test_data.py and beginning from line 74

features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
            'name': tf.io.FixedLenFeature([], dtype=tf.string)}
```


## Environment
* Ubuntu 20.04  
* 12 Intel vCPU  
* 1 NVIDIA Tesla V100s 32GB
* 64G RAM

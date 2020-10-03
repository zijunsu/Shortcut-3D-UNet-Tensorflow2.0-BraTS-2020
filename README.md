# 3D-UNet-Tensorflow2.0-BraTS-2020
Shortcut 3D U-Net powered by tensorflow 2.0 and the fast preprocessing pipeline with TFRecord for BraTS 2020 challenge

## BraTS Data
If you don't have data, you can download it from [CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html ) after registration. Or you can download old BraTS version from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) in public.

## Dependencies
* Python 3
* tensorflow >= 2.0
* tensorlayer (based on your tensorflow version)
* tensorflow_addons (based on your tensorflow version) 

## Usage

### Load Data

You can modify following code based on below descriptions.
#### [Train](/load_data.py#L84-L90)
```python
# code in load_data.py and beginning from line 84

features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string),
            'tumor_type': tf.io.FixedLenFeature([], dtype=tf.string),
            'name': tf.io.FixedLenFeature([], dtype=tf.string)}
parsed_features = tf.io.parse_single_example(tfr, features)
# The shape of dimension, 4, mean the four modalities provided by MRI images.
data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), self.decode_size+tuple([4]))
# The shape of dimension, 3, mean the three sort of true brain tumors, 
# must be NET (non-enhancing tumor), ED (edema) and ET (enhancing Tumor) with 0 (background) or 1 (true) in order.
label = tf.reshape(tf.io.decode_raw(parsed_features['label'], out_type=tf.float32), self.decode_size+tuple([3]))

# self.decode_size is (width, height, depth) >>> (240, 240, 155) in default.
# You can change width and height based on your GPU memory size.

```

You need to generate TFRecords with the above format. Variables, tumor_type and name, are redundant and used for recording the brain tumor type and sample name only. If you dont't need it, you can set the random variable or same texts when you generate TFRecords.

#### [Test](/load_test_data.py#L74-L77)
```python
# code in load_test_data.py and beginning from line 74

features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
            'name': tf.io.FixedLenFeature([], dtype=tf.string)}
parsed_features = tf.io.parse_single_example(tfr, features)
data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), self.decode_size+tuple([4]))
```
The format of test TFRecord is similar with [load_data.py](/load_data.py#L84-L90). The differention is that we don't know true labels and brain tumor types.

### [Model](/model.py)
You can modify the architcture of neural network here and some parameters.

### [Train](/train.py)
Configuring parameters and the path of TFRecord or summaries makes you get the best results and test model easily in [train.py](/train.py#L13-L44).

## Environment
* Ubuntu 20.04  
* 12 Intel vCPU  
* 1 NVIDIA Tesla V100s 32GB
* 64G RAM

## BraTS20 Validation Set Scores (Team Name: BIOMIL)
These results provided by [leaderboard](https://www.cbica.upenn.edu/BraTS20/lboardValidation.html) is computed by [CBICA Image Processing Portal](https://ipp.cbica.upenn.edu/) online with 125 cases in BraTS 2020 validation dataset.
| Dice_ET | Dice_WT | Dice_TC | Hausdorff95_ET | Hausdorff95_WT | Hausdorff95_TC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|
|0.7539|0.8962|0.82134|32.65874|5.35762|6.57599


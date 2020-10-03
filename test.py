import tensorflow as tf
import numpy as np
from tqdm import tqdm
import metrics
import os
import load_test_data
import model
import time
import utils

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    size = (192, 160)
    origin_size =  (240, 240)
    filter_NET = 500
    resize_method = 'resize'
    mixed_precision = False
    summary_path = 'summaries/20200829_020604'
    std_path = None
    pre_scaling = True
    # test_tfr_path = r'E:\Shared folder\BraTS\tfr\3d\test\test_2020_pre_scaling(2020_train_landmarks)\*'
    test_tfr_path = r'E:\Shared folder\BraTS\tfr\3d\test\test_2020_pre_scaling(2020_train_landmarks)\*'
    transform_mean_std_path = None
    base_path = r'E:\Shared folder\BraTS\2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii.gz'
    # test_tfr_path = r'E:\Shared folder\BraTS\tfr\test\2019\*.tfrecord'
    # test_tfr_path = r'E:\Shared folder\BraTS\tfr\non_zero_non_scaling_2019\valid_concat\*_39.*'


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'Setting {gpu} successfully')
        except RuntimeError as e:
            print(e)

    if mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)

    # cast tfr to dataset api
    with tf.name_scope('loader'):
        # is_training = False >>> stop preprocessing
        # test_loader = load_data.tf_dataset(test_tfr_path,
        #                                    std_path,
        #                                    preprocessing=False,
        #                                    batch=batch,
        #                                    shuffle=False,
        #                                    repeat=1)
        test_loader = load_test_data.tf_dataset(test_tfr_path,
                                                None,
                                                transform_mean_std_path=transform_mean_std_path,
                                                scaling_path=std_path,
                                                pre_scaling=pre_scaling,
                                                preprocessing=False,
                                                batch=1,
                                                shuffle=False,
                                                repeat=1,
                                                size=size,
                                                resize_method=resize_method)

    net = model.UNet()
    # eager mode
    @tf.function
    def test_step(x):
        pred = net(x, training=False)
        s_pred = tf.squeeze(pred, axis=0)
        resize_pred = tf.image.resize(s_pred, origin_size)
        result = metrics.logits_2_label(resize_pred, testing=True)
        return result

    # data and labels generator
    test_generator = test_loader.generator()

    step = tf.Variable(0, trainable=False)

    # saver
    ckpt = tf.train.Checkpoint(net=net, step=step)
    ckpt_manager = tf.train.CheckpointManager(ckpt, summary_path, 5)
    latest_path = ckpt_manager.latest_checkpoint
    if latest_path:
        print('-----------------------Restoring: {}-----------------------'.format(latest_path))
        ckpt.restore(latest_path)

    n_step = step.numpy()
    print('--------------------Evaluation--------------------\n')
    now_time = time.strftime('Step_{}_%Y%m%d_%H%M'.format(n_step), time.localtime())
    nii_path = os.path.join(summary_path, now_time+'_results')
    if not os.path.exists(nii_path):
        os.makedirs(nii_path)

    results_string = ''
    for i, (x, name) in tqdm(enumerate(test_generator)):
        result = test_step(x)
        pad_result = utils.fixed_depth_crop(result, False)
        filter_data, string = utils.count_labels(pad_result, filter_NET)
        results_string += '{}: {}\n'.format(name[0], string)
        print('{}: {}'.format(name[0], string))
        utils.save_itk(filter_data, nii_path+'/{}.nii.gz'.format(name[0].numpy().decode()), base_path)
    print('--------------------------------------------------')
    with open(os.path.join(nii_path, 'results.txt'), 'w') as f:
        f.write(results_string)

if __name__ == '__main__':
    main()
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import pickle
# import numexpr as ne
import os
from skimage import io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import metrics
import pandas as pd
import functools

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def combine_transform(array, histogram_transform):
    t_array = array.copy().astype(np.float32)
    tensor = torch.tensor(t_array[:4])
    t_array[:4] = histogram_transform(tensor).numpy()
    return t_array

def distort_imgs(x, y):
    """ data augmentation """
    # (D,W,H,C)--->(C,W,H,D)
    trans_forward = lambda x: np.transpose(x, axes=(3,1,2,0))
    # (C,W,H,D)--->(D,W,H,C)
    trans_backward = lambda x: np.transpose(x, axes=(3,1,2,0))

    ext_transform = torchio.Compose([torchio.RandomFlip(axes=0),
                                     torchio.RandomFlip(axes=1),
                                     torchio.RandomFlip(axes=2)])

    x = trans_forward(x)
    x = combine_transform(x, ext_transform)
    x = trans_backward(x)

    y = trans_forward(y)
    y = combine_transform(y, ext_transform)
    y = trans_backward(y)
    return x, y

def divide_affine(x, affine_func):
    #避免過大matrix抱錯
    w, h, depth = x.shape
    index = depth//2
    x[:, :, :index] = affine_func(x[:, :, :index])
    x[:, :, -index:] = affine_func(x[:, :, -index:])
    return x

def distort_imgs_v2(x, y):

    # cast eager tensor to numpy array

    x_array = np.array(x)
    y_array = np.array(y)
    d, w, h, c = x.shape

    # (D,W,H,C)--->(W,H,D,C)
    trans_forward = lambda x: np.transpose(x, axes=(1,2,0,3)).reshape((w, h, -1))
    # (C,W,H,D)--->(D,W,H,C)
    trans_backward = lambda x: np.transpose(x.reshape(w,h,d,c), axes=(2,0,1,3))

    trans_forward_x = trans_forward(x_array)
    trans_forward_y = trans_forward(y_array)

    # 1. Create required affine transformation matrices

    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-180, 180))
    M_flip_h = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
    M_flip_v = tl.prepro.affine_vertical_flip_matrix(prob=0.5)
    M_shift = tl.prepro.affine_shift_matrix(wrg=(-0.1, 0.1), hrg=(-0.1, 0.1), h=h, w=w)
    M_shear = tl.prepro.affine_shear_matrix(x_shear=(-0.05, 0.05), y_shear=(-0.05, 0.05))
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.1))

    # 2. Combine matrices
    # NOTE: operations are applied in a reversed order (i.e., rotation is performed first)
    M_combined = M_shift.dot(M_zoom).dot(M_shear).dot(M_flip_v).dot(M_flip_h).dot(M_rotate)

    # 3. Convert the matrix from Cartesian coordinates (the origin in the middle of image)
    # to image coordinates (the origin on the top-left of image)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)

    # 4. Transform the image using a single operation
    p_func = functools.partial(tl.prepro.affine_transform_cv2 , transform_matrix=transform_matrix)
    trans_forward_x = divide_affine(trans_forward_x, p_func)
    trans_forward_y = divide_affine(trans_forward_y, p_func)

    trans_backward_x = trans_backward(trans_forward_x)
    trans_backward_y = trans_backward(trans_forward_y)

    return trans_backward_x, trans_backward_y

def dump_data(inputs, path, protocol=4):
    with open(path, 'wb') as f:
        pickle.dump(inputs, f, protocol=protocol)

def resize_images(inputs, size=(512, 512)):
    return tf.image.resize(inputs, size=size)

def decay_lr_rate(alpha, epoch, total_epoch=20000, minimum=1e-6, exp=0.9):
    if epoch>total_epoch:
        lr = minimum
    else:
        lr = max(alpha * (1-epoch/total_epoch)**exp, minimum)
    return lr

def warmup_lr_rate(step, warmup_steps, peak_lr=1e-4, total_step=20000, minimum=1e-6, exp_1=0.9, exp_2=3):
    if step<warmup_steps:
        lr = decay_lr_rate(peak_lr, warmup_steps-step, total_epoch=warmup_steps, exp=exp_1, minimum=minimum)
    else:
        lr = decay_lr_rate(peak_lr, step-warmup_steps, total_epoch=total_step, exp=exp_2, minimum=minimum)

    return lr

def images_summary_processing(inputs,
                              true,
                              pred,
                              n_step=None,
                              threshold=0.5,
                              record_method='tfb',
                              images_path=None):
    inputs = inputs[0]
    true = true[0]
    pred = pred[0]
    i_min = tf.reduce_min(inputs, axis=(1, 2), keepdims=True)
    i_max = tf.reduce_max(inputs, axis=(1, 2), keepdims=True)
    inputs = (inputs-i_min)/(i_max-i_min)

    # NCR, ED, ET = tf.split(pred, 3, axis=-1)

    convert_pred = metrics.logits_2_label(pred)[:, :, :, 1:]
    threshold_true = tf.cast(true>threshold, dtype=true.dtype)[:, :, :, 1:]
    # concat along width axis
    true_pred_concat = tf.concat([threshold_true, convert_pred], axis=2)
    reshape_inputs = tf.concat(tf.unstack(inputs, axis=-1), axis=-1)

    plot_WT = tf.concat([reshape_inputs, true_pred_concat[:, :, :, 0]], axis=2)
    plot_TC = tf.concat([reshape_inputs, true_pred_concat[:, :, :, 1]], axis=2)
    plot_ET = tf.concat([reshape_inputs, true_pred_concat[:, :, :, 2]], axis=2)

    plot_WT = tf.expand_dims(plot_WT, axis=-1)
    plot_TC = tf.expand_dims(plot_TC, axis=-1)
    plot_ET = tf.expand_dims(plot_ET, axis=-1)

    if record_method == 'tfb':
        tf.summary.image('WT' , plot_WT, n_step, max_outputs=1)
        tf.summary.image('TC' , plot_TC, n_step, max_outputs=1)
        tf.summary.image('ET' , plot_ET, n_step, max_outputs=1)
    else:
        # scale to uint8
        concat_all = np.concatenate([plot_WT, plot_TC, plot_ET], axis=1)*255
        # tl.visualize.save_images(concat_all, (1, 1), os.path.join(images_path,'{}.jpg'.format(n_step)))
        for i, data in enumerate(concat_all):
            data = data.astype(np.uint8)
            io.imsave(os.path.join(images_path,'{}_{}.jpg'.format(n_step, i+1)), data)

def pre_max(pred):
    #because three labels not overlap, select the most possible label in axis
    # noise = tf.random.uniform(pred.shape, 1e-6, 1e-5)
    n_pred = pred+np.array([1, 7, 19]).reshape((1, 1, 1, -1))
    m_pred = tf.cast(tf.reduce_max(n_pred, axis=-1, keepdims=True) == n_pred, tf.float32) * pred
    return m_pred

def recover_label(pred, dtype=tf.float32):
    n_label = np.array([1, 2, 4]).reshape((1, 1, 1, -1))
    label = tf.convert_to_tensor(n_label, dtype)
    return tf.reduce_sum(pred*label, axis=-1)

def save_itk(image, filename, base_path):
    base = sitk.ReadImage(base_path)
    base_origin = base.GetOrigin()
    im = sitk.GetImageFromArray(image, isVector=False)
    im.SetOrigin(base_origin)
    sitk.WriteImage(im, filename, True)

def to_excel(value, path, rows, columns=('1-Dice', 'IOU', 'Dice_WT', 'Dice_TC', 'Dice_ET'), round=5):
    main_path = os.path.split(path)[0]
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    mean = value.mean(axis=0)
    std = np.std(value, axis=0)
    q_25 = np.quantile(value, 0.25, axis=0)
    q_50 = np.quantile(value, 0.5, axis=0)
    q_75 = np.quantile(value, 0.75, axis=0)
    stack_ext = np.stack([mean, std, q_50, q_25, q_75], axis=0)
    value = np.concatenate([value, stack_ext], axis=0)
    rows.extend(['Mean', 'StdDev', 'Median', '25quantile', '75quantile'])
    df = pd.DataFrame(value, index=rows, columns=columns).round(round)
    df.to_csv(path)

def fixed_depth_crop(data, is_forward=True, target_depth=128, origin_depth=155, start=None):
    if is_forward:
        b, d, w, h, c = data.shape
        if start == None:
            start = (origin_depth - target_depth) // 2
        return data[:, start: start+target_depth]
    else:
        data = data.numpy()
        #channel合併至w與h了
        d, w, h = data.shape
        if start == None:
            start = (origin_depth - target_depth) // 2
            temp_data = np.zeros((origin_depth, w, h) , data.dtype)
            temp_data[start: start+target_depth] = data
        else:
            temp_data = np.zeros((origin_depth, w, h) , data.dtype)
            temp_data[start: start+target_depth] = data[start: start+target_depth]
        return temp_data

def count_labels(data, filter_NET=500):
    mask_1 = np.sum(data==1)
    mask_2 = np.sum(data==2)
    mask_4 = np.sum(data==4)
    string = np.array([mask_1, mask_2, mask_4])
    if filter_NET > 0 and mask_4<=filter_NET:
        data = np.where(data==4, 1, data)
        mask_1 = np.sum(data == 1)
        mask_4 = np.sum(data == 4)
        new_extend = np.array([mask_1, mask_2, mask_4])
        string = "{} >>>>>> {}".format(string, new_extend)
    return data, string

def batch_resize_images(data, size):
    o_b, o_d, o_w, o_h, o_c = data.shape
    s_w, s_h = size
    flat_data = tf.reshape(data, (-1, o_w, o_h, o_c))
    r_flat_data = tf.image.resize(flat_data, size)
    r_data = tf.reshape(r_flat_data, (o_b, o_d, s_w, s_h, o_c))
    return r_data
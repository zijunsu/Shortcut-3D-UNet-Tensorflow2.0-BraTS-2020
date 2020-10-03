import tensorflow as tf
import tensorlayer as tl
import numpy as np

smooth = 1e-5


def dice_coe(output, target, loss_type='jaccard', batch=False, axis=(1, 2, 3), smooth=smooth):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> outputs = tl.act.pixel_wise_softmax(outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    if batch:
        pass
    else:
        dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def iou_coe(output, target, threshold=0.5, batch=False, axis=(1, 2, 3), smooth=smooth):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    # old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    # new haodong
    iou = (inse + smooth) / (union + smooth)
    if batch:
        pass
    else:
        iou = tf.reduce_mean(iou, name='iou_coe')
    return iou  # , pre, truth, inse, union


def convert_2_brats_dice_label(seg):
    # only for NET, ET, TC, except of background
    seg = seg[:, :, :, :, 1:]
    WT = tf.reduce_sum(seg, axis=-1)
    TC = seg[:, :, :, :, 0] + seg[:, :, :, :, 2]
    ET = seg[:, :, :, :, 2]
    return tf.stack([WT, TC, ET], axis=-1)


def logits_2_label(logits, testing=False, dtype=tf.float32):
    argmax = tf.argmax(logits, axis=-1)
    background = tf.cast(argmax == 0, dtype=dtype)
    net = tf.cast(argmax == 1, dtype=dtype)
    ed = tf.cast(argmax == 2, dtype=dtype)
    et = tf.cast(argmax == 3, dtype=dtype)
    if testing:
        concat_all = tf.stack([net, ed, et], axis=-1)
        n_label = np.array([1, 2, 4]).reshape((1, 1, 1, -1))
        label = tf.convert_to_tensor(n_label, dtype)
        return tf.reduce_sum(concat_all * label, axis=-1)
    else:
        return tf.stack([background, net, ed, et], axis=-1)


def train_losses_v1(out_seg, t_seg, axis=(1, 2, 3), list_weight=[[1,1,1,1]]):
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)

    for 2D NN, axis must be 2-D
    WT = ED + ET + NET
    TC = ET + NET
    ET
    """
    # mixed precision processing
    if out_seg.dtype != t_seg.dtype:
        out_seg = tf.cast(out_seg, t_seg.dtype)

    soft_dice = dice_coe(out_seg, t_seg, axis=axis, batch=True)
    # soft_iou = iou_coe(out_seg, t_seg, axis=axis, batch=False)

    hard_out_seg = logits_2_label(out_seg)
    hard_t_seg = logits_2_label(t_seg)
    convert_out_seg = convert_2_brats_dice_label(hard_out_seg)
    convert_t_seg = convert_2_brats_dice_label(hard_t_seg)

    hard_dice = dice_coe(convert_out_seg, convert_t_seg, axis=axis, batch=True)
    hard_iou = iou_coe(convert_out_seg, convert_t_seg, axis=axis, batch=False)

    WT = tf.reduce_mean(hard_dice[:, 0])
    TC = tf.reduce_mean(hard_dice[:, 1])
    ET = tf.reduce_mean(hard_dice[:, 2])

    # total_dice_loss = 1 - (WT+TC+ET)/3
    tf_weight = tf.constant(list_weight, soft_dice.dtype)
    total_dice_loss = 1 - tf.reduce_sum(soft_dice * tf_weight) / tf.reduce_sum(tf_weight)

    return total_dice_loss, hard_iou, WT, TC, ET


def validate_losses_v1(out_seg, t_seg, axis=(1, 2, 3), threshold=0.5):
    """
    WT = ED + ET + NET
    TC = ET + NET
    ET
    """
    # mixed precision processing
    if out_seg.dtype != t_seg.dtype:
        out_seg = tf.cast(out_seg, t_seg.dtype)

    # insert background label
    if t_seg.shape[-1] == 3:
        mask = tf.reduce_sum(t_seg, axis=-1, keepdims=True)
        inverse_mask = tf.logical_not(tf.cast(mask, tf.bool))
        float_inverse_mask = tf.cast(inverse_mask, dtype=t_seg.dtype)
        t_seg = tf.concat([float_inverse_mask, t_seg], axis=-1)

    soft_dice = dice_coe(out_seg, t_seg, axis=axis, batch=True)
    # soft_iou = iou_coe(out_seg, t_seg, axis=axis, batch=False)

    hard_out_seg = logits_2_label(out_seg)
    hard_t_seg = logits_2_label(t_seg)
    convert_out_seg = convert_2_brats_dice_label(hard_out_seg)
    convert_t_seg = convert_2_brats_dice_label(hard_t_seg)
    hard_dice = dice_coe(convert_out_seg, convert_t_seg, axis=axis, batch=True)
    hard_iou = iou_coe(convert_out_seg, convert_t_seg, axis=axis, batch=True)

    reduce_hard_iou = tf.reduce_mean(hard_iou)
    WT = tf.reduce_mean(hard_dice[:, 0])
    TC = tf.reduce_mean(hard_dice[:, 1])
    ET = tf.reduce_mean(hard_dice[:, 2])
    total_dice_loss = 1 - (WT + TC + ET) / 3

    return total_dice_loss, reduce_hard_iou, WT, TC, ET

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=1.):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    # reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(fl)
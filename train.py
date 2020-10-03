import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time, strftime
import os
import load_data
import model
import utils
import metrics
import checkpointing

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB ']= '0'
    tf.config.optimizer.set_jit(True)
    learning_rate = 0.00005
    EPOCHS = 20
    TOTAL_STEPS = 20000
    batch = 1
    val_batch = 1
    size = (112, 80)
    origin_size = (240, 240)
    pad = 0
    resize_method = 'resize'
    axis = (1, 2, 3)
    print_step = 20
    eval_step = 200
    save_step = 300
    epsilon = 1e-7
    list_weight = None
    mixed_precision = False
    is_eval = False
    summary_time = strftime('%Y%m%d_%H%M%S')
    # summary_time = '20200903_112349'
    summary_path = 'summaries/{}'.format(summary_time)
    std_path = None
    pre_scaling = True
    train_transform_mean_std_path = '/home/gettgod/BraTS/pickle/2020_train_landmarks_zprarms(2020_train_landmarks).pkl'
    train_transform_mean_std_path = None
    main_tfr_path = r'C:\BraTS\3d\train_2020_pre_scaling(2020_train_landmarks)'
    HGG_tfr = os.path.join(main_tfr_path, '*HGG*')
    LGG_tfr = os.path.join(main_tfr_path, '*LGG*')

    test_tfr_path = r'C:\BraTS\3d\valid_2020_pre_scaling(2020_train_landmarks)\*.tfrecord*'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(9200)])
                print(f'Setting {gpu} successfully')
        except RuntimeError as e:
            print(e)

    #利用資料夾紀錄參數
    store_params = 'batch_{}~lr_{}~dp_{}~gp_{}~total_steps_{}~l2_{}~is_sample_{}~size_{}_pad_{}~axis{}~weight_{}~resize_method_{}'.\
        format(batch, learning_rate, model.drop_prob, model.groups, TOTAL_STEPS, model.l2,
               model.sample, size, pad, axis, list_weight, resize_method).replace('(', '_').replace(')', '_')
    store_params_1 = 'transform_mean_std_path_{}'\
        .format(os.path.split(train_transform_mean_std_path)[-1] if train_transform_mean_std_path else train_transform_mean_std_path,)

    if not os.path.exists(os.path.join(summary_path, store_params)):
        os.makedirs(os.path.join(summary_path, store_params))
    if not os.path.exists(os.path.join(summary_path, store_params_1)):
        os.makedirs(os.path.join(summary_path, store_params_1))

    if mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)

    # cast tfr to dataset api
    with tf.name_scope('loader'):
        train_loader = load_data.tf_dataset(HGG_tfr,
                                            LGG_tfr,
                                            std_path,
                                            transform_mean_std_path=train_transform_mean_std_path,
                                            pre_scaling=pre_scaling,
                                            preprocessing=True,
                                            batch=batch,
                                            buffer_size=8,
                                            repeat=1,
                                            size=size,
                                            pad=pad,
                                            resize_method=resize_method)

        test_loader = load_data.tf_dataset(test_tfr_path,
                                           None,
                                           std_path,
                                           transform_mean_std_path=train_transform_mean_std_path,
                                           pre_scaling=pre_scaling,
                                           preprocessing=False,
                                           batch=val_batch,
                                           shuffle=False,
                                           repeat=1,
                                           size=size,
                                           resize_method=resize_method)

    with tf.name_scope('optimizer'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        if mixed_precision:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    net = model.UNet()

    # @checkpointing.checkpointable
    # def checkpointing_net(inputs):
    #     return net(inputs)

    @tf.function
    def train_step(x, y):
        with tf.name_scope('train'):
            with tf.GradientTape() as tape:
                pred = net(x)
                resized_pred = utils.batch_resize_images(pred, origin_size)
                resized_true = utils.batch_resize_images(y, origin_size)
                total_dice_loss, iou, WT, TC, ET = metrics.train_losses_v1(resized_pred, resized_true, axis=axis)
                regularizer_losses = net.losses
                total_regularizer_loss = tf.add_n(regularizer_losses)
                total_loss = total_dice_loss+total_regularizer_loss

            # gradients = tape.gradient(total_loss, net.variables)
            gradients = tape.gradient(total_loss, net.variables)
            optimizer.apply_gradients(zip(gradients, net.variables))

            return pred, total_dice_loss, iou, WT, TC, ET, total_regularizer_loss

    @tf.function
    def test_step(x, y, **kwargs):
        # resize_x = utils.resize_images(x, size=size)
        # resize_y = utils.resize_images(y, size=size)
        pred = net(x, training=False)
        return pred



    file_writter = tf.summary.create_file_writer(summary_path)
    tf.summary.trace_on()
    step = tf.Variable(0, trainable=False)
    epoch = tf.Variable(0, trainable=False)

    # saver
    ckpt = tf.train.Checkpoint(net=net, optimizer=optimizer, step=step, epoch=epoch)
    ckpt_manager = tf.train.CheckpointManager(ckpt, summary_path, 5)
    latest_path = ckpt_manager.latest_checkpoint

    if latest_path:
        print('-----------------------Restoring: {}-----------------------'.format(latest_path))
        ckpt.restore(latest_path)

    # start training
    t0 = time()
    n_step = 0
    for _ in range(EPOCHS):
        # data and labels generator
        train_generator = train_loader.generator()
        test_generator = test_loader.generator()

        epoch.assign_add(1)
        n_epoch = epoch.numpy()
        for (x, y, detail) in train_generator:
            # 防止過小batch造成梯度計算錯誤
            decay_learning_rate = utils.decay_lr_rate(learning_rate, n_step, TOTAL_STEPS, 1e-8)
            optimizer.lr.assign(decay_learning_rate)
            b = x.shape[0]
            if b == batch:
                pred, total_dice_loss, iou, WT, TC, ET, regularizer_losses = train_step(x, y)
                step.assign_add(1)
                n_step = step.numpy()

                # recording graph
                if n_step != 1 and n_step-1 == 0:
                    with file_writter.as_default():
                        print('-----------------------Saving CKPT-----------------------')
                        tf.summary.trace_export(summary_path, step=0)

                if n_step == 0 or n_step % print_step == 0:
                    speed = abs(batch*print_step/(time()-t0))
                    print('Epoch: {} Step: {} 1-Dice: {:.4f} L2: {:.4f} IOU: {:.4f} '
                          'Dice_WT: {:.4f}, Dice_TC: {:.4f}, Dice_ET: {:.4f}, Speed: {:.2f} images/sec'
                          .format(n_epoch, n_step, total_dice_loss, regularizer_losses, iou,
                                  WT, TC, ET, speed))
                    if mixed_precision:
                        print('Loss scale: {}'.format(policy.loss_scale))
                    t0 = time()

                    with file_writter.as_default():
                        with tf.name_scope('train'):
                            utils.images_summary_processing(x, y, pred, n_step, threshold=0.5)
                            tf.summary.scalar('lr_rate', optimizer.lr, n_step)
                            tf.summary.scalar('1-Dice', total_dice_loss, n_step)
                            tf.summary.scalar('L2', regularizer_losses, n_step)
                            tf.summary.scalar('iou', iou, n_step)
                            tf.summary.scalar('Dice_WT', WT, n_step)
                            tf.summary.scalar('Dice_TC', TC, n_step)
                            tf.summary.scalar('Dice_ET', ET, n_step)
                            tf.summary.scalar('Speed', speed, n_step)


                if n_step % save_step == 0:
                    ckpt.save(file_prefix=os.path.join(summary_path, str(n_step)))

                if n_step % eval_step == 0 and is_eval:
                    print('--------------------Evaluation--------------------\n')
                    name_list = []
                    lost_list = []
                    for val_i, (x, y, detail) in enumerate(tqdm(test_generator)):
                        pred = test_step(x, y)
                        name = detail[0, 1].numpy().decode()
                        name_list.append(name)
                        total_dice_loss, iou, WT, TC, ET = metrics.validate_losses_v1(pred, y)
                        lost_list.append(np.stack([total_dice_loss, iou, WT, TC, ET], axis=0))
                    if len(lost_list) < 2:
                        result_values = np.expand_dims(lost_list[0], axis=0)
                        mean_lost_array = lost_list[0]
                    else:
                        result_values = np.stack(lost_list, axis=0)
                        mean_lost_array = result_values.mean(axis=0)

                    print('1-Dice: {:.4f} IOU: {:.4f} '
                          'Dice_WT: {:.4f}, Dice_TC: {:.4f}, Dice_ET: {:.4f}'.
                          format(mean_lost_array[0], mean_lost_array[1],
                                 mean_lost_array[2], mean_lost_array[3], mean_lost_array[4]))

                    csv_path = os.path.join(summary_path, 'training_csv_results', 'step_'+str(step.numpy())+'.csv')
                    utils.to_excel(result_values, csv_path, name_list)
                    print('--------------------------------------------------')

                    with file_writter.as_default():
                        with tf.name_scope('validation'):
                            tf.summary.scalar('1-Dice', mean_lost_array[0], n_step)
                            tf.summary.scalar('iou', mean_lost_array[1], n_step)
                            tf.summary.scalar('Dice_WT', mean_lost_array[2], n_step)
                            tf.summary.scalar('Dice_TC', mean_lost_array[3], n_step)
                            tf.summary.scalar('Dice_ET', mean_lost_array[4], n_step)
            else:
                print('Skip this batch({})'.format(b))

    ckpt.save(file_prefix=os.path.join(summary_path, str(n_step)))


if __name__ == '__main__':
    main()
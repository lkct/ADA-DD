import argparse
import logging
import os
import struct

import mxnet as mx
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def get_mnist():
    def read_data(label_url, image_url):
        with open(label_url) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with open(image_url, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
                len(label), rows, cols)
            image = image.reshape(
                image.shape[0], 1, 28, 28).astype(np.float32)
            image = np.tile(image, (1, 3, 1, 1))
        return (label, image)

    path = './'
    (train_lbl, train_img) = read_data(
        path+'train-labels.idx1-ubyte', path+'train-images.idx3-ubyte')
    (test_lbl, test_img) = read_data(
        path+'t10k-labels.idx1-ubyte', path+'t10k-images.idx3-ubyte')
    path = './data/'
    (etrain_lbl, etrain_img) = read_data(
        path+'emnist-digits-train-labels.idx1-ubyte', path+'emnist-digits-train-images.idx3-ubyte')
    (etest_lbl, etest_img) = read_data(
        path+'emnist-digits-test-labels.idx1-ubyte', path+'emnist-digits-test-images.idx3-ubyte')

    img = np.concatenate([train_img, test_img, etrain_img, etest_img], axis=0)
    lbl = np.concatenate([train_lbl, test_lbl, etrain_lbl, etest_lbl], axis=0)
    return {'train_data': img, 'train_label': lbl}


# def lenet(num_classes=10):
#     data = mx.symbol.Variable('data')
#     # first conv
#     conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
#     relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
#     pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
#                               kernel=(2, 2), stride=(2, 2))
#     # second conv
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
#     relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
#     pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
#                               kernel=(2, 2), stride=(2, 2))
#     # first fullc
#     flatten = mx.symbol.Flatten(data=pool2)
#     fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500, name='fc6')
#     relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
#     # second fullc
#     fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=120, name='fc7')
#     relu4 = mx.symbol.Activation(data=fc2, act_type="relu")
#     # third fullc
#     fc3 = mx.symbol.FullyConnected(data=relu4, num_hidden=num_classes)
#     # loss
#     return mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
def residual_unit(data, num_filter, stride, dim_match, bn_mom=0.9):
    bn1 = mx.sym.BatchNorm(data, fix_gamma=False, momentum=bn_mom, eps=2e-5)
    relu1 = mx.sym.Activation(bn1, act_type='relu')
    conv1 = mx.sym.Convolution(relu1, num_filter=num_filter, kernel=(
        3, 3), stride=stride, pad=(1, 1), no_bias=True)
    bn2 = mx.sym.BatchNorm(conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5)
    relu2 = mx.sym.Activation(bn2, act_type='relu')
    conv2 = mx.sym.Convolution(relu2, num_filter=num_filter, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(
            relu1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True)
    return conv2 + shortcut


def lenet(num_classes=10, bn_mom=0.9):
    data = mx.sym.Variable('data')
    data = mx.sym.BatchNorm(data, fix_gamma=True, eps=2e-5, momentum=bn_mom)

    conv1 = mx.sym.Convolution(data, num_filter=16, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
    body = residual_unit(conv1, 16, (1, 1), False)
    body = residual_unit(body, 16, (1, 1), True)
    body = residual_unit(body, 32, (2, 2), False)
    body = residual_unit(body, 32, (1, 1), True)
    body = residual_unit(body, 64, (2, 2), False)
    body = residual_unit(body, 64, (1, 1), True)
    bn1 = mx.sym.BatchNorm(body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    relu1 = mx.sym.Activation(bn1, act_type='relu')

    flat = mx.symbol.Flatten(relu1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=500, name='fc6')
    relu2 = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=relu2, num_hidden=120, name='fc7')
    relu3 = mx.symbol.Activation(data=fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=relu3, num_hidden=num_classes)
    return mx.symbol.SoftmaxOutput(data=fc3, name='softmax')


def main():
    symbol = lenet(num_classes=args.num_classes)
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    begin_epoch = args.model_load_epoch if args.retrain else 0
    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/lenet-mnist-{}".format(kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    speedometer = mx.callback.Speedometer(args.batch_size, args.frequent)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, args.model_load_epoch)
    mnist = get_mnist()
    train = mx.io.NDArrayIter(
        data=mnist['train_data'], label=mnist['train_label'], batch_size=args.batch_size, shuffle=True)
    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join("data", "rec.rec"),
        label_width=1,
        data_shape=(3, 28, 28),
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        batch_size=args.batch_size,
    )
    model = mx.mod.Module(symbol=symbol, context=devs)
    model.fit(
        train_data=train,
        eval_data=val,
        epoch_end_callback=checkpoint,
        batch_end_callback=speedometer,
        kvstore=kv,
        optimizer='nag',
        optimizer_params=(('learning_rate', args.lr), ('lr_scheduler', multi_factor_scheduler(
            begin_epoch, epoch_size, step=[10, 20]))),
        initializer=mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2),
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=begin_epoch,
        num_epoch=args.end_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training modified lenet-5")
    parser.add_argument('--gpus', type=str, default=None,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initialization learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the class number of your task')
    parser.add_argument('--num-examples', type=int, default=350000,  # 60000,
                        help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--end-epoch', type=int, default=30,
                        help='training ends at this num of epoch')
    parser.add_argument('--frequent', type=int, default=50,
                        help='frequency of logging')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='true means continue training')
    args = parser.parse_args()
    logging.info(args)
    main()

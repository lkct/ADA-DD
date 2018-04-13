import argparse
import logging
import os
import struct

import mxnet as mx
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)


<<<<<<< HEAD
def lenet(num_class=10):
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20, no_bias=True)
=======
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
                image.shape[0], 1, 28, 28).astype(np.float32)/255
        return (label, image)

    path = './'
    (train_lbl, train_img) = read_data(
        path+'train-labels.idx1-ubyte', path+'train-images.idx3-ubyte')
    (test_lbl, test_img) = read_data(
        path+'t10k-labels.idx1-ubyte', path+'t10k-images.idx3-ubyte')
    return {'train_data': train_img, 'train_label': train_lbl,
            'test_data': test_img, 'test_label': test_lbl}


def lenet(num_class=10):
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, kernel=(
        5, 5), num_filter=20, no_bias=True)
>>>>>>> upstream/master
    data = mx.sym.BatchNorm(data=data, eps=1e-5, fix_gamma=False)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Pooling(data=data, kernel=(
        2, 2), pool_type='max', stride=(2, 2))
<<<<<<< HEAD
    data = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=50, no_bias=True)
=======
    data = mx.sym.Convolution(data=data, kernel=(
        5, 5), num_filter=50, no_bias=True)
>>>>>>> upstream/master
    data = mx.sym.BatchNorm(data=data, eps=1e-5, fix_gamma=False)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Pooling(data=data, kernel=(
        2, 2), pool_type='max', stride=(2, 2))
    data = mx.sym.Flatten(data=data)
    data = mx.sym.FullyConnected(data=data, num_hidden=1000)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.FullyConnected(data=data, num_hidden=num_class)

    return mx.sym.SoftmaxOutput(data=data, name='softmax')

def main():
    symbol = lenet(num_class=args.num_classes)
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
    val = mx.io.NDArrayIter(
        data=mnist['test_data'], label=mnist['test_label'], batch_size=args.batch_size)
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
    parser.add_argument('--num-examples', type=int, default=60000,
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

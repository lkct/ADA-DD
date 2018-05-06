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


def get_mnist(lbl, img):
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
    (test_lbl, test_img) = read_data(path+lbl, path+img)
    return {'test_data': test_img, 'test_label': test_lbl}


def main():
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    model_prefix = "model/lenet-mnist-{}".format(kv.rank)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.model_load_epoch)
    mnist = get_mnist(args.label, args.image)
    val = mx.io.NDArrayIter(
        data=mnist['test_data'], label=mnist['test_label'], batch_size=args.batch_size)
    model = mx.mod.Module(symbol=symbol, context=devs)
    model.bind(data_shapes=val.provide_data, label_shapes=val.provide_label,
               for_training=False, grad_req='null')
    model.set_params(arg_params, aux_params)
    pred = model.predict(val).asnumpy()
    score = model.score(val, ['acc'])
    print score[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training modified lenet-5")
    parser.add_argument('--gpus', type=str, default=None,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=30,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--image', type=str, default='emnist-mnist-train-images-idx3-ubyte',
                        help='the kvstore type')
    parser.add_argument('--label', type=str, default='emnist-mnist-train-labels-idx1-ubyte',
                        help='the kvstore type')
    args = parser.parse_args()
    logging.info(args)
    main()

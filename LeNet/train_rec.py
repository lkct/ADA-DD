import argparse
import logging
import os

import mxnet as mx

from lenet import lenet

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():
    symbol = lenet(num_classes=args.num_classes)
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/lenet-mnist-{}".format(kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, args.model_load_epoch)
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join("data", "rec.rec"),
        label_width         = 1,
        data_shape          = (3, 28, 28),
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank,
        batch_size          = args.batch_size,
    )
    # val = mx.io.ImageRecordIter(
    #     path_imgrec         = os.path.join(args.data_dir, "val_256_q90.rec"),
    #     label_width         = 1,
    #     data_shape          = (3, 224, 224),
    #     num_parts           = kv.num_workers,
    #     part_index          = kv.rank,
    #     batch_size          = args.batch_size,
    # )
    model = mx.mod.Module(
        symbol              = symbol,
        data_names          = ('data', ),
        label_names         = ('softmax_label', ),
        context             = devs,
    )
    model.fit(
        train_data          = train,
        # eval_data           = val,
        eval_metric         = ['acc'],
        epoch_end_callback  = checkpoint,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.frequent),
        kvstore             = kv,
        optimizer           = 'nag',
        optimizer_params=(('learning_rate', args.lr), ('lr_scheduler', multi_factor_scheduler(
            begin_epoch, epoch_size, step=[10, 20]))),
        initializer=mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = begin_epoch,
        num_epoch           = args.end_epoch,
    )
    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training modified lenet-5")
    parser.add_argument('--gpus', type=str, default=None,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initialization learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the class number of your task')
    parser.add_argument('--num-examples', type=int, default=8747,
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
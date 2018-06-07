import numpy as np
import cv2
import mxnet as mx
import argparse

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def main(imgs):
    ctx = mx.gpu(0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('../lenet-mnist-0', 7)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    arg_params["data"] = mx.nd.array(imgs, ctx)
    arg_params["softmax_label"] = mx.nd.empty((imgs.shape[0],), ctx)
    exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)

    prob = exe.outputs[0].asnumpy()
    pred = np.argmax(prob, axis=1)
    prob = np.float32([prob[i, pred[i]] for i in range(pred.shape[0])])
 
    return pred, prob

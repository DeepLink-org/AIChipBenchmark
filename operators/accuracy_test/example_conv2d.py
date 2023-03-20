import os, sys
import torch
import torch.nn as nn
from utils import fix_rand, result_diff, turn_off_low_precision

fix_rand()
turn_off_low_precision()


def allclose(t1, t2, msg=""):
    t1 = t1.cpu()
    t2 = t2.cpu()
    result_diff(t1, t2, msg=msg)


def gen_data(model, path):
    model.zero_grad()
    input = torch.rand(2, 16, 50, 40).cuda().half()
    input.requires_grad = True
    output = model(input)
    out_grad = torch.ones_like(output)
    output.backward(out_grad)

    torch.save(input, os.path.join(path, "input.pt"))
    torch.save(output, os.path.join(path, "output.pt"))
    torch.save(model.state_dict(), os.path.join(path, "state_dict.pt"))
    torch.save(input.grad, os.path.join(path, "input_grad.pt"))
    torch.save(model.weight.grad, os.path.join(path, "weight_grad.pt"))
    torch.save(model.bias.grad, os.path.join(path, "bias_grad.pt"))


def verify(model, path):

    input = torch.load(os.path.join(path, "input.pt")).half().requires_grad_(True)
    gt_output = torch.load(os.path.join(path, "output.pt"))
    state_dict = torch.load(os.path.join(path, "state_dict.pt"))
    model.load_state_dict(state_dict)
    model = model.half()
    gt_input_grad = torch.load(os.path.join(path, "input_grad.pt"))
    gt_weight_grad = torch.load(os.path.join(path, "weight_grad.pt"))
    gt_bias_grad = torch.load(os.path.join(path, "bias_grad.pt"))
    model.zero_grad()
    cur_output = model(input)
    # cur_output.sum().backward()
    out_grad = torch.ones_like(cur_output)
    cur_output.backward(out_grad)
    # compare
    allclose(cur_output, gt_output, "fwd output ")
    allclose(model.weight.grad, gt_weight_grad, "model.weight.grad ")
    allclose(model.bias.grad, gt_bias_grad, "model.bias.grad ")
    allclose(input.grad, gt_input_grad, "input.grad ")


if __name__ == "__main__":
    conv2d = nn.Conv2d(
        16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), groups=1, bias=True, padding_mode="zeros"
    ).cuda()
    path = "./data"
    if sys.argv[1] == "1":
        gen_data(conv2d, path)
    else:
        verify(conv2d, path)

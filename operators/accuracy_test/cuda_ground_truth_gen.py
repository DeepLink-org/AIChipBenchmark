import torch
import torch.nn as nn
import os
import sys
import json

from op_config import configs, device
from utils import to_list, output_to_list, fix_rand, try_deterministic, turn_off_low_precision
from op_conv_config import get_conv_config


def fix_weight(model):
    if isinstance(model, torch.nn.Module):
        for val in model.parameters():
            val.data = torch.ones_like(val) + 0.11


class OpTestDataGenerator(object):
    def __init__(self, config, op_name, dst_dir="./"):
        self.module = config[0]
        # info could be shapes or tensors
        input_info = config[1]
        self.params = None
        if len(config) >= 3:
            self.params = config[2]
        self.kwargs = None
        if len(config) >= 4:
            self.kwargs = config[3]

        self.input_info = to_list(input_info)
        self.base_dir = os.path.join(dst_dir, op_name)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.fp16_path = os.path.join(self.base_dir, "fp16")
        self.fp32_path = os.path.join(self.base_dir, "fp32")
        self.run_backward = False
        self.fp16_support = True
        
        print("module:", str(self.module), '\tinput_info:', str(self.input_info))

    def gen_inputs(self, input_info, data_type):
        """
        Description: Generate input data based on information about shape and data type.

        Param:
            input_info (list): the shape of input data
            data_type (torch.Tensor): torch.float32 or torch.float16
        return: list
            the generated input data
        """
        ret = []
        input_shapes = []
        for info in input_info:
            if isinstance(info, torch.Tensor):
                rq_grad = info.requires_grad

                info = info.to(device).detach().requires_grad_(rq_grad)
                # int tensors should not be transfered to float
                if info.dtype is torch.float16 or info.dtype is torch.float32:
                    ret.append(info.to(data_type).detach().requires_grad_(rq_grad))
                else:
                    ret.append(info)
                input_shapes.append(list(info.shape))
            else:
                input_shapes.append(info)
                x = torch.rand(info, dtype=data_type, device=device)
                x.requires_grad = True
                ret.append(x)
        return ret

    def get_inputshapes(self):
        input_shapes = []
        for info in self.input_info:
            if isinstance(info, torch.Tensor):
                input_shapes.append(list(info.shape))
            else:
                input_shapes.append(info)
        return input_shapes

    def save_all(self, dir, model, inputs, outputs, out_grads):
        """
        Save the input/output and the gradient of input/output.

        """
        outputs = output_to_list(outputs)
        out_grads = to_list(out_grads)

        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = dir + "/"
        if isinstance(model, torch.nn.Module):
            if model.state_dict():
                torch.save(model.state_dict(), dir + "state_dict.pt")
            for idx, val in enumerate(model.parameters()):
                grad_name = "paramGrad_" + str(idx) + ".pt"
                torch.save(val.grad, dir + grad_name)
        for idx, val in enumerate(inputs):
            input_name = "input_" + str(idx) + ".pt"
            torch.save(val, dir + input_name)
            if val.grad is not None:
                input_grad_name = "inputGrad_" + str(idx) + ".pt"
                torch.save(val.grad, dir + input_grad_name)

        for idx, val in enumerate(outputs):
            output_name = "output_" + str(idx) + ".pt"
            torch.save(val, dir + output_name)
        for idx, val in enumerate(out_grads):
            output_name = "outputGrad_" + str(idx) + ".pt"
            torch.save(val, dir + output_name)

    def run_with_model_and_inputs(self, model, inputs, dir):
        """
        Run the model and save the input/output and the gradient of input/output.

        """
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.zero_grad()
        outs = None
        if self.params is None:
            outs = model(*inputs)
        else:
            if isinstance(self.params, dict):
                outs = model(*inputs, **self.params)
            else:
                outs = model(*inputs, *self.params)

        out_grads = []

        if self.kwargs and (self.kwargs.get("backward") is False):
            print("omit backward")
        else:
            input_grad = False
            for input in inputs:
                if input.requires_grad:
                    input_grad = True

            if input_grad:
                if isinstance(outs, torch.Tensor):
                    if outs.dtype is torch.float32 or outs.dtype is torch.float16:
                        out_grad = torch.ones_like(outs)
                        out_grads = out_grad
                        outs.backward(out_grad)
                        self.run_backward = True
                elif isinstance(outs, list) or isinstance(outs, tuple):
                    for out in outs:
                        if out.dtype is torch.float32 or out.dtype is torch.float16:
                            out_grad = None
                            out_grad = torch.ones_like(out)
                            out_grads.append(out_grad)
                            out.backward(out_grad, retain_graph=True)
                            self.run_backward = True

        self.save_all(dir, model, inputs, outs, out_grads)

    def run_fp16(self):
        """
        Generate the input data and run the model.

        """
        fp16_model = self.module
        if isinstance(self.module, torch.nn.Module):
            fp16_model = self.module.half()
        fp16_inputs = self.gen_inputs(self.input_info, torch.float16)

        self.run_with_model_and_inputs(fp16_model, fp16_inputs, self.fp16_path)

    def run_fp32(self):
        fp32_inputs = self.gen_inputs(self.input_info, torch.float32)
        self.run_with_model_and_inputs(self.module, fp32_inputs, self.fp32_path)

    @try_deterministic
    def run_all(self):
        self.run_fp32()
        try:
            self.run_fp16()
        except Exception as e:
            print(e)
            self.fp16_support = False


all_info = dict()

if __name__ == "__main__":
    out_path = "./tmp_data"
    case_name = None
    if len(sys.argv) >= 2:
        out_path = sys.argv[1]
    if len(sys.argv) >= 3:
        case_name = sys.argv[2]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # fix random numbers and use deterministic algorithms
    fix_rand()
    # disable allow tf32
    turn_off_low_precision()
    print(f"Using device {device}")
    json_file = out_path + "/info.json"
    config = configs
    if case_name and case_name != 'CONV':
        tester = OpTestDataGenerator(config[case_name], case_name, out_path)
        tester.run_all()
        exit()
    else:
        if case_name == 'CONV':
            config = get_conv_config()
        for name, cfg in config.items():
            tester = OpTestDataGenerator(cfg, name, out_path)
            tester.run_all()

            args = []
            if tester.params is not None:
                if isinstance(tester.params, dict):
                    for k, v in tester.params.items():
                        if isinstance(v, torch.Tensor):
                            pass
                        elif isinstance(v, tuple):
                            ret = []
                            for t in v:
                                if isinstance(t, torch.Tensor):
                                    pass
                                else:
                                    ret.append(t)
                            args.append({k: ret})
                        else:
                            args.append({k: v})
                else:
                    for p in tester.params:
                        if p is None:
                            continue
                        if isinstance(p, torch.Tensor):
                            pass
                        elif isinstance(p, dict):
                            for k, v in p.items():
                                if isinstance(v, torch.Tensor):
                                    pass
                                else:
                                    args.append({k: v})
                        else:
                            args.append(p)

            code = str(cfg[0])
            if isinstance(cfg[0], torch.nn.Module):
                code = "nn." + str(cfg[0])
            all_info[name] = dict(
                code=code,
                input_shapes=tester.get_inputshapes(),
                need_backward=tester.run_backward,
                args=args,
                support_fp16=tester.fp16_support,
            )

        with open(json_file, "w") as jsonf:
            json.dump(all_info, jsonf)
    print("INFO:successfully generate ground_truth_input.")

import torch
import os
import sys
import json
import csv
import pandas
import logging
from op_config import configs, device
from op_conv_config import get_conv_config
from utils import to_list, output_to_list, result_diff, fix_rand, try_deterministic, turn_off_low_precision

logger = logging.getLogger("OpValidater")
LOGLEVEL = os.environ.get("PYLOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)


class CudaOpValidater(object):
    def __init__(self, config, op_name, data_dir):
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
        self.input_num = len(self.input_info)
        self.base_dir = os.path.join(data_dir, op_name)
        self.op_name = op_name

    # runs accuracy validation with given datatype
    def run(self, model, dt="fp32", abs_thresh=1e-8, relative_thresh=1e-5) -> bool:
        logger.info(f"running op={self.op_name} dtype={dt} ...")
        passed = True
        data_dir = os.path.join(self.base_dir, dt)
        if not os.path.exists(data_dir):
            return True
        inputs = []
        for i in range(0, self.input_num):
            print(os.path.join(data_dir, "input_{0}.pt".format(str(i))))
            loaded_input = torch.load(os.path.join(data_dir, "input_{0}.pt".format(str(i)))).to(device)
            # loaded_input contains requires_grad info
            loaded_requires_grad = loaded_input.requires_grad
            input = loaded_input.detach()
            input.requires_grad = loaded_requires_grad
            inputs.append(input)

        # load state_dict for Modules
        if isinstance(model, torch.nn.Module) and model.state_dict():
            state_dict_path = os.path.join(data_dir, "state_dict.pt")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path)
                model.load_state_dict(state_dict)
            else:
                logger.info(self.module, " has no state_dict file to load")

        if isinstance(model, torch.nn.Module):
            model.zero_grad()
            model = model.to(device)

        # run forward
        outs = None
        if self.params is None:
            outs = model(*inputs)
        else:
            if isinstance(self.params, dict):
                outs = model(*inputs, **self.params)
            else:
                outs = model(*inputs, *self.params)

        outs = output_to_list(outs)

        for i in range(0, len(outs)):
            gt_out = torch.load(os.path.join(data_dir, "output_{0}.pt".format(str(i))))
            # compare outs
            if not result_diff(outs[i], gt_out, abs_thresh, relative_thresh, msg=f"{self.op_name} forward output {i} "):
                passed = False

            # do backward
            gt_out_grad_file = os.path.join(data_dir, "outputGrad_{0}.pt".format(str(i)))
            if os.path.exists(gt_out_grad_file):
                gt_out_grad = torch.load(gt_out_grad_file).to(device)
                outs[i].backward(gt_out_grad, retain_graph=True)

        for i, val in enumerate(inputs):
            # compare input_grad
            if val.requires_grad:
                path = os.path.join(data_dir, "inputGrad_{0}.pt".format(str(i)))
                if not os.path.exists(path):
                    continue
                gt_input_grad = torch.load(path)
                if not result_diff(
                    val.grad, gt_input_grad, abs_thresh, relative_thresh, msg=f"{self.op_name} input grad {i} "
                ):
                    passed = False

        # compare param_grad
        if isinstance(model, torch.nn.Module):
            for i, val in enumerate(model.parameters()):
                gt_param_grad = torch.load(os.path.join(data_dir, "paramGrad_{0}.pt".format(str(i))))
                if not result_diff(
                    val.grad, gt_param_grad, abs_thresh, relative_thresh, msg=f"{self.op_name} param grad {i} "
                ):
                    passed = False

        return passed

    @try_deterministic
    def run_all(self):
        # 1. run float32 validation
        self.fp32_result = self.run(self.module, "fp32", abs_thresh=1e-5, relative_thresh=1e-4)

        # 2. run float16 validation
        fp16_model = self.module
        if isinstance(self.module, torch.nn.Module):
            fp16_model = self.module.half()
        self.fp16_result = False
        try:
            self.fp16_result = self.run(fp16_model, "fp16", abs_thresh=1e-3, relative_thresh=1e-3)
        except Exception as e:
            logger.warning(e)


all_info = dict()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: validate.py gt_data_path path/to/result")
        exit(1)
    gt_path = sys.argv[1]
    if not os.path.exists(gt_path):
        logger.info("Invalid gt_data_path!")
        exit(1)
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # fix randomness and do not use TF32
    fix_rand()
    turn_off_low_precision()

    logger.warning(f"Using device {device}")
    config = configs

    # test a single op
    case = None
    if len(sys.argv) == 4:
        case = sys.argv[3]
    if case and case != 'CONV':
        tester = CudaOpValidater(config[case], case, gt_path)
        tester.run_all()
        exit(0)
    # test all op
    else:
        if case == 'CONV':
            config = get_conv_config()
        for name, cfg in config.items():
            tester = CudaOpValidater(cfg, name, gt_path)
            tester.run_all()
            all_info[name] = dict(passed=(tester.fp32_result and tester.fp16_result))
        #  save result to json and csv
        json_file = output_path + "/cuda_val_result.json"
        with open(json_file, "w") as jsonf:
            json.dump(all_info, jsonf)

        csv_file = output_path + "/cuda_val_result.csv"
        oplist = list(all_info.keys())
        pd_data = dict(
            NO=[i for i in range(0, len(oplist))], op=oplist, validationResult=[val["passed"] for val in all_info.values()]
        )
        df = pandas.DataFrame.from_dict(pd_data)
        df.to_csv(csv_file, index=False)
        print("INFO:successfully validate")

import math
import torch
import transformers
import torch.nn.functional as F
from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

        # TODO: Implement lora left and right weights
        self.lora_right_weight = torch.nn.Parameter(torch.zeros((lora_dim, weight.size(1))))
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((weight.size(0), lora_dim)))
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        torch.nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_left_weight)
        ##################################

    def forward(self, input):
        # Print shapes
        # TODO: Implement the forward function
        original_output = F.linear(input, self.weight, self.bias)
        # LoRA transformation
        lora_output = F.linear(F.linear(input, self.lora_right_weight), self.lora_left_weight) * self.lora_scaling
        # Combine original and LoRA outputs
        return original_output + lora_output
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model: torch.nn.Module):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    return model
    ##############################################################################

def get_lora_state_dict(model: torch.nn.Module):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_right_weight' in name or 'lora_left_weight' in name:
            lora_state_dict[name] = param.data
    return lora_state_dict
    ########################################################
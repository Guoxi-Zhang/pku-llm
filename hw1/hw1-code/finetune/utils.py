import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def log(f, msg):
    f.write(msg + '\n')
    print(msg)


def plot_learning_curve(train_data, output_dir):
    train_steps = train_data["train_steps"]
    train_loss = train_data["train_loss"]
    eval_steps = train_data["eval_steps"]
    eval_loss = train_data["eval_loss"]
    plt.plot(train_steps, train_loss, label="train")
    plt.plot(eval_steps, eval_loss, label="eval")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(output_dir + "/learning_curve.png")
    plt.close()


# The following code is adapted from DeepSpeed's helper.py
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/compression/helper.py
def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


class LoraLossCurveData:
    data_file_name = 'train_data.json'
    def __init__(self, path, lora_rank=1):
        self.path = path
        self.lora_rank = lora_rank
        self.train_steps = []
        self.train_loss = []
        self.eval_steps = []
        self.eval_loss = []
    
    def load_data(self):
        with open(self.path + '/' + self.data_file_name, 'r') as f:
            data = json.load(f)
        self.train_steps = data['train_steps']
        self.train_loss = data['train_loss']
        self.eval_steps = data['eval_steps']
        self.eval_loss = data['eval_loss']

def draw_loss_curve(lora_loss_curve_data_list:list[LoraLossCurveData], output_dir):
    plt.figure()
    for lora_loss_curve_data in lora_loss_curve_data_list:
        label=f"lora_rank_{lora_loss_curve_data.lora_rank}" if lora_loss_curve_data.lora_rank > 0 else "no lora"
        plt.plot(lora_loss_curve_data.eval_steps, lora_loss_curve_data.eval_loss, label=label)
    plt.xlabel("steps")
    plt.ylabel("eval loss")
    # 将lora rank作为legend
    plt.legend()
    plt.savefig(output_dir + "/lora_loss_curve.png")
    plt.close()

if __name__ == '__main__':
    lora_loss_curve_data_list = []
    lora_loss_curve_path_list = [
        './results/gpt2-alpaca_1-20241117-200553',
        './results/gpt2-alpaca_lora_1_32-20241216-184957', 
        './results/gpt2-alpaca_lora_2_32-20241216-195605',
        './results/gpt2-alpaca_lora_4_32-20241216-210737',
        './results/gpt2-alpaca_lora_8_32-20241216-171137',
        './results/gpt2-alpaca_lora_16_32-20241216-215609',
        './results/gpt2-alpaca_lora_32_32-20241216-225948',
    ]
    lora_ranks = [0, 1, 2, 4, 8, 16, 32]
    for i in range(len(lora_loss_curve_path_list)):
        lora_loss_curve_data = LoraLossCurveData(lora_loss_curve_path_list[i], lora_ranks[i])
        lora_loss_curve_data.load_data()
        lora_loss_curve_data_list.append(lora_loss_curve_data)
    draw_loss_curve(lora_loss_curve_data_list, './results')
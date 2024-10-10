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
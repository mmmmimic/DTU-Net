import torch

class Logger:
    def __init__(self, log_dir, mode='a'):
        super().__init__()
        assert mode in ['w', 'a'], f"mode {mode} is neither 'w' nor 'a'"
        self.mode = mode
        self.log_dir = log_dir

    def fprint(self, log):
        log = str(log) if not isinstance(log, str) else log
        print(log)
        with open(self.log_dir, mode=self.mode) as f:        
            f.write(log)
            f.write("\n")


def to_gpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        cuda_dict = {}
        for k, v in tensor.items():
            cuda_dict['k'] = v.to(device)
        del tensor
        return cuda_dict
    else:
        raise NotImplementedError()

def to_numpy(tensor):
    return tensor.squeeze().cpu().numpy()
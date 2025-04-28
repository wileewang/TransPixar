import torch

# Simple wrapper for use with gradient release. Grad hooks do the optimizer steps, so this no-ops
# the step() and zero_grad() methods. It also handles state_dict.
class GradientReleaseOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        ret = []
        for opt in self.optimizers:
            ret.extend(opt.param_groups)
        return ret

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, sd in state_dict.items():
            self.optimizers[i].load_state_dict(sd)

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass
import torch
import deepspeed.comm.comm as dist


class EMA:
    def __init__(self, parameters, decay=0.999, device=None, sync_every_step=False):
        """
        parameters: iterable of model parameters (typically requires_grad=True)
        decay: EMA decay factor (closer to 1.0 means slower update)
        device: if set (e.g., "cpu"), store EMA shadow weights on that device
        sync_every_step: if True, sync EMA shadow across all GPUs after update (usually unnecessary)
        """
        self.decay = decay
        self.device = device
        self.sync_every_step = sync_every_step
        self.shadow = {}
        self.backup = {}

        self._register(parameters)

    def _register(self, parameters):
        for p in parameters:
            if p.requires_grad:
                data = p.detach().clone()
                if self.device:
                    data = data.to(self.device)
                self.shadow[p] = data

    @torch.no_grad()
    def update(self):
        for p in self.shadow:
            if not p.requires_grad:
                continue
            current = p.detach()
            if self.device:
                current = current.to(self.device)
            self.shadow[p].mul_(self.decay).add_(current, alpha=1.0 - self.decay)

        if self.sync_every_step:
            self._sync_shadow()

    @torch.no_grad()
    def apply_shadow(self):
        """Replace model params with EMA weights (typically before evaluation)."""
        self.backup = {}
        for p in self.shadow:
            self.backup[p] = p.data.clone()
            shadow_data = self.shadow[p]
            if self.device:
                shadow_data = shadow_data.to(p.device)
            p.data.copy_(shadow_data)

    @torch.no_grad()
    def restore(self):
        """Restore original weights after evaluation."""
        for p in self.backup:
            p.data.copy_(self.backup[p])
        self.backup = {}

    @torch.no_grad()
    def _sync_shadow(self, src=0):
        """Broadcast EMA weights from src rank to all other ranks."""
        if not dist.is_initialized():
            return
        for p in self.shadow:
            t = self.shadow[p]
            if self.device:
                t = t.to(p.device)
            dist.broadcast(t, src=src)
            if self.device:
                self.shadow[p] = t.to(self.device)
            else:
                self.shadow[p] = t

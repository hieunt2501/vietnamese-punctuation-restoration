import torch

from packaging import version

v = version.parse
PyTorch_over_1_6 = v(torch.__version__) >= v('1.6')


# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class MixedPrecisionManager:
    def __init__(self, activated):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer, scheduler, max_grad_norm):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scale_before = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scale_after = self.scaler.get_scale()

            if scale_before <= scale_after:
                scheduler.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

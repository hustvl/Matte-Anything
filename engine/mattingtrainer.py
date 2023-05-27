from detectron2.engine import AMPTrainer
import torch
import time

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class MattingTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        super().__init__(model, data_loader, optimizer, grad_scaler=None)
        self.data_loader_iter = iter(cycle(self.data_loader))

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        #matting pass
        start = time.perf_counter()        
        data = next(self.data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
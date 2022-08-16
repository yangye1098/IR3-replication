import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time
from datetime import timedelta
import numpy as np
from torchvision.utils import make_grid
import math
import cv2

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # or debug purpose
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        cfg_trainer = config['trainer']
        self.n_valid_data_batch = cfg_trainer.get('n_valid_data_batch', 2)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # get log step
        self.log_step = cfg_trainer.get('log_step', 100)
        self.max_grad_norm = cfg_trainer.get('max_grad_norm', 1.0)

        # only loss for train
        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # audio sample dir
        sample_path = config.save_dir / 'samples'
        sample_path.mkdir(parents=True, exist_ok=True)

        self.target_path = sample_path / 'target'
        self.output_path = sample_path / 'output'
        self.condition_path = sample_path / 'condition'
        self.target_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.condition_path.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.epoch_start = time.time()
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (target, condition, _) in enumerate(self.data_loader):
            target, condition = target.to(self.device), condition.to(self.device)
            self.optimizer.zero_grad()
            output, noise = self.model(target, condition)
            # use noise in the loss function instead of target (y_0)
            loss = self.criterion(output, noise)

            loss.backward()
            #grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if batch_idx>0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation and (epoch % self.valid_period == 0):
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.logger.debug('')
        self.logger.debug('Valid Epoch: {} started at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (target, condition, _) in enumerate(self.valid_data_loader):
                if batch_idx >= self.n_valid_data_batch:
                    break

                target, condition = target.to(self.device), condition.to(self.device)

                # infer from noisy conditional input only
                output = self.model.infer(condition)
                loss = self.criterion(output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                output_img = tensor2img(output)  # uint8
                target_img = tensor2img(target)  # uint8
                condition_img = tensor2img(condition)  # uint8
                # save the validation output

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output_img, target_img))

                save_img(
                    target_img, self.target_path/'b{}.png'.format(batch_idx))
                save_img(
                    condition_img, self.condition_path/'b{}.png'.format(batch_idx))
                save_img(
                    output_img, self.output_path/'b{}.png'.format(batch_idx))

        self.logger.debug('\nValid Epoch: {} finished at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        lapsed = time.time() - self.epoch_start
        base = '[{}/{} | {:.0f}s/{}, ({:.0f}%), ]'
        current = batch_idx
        total = self.len_epoch

        time_left = lapsed * ((total/current) - 1)
        time_left = timedelta(seconds=time_left)
        return base.format(current, total, lapsed, time_left, 100.0 * current / total)

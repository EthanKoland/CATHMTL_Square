import os
import time

import torch
import torch.optim as optim

from models.network import ODSNet
from models.networks import *
from models.losses import lossManager
from utils.log_integrator import Integrator
from utils.image_saver import pool_pairs


class Trainer:

    def __init__(self, config, logger=None, save_path=None):
        self.config = config

        self.model = self.get_model().cuda()
        self.criterion = lossManager(config)

        # Set up logger
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            if self.logger is not None:
                self.logger.log_string(
                    'model_size', str(sum([param.nelement() for param in self.model.parameters()]) / 1000 / 1000)
                )
        self.train_integrator = Integrator(self.logger)

        self.train()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])

        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        output = self.model(data['img'])

        if self._do_log or self._is_train:
            losses = self.criterion(output, data, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.log_image_interval == 0 and it != 0:
                        if self.logger is not None:
                            # images = {**data, **output}
                            images = {'input': data, 'output': output}
                            size = (384, 384)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size), it)

        if self._is_train:
            if (it) % self.log_text_interval == 0 and it != 0:
                if self.logger is not None:
                    self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time() - self.last_time) / self.log_text_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if (it) % self.save_network_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save_network(it)

            if it % self.save_checkpoint_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        losses['total_loss'].backward()
        self.optimizer.step()

        self.scheduler.step()

    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = {
            'it': it,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:0'
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:0'
        self.model.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states,and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.model.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:0'
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def get_model(self):
        if self.config['arch'] == 'unet':
            return UNet(in_channels=1, out_channels=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'unet1':
            return U_Net(in_channels=1, out_channels=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'unetplus':
            return ResNet34UnetPlus(num_channels=1, num_class=self.config['num_classes'])
        elif self.config['arch'] == 'attunet':
            return AttU_Net(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunet_v1':
            return CMUNet(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunet_v2':
            return CMUNetv2_CM(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunext':
            return CMUNeXt(input_channel=1, num_classes=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'transunet':
            return TransUnet(img_ch=1, output_ch=self.config['num_classes'])
        elif self.config['arch'] == 'odsnet':
            return ODSNet(in_channels=3, heads=self.config['heads'])
        else:
            raise NotImplementedError

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.model.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.model.eval()
        return self

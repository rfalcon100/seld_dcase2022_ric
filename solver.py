import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

from utils import GradualWarmupScheduler
from models.crnn import CRNN10, CRNN


class Solver(object):
    def __init__(self, config, tensorboard_writer=None):

        # Data and configuration parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #if torch.cuda.is_available():
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
        print(f'Device set to = {self.device}')
        self.config = config
        self.writer = tensorboard_writer
        self.data_gen = {
            'x': None,
            'y': None,
            'y_hat': None}

        self._fixed_input_id = 30  # Id for the fixed input used to monitor the performance of the generator
        self._fixed_input = None
        self._fixed_input_counter = 0
        self._fixed_label = None

        # These are the names for the losses that will be saved
        self.loss_names = ['rec']
        self.losses = {x: 0 for x in self.loss_names}
        self.predictor = self.build_predictor()
        self.init_optimizers()
        self.criterionRec = nn.MSELoss()

        print(f'Input predictor = {self.config.input_shape}')
        summary(self.predictor, input_size=tuple(self.config.input_shape))

    def build_predictor(self):
        predictor = CRNN10(class_num=self.config.output_shape[1],
                           out_channels=self.config.output_shape[0],
                           in_channels=self.config.input_shape[0],
                           multi_track=self.config.dataset_multi_track)
        return predictor.to(self.device)

    def init_optimizers(self):
        self.optimizer_predictor = optim.Adam(self.predictor.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_predictor,
            factor=self.config.lr_decay_rate,
            patience=self.config.lr_patience_times,
            min_lr=self.config.lr_min
        )
        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer_predictor,
            multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)

    def set_input(self, x: torch.Tensor, y: torch.Tensor):
        """ Sets the local x (input data) and y (labels) from the minibatch
        This also supports using independent and multiple batches for the discrminator."""
        self.data_gen['x'] = x
        self.data_gen['y'] = y

        # Assign a fixed input to monitor the Generator (SELD net) task
        if self._fixed_input is None and self._fixed_input_counter == self._fixed_input_id:
            self._fixed_input = self.data_gen['x'].detach()
            self._fixed_label = self.data_gen['y'].detach()
        else:
            self._fixed_input_counter += 1

    def lr_step(self, val_loss, step=None):
        """ step in iterations"""
        if step % 1000 == 0:   # Hard-coded update at 1000 for generator
            self.lr_scheduler.step(metrics=val_loss)

    def get_lrs(self):
        return self.optimizer_predictor.state_dict()['param_groups'][0]['lr']

    def get_fixed_output(self):
        if self._fixed_input is not None:
            out = self.predictor(self._fixed_input).detach().cpu()
            return out

    def get_fixed_label(self):
        if self._fixed_label is not None:
            out = self._fixed_label.detach().cpu()
            return out

    def forward(self):
        self.data_gen['y_hat'] = self.predictor(self.data_gen['x'])

    def backward_predictor(self):
        """Calculate GAN and reconstruction loss for the generator"""
        loss_G_rec = self.criterionRec(self.data_gen['y_hat'], self.data_gen['y'])
        # Total weighted loss
        self.losses['rec'] = loss_G_rec
        loss_G = loss_G_rec
        loss_G.backward()

    def train_step(self):
        """ Calculates losses, gradients, and updates the network parameters"""
        self.predictor.train()
        self.forward()
        # Update Predictor
        self.predictor.zero_grad()
        self.backward_predictor()
        self.optimizer_predictor.step()



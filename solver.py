import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
#from torchsummary import summary
from torchinfo import summary

from utils import GradualWarmupScheduler, grad_norm, mixup_data, mixup_criterion
from models.crnn import CRNN10, CRNN
from models.samplecnn_raw import SampleCNN, SampleCNN_GRU
from models.losses import MSELoss_ADPIT


class Solver(object):
    def __init__(self, config, tensorboard_writer=None, model_checkpoint=None):

        # Data and configuration parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #if torch.cuda.is_available():
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
        print(f'Device set to = {self.device}')
        self.config = config
        self.writer = tensorboard_writer
        self.model_checkpoint = model_checkpoint
        self.data_gen = {
            'x': None,
            'y': None,
            'y_hat': None}

        self._fixed_input_id = 30  # Id for the fixed input used to monitor the performance of the generator
        self._fixed_input = None
        self._fixed_input_counter = 0
        self._fixed_label = None
        self._p_comp = 0.0
        self.lam = 1 # For mixup
        self.curriculum_scheduler = config.curriculum_scheduler
        self.curriculum_loss = 1e6
        self.curriculum_seld_metric = 1.0

        # If using multiple losses, each loss has a name, value, and function (criterion)
        self.loss_names = ['rec']
        self.loss_values = {x: 0 for x in self.loss_names}
        self.loss_fns = {x: self._get_loss_fn() for x in self.loss_names}

        # Build models
        self.predictor = self.build_predictor()
        self.init_optimizers()

        print(f'Input predictor = {self.config.input_shape}')
        #summary(self.predictor, input_size=tuple(self.config.input_shape))
        summary(self.predictor, input_size=tuple([1, *self.config.input_shape]),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['var_names'], verbose=1)

        if self.model_checkpoint is not None:
            print("Loading model state from {}".format(self.model_checkpoint))
            self.predictor.load_state_dict(torch.load(self.model_checkpoint, map_location=self.device))

    def save(self, each_monitor_path=None, iteration=None, start_time=None):
        raise NotImplementedError
        # TODO Finish this
        self._each_checkpoint_path = '{}/{}_params_{}_{}_{:07}.pth'.format(
            each_monitor_path,
            self._args.net,
            os.path.splitext(os.path.basename(self._train_list))[0],
            start_time,
            iteration)
        if self._args.parallel_gpu:
            torch_generator_net_state_dict = self._torch_generator_net.module.state_dict()
            torch_discrminator_net_state_dict = self._torch_discriminator_net.module.state_dict()
        else:
            torch_generator_net_state_dict = self._torch_generator_net.state_dict()
            torch_discrminator_net_state_dict = self._torch_discriminator_net.state_dict()
        checkpoint = {'generator_model_state_dict': torch_generator_net_state_dict,
                      'generator_optimizer_state_dict': self._torch_generator_optimizer.state_dict(),
                      'generator_scheduler_state_dict': self._torch_generator_lr_scheduler.state_dict(),
                      'discriminator_model_state_dict': torch_discrminator_net_state_dict,
                      'discriminator_optimizer_state_dict': self._torch_discriminator_optimizer.state_dict(),
                      'discriminator_scheduler_state_dict': self._torch_discriminator_lr_scheduler.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state()}
        torch.save(checkpoint, self._each_checkpoint_path)
        print('Checkpoint saved to {}.'.format(self._each_checkpoint_path))

        np.save('{}/example_input_latest'.format(each_monitor_path), self._input)
        np.save('{}/example_label_latest'.format(each_monitor_path), self._label)
        if type(self._torch_y_hat) is tuple:
            np.save('{}/example_pred_latest'.format(each_monitor_path),
                    self._torch_y_hat[0].cpu().detach().numpy())
        else:
            np.save('{}/example_pred_latest'.format(each_monitor_path), self._torch_y_hat.cpu().detach().numpy())

    def load(self):
        raise NotImplementedError
        # TODO Finish this
        checkpoint = torch.load(self._args.load_model, map_location=lambda storage, loc: storage)
        if self._args.parallel_gpu:
            self._torch_generator_net.module.load_state_dict(checkpoint['generator_model_state_dict'])
            self._torch_discriminator_net.module.load_state_dict(checkpoint['discriminator_model_state_dict'])
        else:
            self._torch_generator_net.load_state_dict(checkpoint['generator_model_state_dict'])
            self._torch_discriminator_net.load_state_dict(checkpoint['discriminator_model_state_dict'])
        # use for restart the same training
        self._torch_generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self._torch_generator_lr_scheduler.load_state_dict(checkpoint['generator_scheduler_state_dict'])
        self._torch_discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self._torch_discriminator_lr_scheduler.load_state_dict(checkpoint['discriminator_scheduler_state_dict'])
        print('Checkpoint was loaded from to {}.'.format(self._args.load_model))


    def build_predictor(self):
        if self.config.model == 'crnn10':
            predictor = CRNN10(class_num=self.config.output_shape[1],
                               out_channels=self.config.output_shape[0],
                               in_channels=self.config.input_shape[0],
                               multi_track=self.config.dataset_multi_track)
        elif self.config.model == 'samplecnn':
            predictor = SampleCNN(output_timesteps=math.ceil(self.config.dataset_chunk_size_seconds * 10),
                                  num_class=self.config.unique_classes)
        elif self.config.model == 'samplecnn_gru':
            predictor = SampleCNN_GRU(output_timesteps=math.ceil(self.config.dataset_chunk_size_seconds * 10),
                                      num_class=self.config.unique_classes, filters=[128,128,256,256,256,512,512])
        else:
            raise ValueError(f'Model : {self.config.model} is not supported.')

        return predictor.to(self.device)

    def _get_loss_fn(self) -> torch.nn.Module:
        loss_fn = None
        if self.config['dataset_multi_track']:
            loss_fn = MSELoss_ADPIT()
        elif self.config.model_loss_fn == 'mse':
            loss_fn = torch.nn.MSELoss()
        elif self.config.model_loss_fn == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss()

        assert loss_fn is not None, 'Wrong loss function'
        return loss_fn

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
        if step % 1 == 0:   # Step every validation, controlled from outside
            self.lr_scheduler.step(metrics=val_loss)

    def get_lrs(self):
        return self.optimizer_predictor.state_dict()['param_groups'][0]['lr']

    def curriculum_scheduler_step(self, step: int, val_loss=None, seld_metric=0):
        """Updates parameters for curriculum learning.
        For now, this supports:
            - p_comp for the augmentations
        """
        p_comp_max = 1.0   # TODO Hardcoded max

        if self.curriculum_scheduler == 'fixed':  # No update, augmentaiton always active
            self._p_comp = p_comp_max
        elif self.curriculum_scheduler == 'linear':  # Update p_comp linearly, every validation step
            update_every = self.config.logging_interval  # TODO: Hardcoded for now, find a better value
            if (step % update_every == 0) and step != 0:
                # Update _p_comp
                if self._p_comp < p_comp_max:
                    self._p_comp += 0.07  # TODO Hardcoded step size
        elif self.curriculum_scheduler == 'loss':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not val_loss < self.curriculum_loss:
                    if self._p_comp < p_comp_max:
                        self._p_comp += 0.07  # TODO Hardcoded step size
                if val_loss > 0.0:
                    self.curriculum_loss = val_loss
        elif self.curriculum_scheduler == 'seld_metric':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not seld_metric < self.curriculum_seld_metric:
                    if self._p_comp < p_comp_max:
                        self._p_comp += 0.07  # TODO Hardcoded step size
                #else:  # REmermbers the best metric ever,
                #    self.curriculum_seld_metric = seld_metric
                if True:  # REmermbers the last validation step, so updates are smoother
                    self.curriculum_seld_metric = seld_metric
        if self._p_comp > p_comp_max:
            self._p_comp = p_comp_max

    def get_curriculum_params(self):
        """ Returns the params that are being update via curriculum learning. """
        return self._p_comp
        p_comp = None
        if self._p_comp != 0.0:
            p_comp = self._p_comp
        return p_comp

    def get_fixed_output(self):
        if self._fixed_input is not None:
            out = self.predictor(self._fixed_input).detach().cpu()

            if self.config.dataset_multi_track:
                b, ts, other = out.shape
                out = out.view(-1, ts, 3, 3, self.config.unique_classes)
                out = out[..., :, 0, :, :].permute([0, 2, 3, 1])
            return out

    def get_fixed_label(self):
        if self._fixed_label is not None:
            out = self._fixed_label.detach().cpu()
            if self.config.dataset_multi_track:
                b, ts, tra, ch, _ = out.shape
                out = out[..., 0, 0:3, :].permute([0, 2, 3, 1])  # get first track only
            return out

    def get_grad_norm(self):
        """ Returns the gradient norms for the generator (SELD net) and discriminator. """
        grad_norm_model = grad_norm(self.predictor.parameters())
        return grad_norm_model

    def forward(self):
        # mixup
        if self.config.use_mixup:
            self.data_gen['x'], self.data_gen['y_a'], self.data_gen['y_b'], self.lam = mixup_data(self.data_gen['x'],
                                                                                             self.data_gen['y'],
                                                                                             alpha=self.config.mixup_alpha,
                                                                                             use_cuda=self.device == 'cuda')
        else:
            self.lam = 1.
        self.data_gen['y_hat'] = self.predictor(self.data_gen['x'])

    def backward_predictor(self):
        """Calculate GAN and reconstruction loss for the generator"""
        if self.config.use_mixup:
            loss_func = mixup_criterion(self.data_gen['y_a'], self.data_gen['y_b'], self.lam)
            loss_G_rec = loss_func(self.loss_fns['rec'], self.data_gen['y_hat'])
        else:
            loss_G_rec = self.loss_fns['rec'](self.data_gen['y_hat'], self.data_gen['y'])
        # Total weighted loss
        self.loss_values['rec'] = loss_G_rec
        loss_G_rec.backward()

    def train_step(self):
        """ Calculates losses, gradients, and updates the network parameters"""
        self.predictor.train()
        self.forward()
        # Update Predictor
        self.predictor.zero_grad()
        self.backward_predictor()
        self.optimizer_predictor.step()



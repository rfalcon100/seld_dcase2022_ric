import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
#from torchsummary import summary
from torchinfo import summary

from typing import List

from utils import GradualWarmupScheduler, grad_norm, mixup_data, mixup_criterion
from models.crnn import CRNN10, CRNN
from models.samplecnn_raw import SampleCNN, SampleCNN_GRU
from models.losses import MSELoss_ADPIT, generator_loss, discriminator_loss, AccDoaSpectralLoss
from models.discriminators import DiscriminatorModularThreshold


def set_requires_grad(nets: List, requires_grad=False):
    """Set requies_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not

    Adapted from:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/361f8b00d671b66db752e66493b630be8bc7d67b/models/base_model.py#L219
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class SolverGeneric(object):
    def __init__(self, config, tensorboard_writer=None, model_checkpoint=None):

        raise NotImplementedError


class SolverBasic(SolverGeneric):
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
        self.lam = 1 # For mixup

        self.curriculum_params = {'p_comp': 0.0}
        self.curriculum_scheduler = config.curriculum_scheduler
        self.curriculum_loss = 1e6
        self.curriculum_seld_metric = 1.0

        # If using multiple losses, each loss has a name, value, and function (criterion)
        self.loss_names = ['G_rec', 'G_rec_spec']
        self.loss_values = {x: 0 for x in self.loss_names}
        self.loss_fns = self._get_loss_fn()

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
        loss_fn, loss_spec = None, None
        if self.config['dataset_multi_track']:
            loss_fn = MSELoss_ADPIT()
        elif self.config.model_loss_fn == 'mse':
            loss_fn = torch.nn.MSELoss()
        elif self.config.model_loss_fn == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config.model_loss_fn == 'l1':
            loss_fn = torch.nn.L1Loss()

        if self.config.G_rec_spec == 'l1':
            loss_spec = AccDoaSpectralLoss(n_ffts=[256,128,64,32,16,8], distance='l1', device=self.device)
        elif self.config.G_rec_spec == 'l2':
            loss_spec = AccDoaSpectralLoss(n_ffts=[256, 128, 64, 32, 16, 8], distance='l2', device=self.device)
        assert loss_fn is not None, 'Wrong loss function'
        assert loss_spec is not None, 'Wrong loss function'

        losses = {'G_rec': loss_fn,
                  'G_rec_spec': loss_spec}
        return losses

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

    def lr_step(self, val_loss):
        """ step in iterations"""
        self.lr_scheduler.step(metrics=val_loss)

    def get_lrs(self):
        return [self.optimizer_predictor.state_dict()['param_groups'][0]['lr']]

    def curriculum_scheduler_step(self, step: int, val_loss=None, seld_metric=0):
        """Updates parameters for curriculum learning.
        For now, this supports:
            - p_comp for the augmentations
        """
        def update_curriculum_values():
            # Update _p_comp
            self.curriculum_params['p_comp'] += 0.07  # TODO Hardcoded step size

        p_comp_max = 1.0  # TODO Hardcoded max value
        if self.curriculum_scheduler == 'fixed':  # No update, augmentaiton always active
            self.curriculum_params['p_comp'] = p_comp_max
        elif self.curriculum_scheduler == 'linear':  # Update p_comp linearly, every validation step
            update_every = self.config.logging_interval
            if (step % update_every == 0) and step != 0:
                update_curriculum_values()
        elif self.curriculum_scheduler == 'loss':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not val_loss < self.curriculum_loss:
                    update_curriculum_values()
                if val_loss > 0.0:
                    self.curriculum_loss = val_loss
        elif self.curriculum_scheduler == 'seld_metric':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not seld_metric < self.curriculum_seld_metric:
                    update_curriculum_values()
                #else:  # REmermbers the best metric ever,
                #    self.curriculum_seld_metric = seld_metric
                if True:  # REmermbers the last validation step, so updates are smoother
                    self.curriculum_seld_metric = seld_metric
        if self.curriculum_params['p_comp'] > p_comp_max:
            self.curriculum_params['p_comp'] = p_comp_max

    def get_curriculum_params(self) -> List:
        """ Returns the params that are being update via curriculum learning. """
        p_comp = self.curriculum_params['p_comp']

        return [p_comp]

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
        return [grad_norm_model]

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
            loss_G_rec = loss_func(self.loss_fns['G_rec'], self.data_gen['y_hat'])
            loss_G_spec = 0.0
        else:
            loss_G_rec = self.loss_fns['G_rec'](self.data_gen['y_hat'], self.data_gen['y'])
            if self.config.w_rec_spec > 0.0:
                loss_G_spec = self.loss_fns['G_rec_spec'](self.data_gen['y_hat'], self.data_gen['y'])
            else:
                loss_G_spec = 0.0
        # Total weighted loss
        self.loss_values['G_rec'] = loss_G_rec
        self.loss_values['G_rec_spec'] = loss_G_spec
        total_loss = loss_G_rec + self.config.w_rec_spec * loss_G_spec
        total_loss.backward()

    def train_step(self):
        """ Calculates losses, gradients, and updates the network parameters"""
        self.predictor.train()
        self.forward()
        # Update Predictor
        self.predictor.zero_grad()
        self.backward_predictor()
        self.optimizer_predictor.step()


class SolverDAN(SolverGeneric):
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
        self.data_disc = {
            'x': None,
            'y': None,
            'y_hat': None}

        self._fixed_input_id = 30  # Id for the fixed input used to monitor the performance of the generator
        self._fixed_input = None
        self._fixed_input_counter = 0
        self._fixed_label = None
        self.curriculum_params = {'p_comp': 0.0,
                                  'w_adv': 0.0,
                                  'D_threshold_min': 0.0,
                                  'D_threshold_max': 10.0}

        self.lam = 1 # For mixup
        self.curriculum_scheduler = config.curriculum_scheduler
        self.curriculum_loss = 1e6
        self.curriculum_seld_metric = 1.0

        # If using multiple losses, each loss has a name, value, and function (criterion)
        self.loss_names = ['G_rec', 'G_adv', 'D_real', 'D_fake']
        self.loss_values = {x: 0 for x in self.loss_names}
        self.loss_fns = self._get_loss_fn()

        # Build models
        self.predictor, self.discriminator = self.build_models()
        self.init_optimizers()

        print(f'Input predictor = {self.config.input_shape}')
        #summary(self.predictor, input_size=tuple(self.config.input_shape))
        summary(self.predictor, input_size=tuple([1, *self.config.input_shape]),
                col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
                row_settings=['var_names'], verbose=1)
        print(f'Input discriminator = {self.config.disc_input_shape}')
        summary(self.discriminator, input_size=tuple([1, *self.config.disc_input_shape]),
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

    def build_models(self):
        if self.config.model == 'crnn10':
            predictor = CRNN10(class_num=self.config.output_shape[1],
                               out_channels=self.config.output_shape[0],
                               in_channels=self.config.input_shape[0],
                               multi_track=self.config.dataset_multi_track)
        elif self.config.model == 'samplecnn':
            raise NotImplementedError(f'Model : {self.config.model} is not supported.')
            predictor = SampleCNN(output_timesteps=math.ceil(self.config.dataset_chunk_size_seconds * 10),
                                  num_class=self.config.unique_classes)
        elif self.config.model == 'samplecnn_gru':
            raise NotImplementedError(f'Model : {self.config.model} is not supported.')
            predictor = SampleCNN_GRU(output_timesteps=math.ceil(self.config.dataset_chunk_size_seconds * 10),
                                      num_class=self.config.unique_classes, filters=[128,128,256,256,256,512,512])
        else:
            raise ValueError(f'Model : {self.config.model} is not supported.')

        if self.config.disc == 'DiscriminatorModularThreshold':
            discriminator = DiscriminatorModularThreshold(input_shape=self.config['disc_input_shape'],
                                                          n_feature_maps=self.config['disc_feature_maps'],
                                                          final_activation=self.config['disc_final'],
                                                          kernels=self.config['disc_kernels'],
                                                          strides=self.config['disc_strides'],
                                                          padding=self.config['disc_padding'],
                                                          normalization=self.config['disc_normalization'],
                                                          block=self.config['disc_block'],
                                                          conditioning=self.config['disc_conditioning'],
                                                          last_layer_multiplier=self.config['disc_final_multi'],
                                                          use_spectral_norm=self.config['disc_use_spectral_norm'],
                                                          threshold_min=self.config['disc_threshold_min'],
                                                          threshold_max=self.config['disc_threshold_max'],
                                                          use_threshold_norm=self.config['disc_use_threshold_norm'],
                                                          use_threshold_binarize=self.config['disc_use_threshold_binarize'],
                                                          use_low_pass=self.config['disc_use_low_pass'],
                                                          with_r=self.config['disc_with_r'],
                                                          freq_pooling=self.config['disc_freq_pooling'])
        else:
            raise ValueError(f'Discrminator : {self.config.disc} is not supported.')

        return predictor.to(self.device), discriminator.to(self.device)

    def _get_loss_fn(self) -> torch.nn.Module:
        loss_fn = None
        if self.config['dataset_multi_track']:
            loss_fn = MSELoss_ADPIT()
        elif self.config.model_loss_fn == 'mse':
            loss_fn = torch.nn.MSELoss()
        elif self.config.model_loss_fn == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config.model_loss_fn == 'l1':
            loss_fn = torch.nn.L1Loss()

        assert loss_fn is not None, 'Wrong loss function'

        losses = {'G_rec': loss_fn,
                  'G_adv': generator_loss,
                  'D_real': discriminator_loss,
                  'D_fake': discriminator_loss}
        return losses

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

        # Optimizer and scheduler for Discriminator
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.D_lr,
            betas=(0.5, 0.999),
            weight_decay=self.config.D_lr_weight_decay)
        if self.config.D_lr_scheduler == 'warmup':
            discriminator_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_discriminator,
                factor=self.config.D_lr_decay_rate,
                patience=self.config.D_lr_patience_times,
                min_lr=self.config.D_lr_min)
            self.lr_scheduler_discriminator = GradualWarmupScheduler(
                self.optimizer_discriminator,
                multiplier=1, total_epoch=5, after_scheduler=discriminator_lr_scheduler)  # hard coding
        elif self.config.D_lr_scheduler == 'lrstep':
            self.lr_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.optimizer_discriminator,
                                                                              step_size=self.config.D_lr_scheduler_step,
                                                                              gamma=self.config.D_lr_decay_rate)
        elif self.config.D_lr_scheduler == 'cosine':
            self.lr_scheduler_discriminator = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_discriminator,
                                                                                   T_max=self.config.num_iters,
                                                                                   eta_min=self.config.D_lr_min,
                                                                                   verbose=False)
        elif self.config.D_lr_scheduler == 'cosine-restart':
            self.lr_scheduler_discriminator = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer_discriminator,
                                                                                             T_0=self.config.D_lr_scheduler_T_0,
                                                                                             T_mult=self.config.D_lr_scheduler_T_mult,
                                                                                             eta_min=self.config.D_lr_min,
                                                                                             verbose=False)
        else:
            raise ValueError('Wrong value for the discriminator scheduler.')

    def set_input(self, x: torch.Tensor, y: torch.Tensor):
        """ Sets the local x (input data) and y (labels) from the minibatch
        This also supports using independent and multiple batches for the discrminator."""
        self.data_gen['x'] = x
        self.data_gen['y'] = y

        # Get data for discriminator
        # When self._args.D_batches == 0, we use the same batch for both Gen and Disc
        # When self._args.D_batches == 1, one independent batch for Gen and Disc
        # When self._args.D_batches > 1, one batch for Gen, and multiple independent batches for Disc.
        # WARNING:  D_batches >= 1 increases training time, due to feature extraction (spectrograms), and disk loading
        # WARNING:  D_batches > 1 increases GPU memory usage
        self.data_disc['x'] = []
        self.data_disc['y_real'] = []

        if self.config.D_batches > 0:
            raise NotImplementedError('Not ready for mutiple batches in the discrmintaro')
            for i in range(self.config.D_batches):
                tmp_input, tmp_label, _, _ = self._data.next()
                self._torch_input_disc.append(torch.tensor(tmp_input, dtype=torch.float).to(self._torch_device))
                self._torch_labels_disc.append(torch.tensor(tmp_label, dtype=torch.float).to(self._torch_device))
                if self._args.nda_func != 'none':
                    raise NotImplementedError('Not ready for NDA when using multiple batches for discrminator')
        else:
            self.data_disc['x'] = x.detach()
            self.data_disc['y'] = y.detach()

        batch_size = self.data_disc['x'].shape[0]
        self.data_disc['y_real'] = torch.ones(batch_size, 1).to(self.device)
        self.data_disc['y_fake'] = torch.zeros(batch_size, 1).to(self.device)

        # Assign a fixed input to monitor the Generator (SELD net) task
        if self._fixed_input is None and self._fixed_input_counter == self._fixed_input_id:
            self._fixed_input = self.data_gen['x'].detach()
            self._fixed_label = self.data_gen['y'].detach()
        else:
            self._fixed_input_counter += 1

    def lr_step(self, val_loss):
        """ step in iterations"""
        if val_loss is not None:
            self.lr_scheduler.step(metrics=val_loss)

    def lr_step_discriminator(self, val_loss):
        """ step in iterations"""
        if self.config.D_lr_scheduler == 'warmup' and val_loss is not None:
            self.lr_scheduler_discriminator.step(metrics=val_loss)
        else:
            if self.config.D_lr_scheduler == 'lrstep' and self.get_lrs()[-1] < self.config.D_lr_min:  # Manual check for D_lr_min
                return
            self.lr_scheduler_discriminator.step()

    def get_lrs(self):
        return [self.optimizer_predictor.state_dict()['param_groups'][0]['lr'],
                self.optimizer_discriminator.state_dict()['param_groups'][0]['lr']]

    def curriculum_scheduler_step(self, step: int, val_loss=None, seld_metric=0):
        """Updates parameters for curriculum learning.
        For now, this supports:
            - p_comp for the augmentations
            - D_threshold
            - w-adv
        """
        def update_curriculum_values():
            # Update _p_comp
            self.curriculum_params['p_comp'] += 0.07  # TODO Hardcoded step size
            # Update w_adv
            self.curriculum_params['w_adv'] += self.config['curriculum_w_adv']
            # Update D_threshold
            self.curriculum_params['D_threshold_min'] += self.config['curriculum_D_threshold_min']
            self.curriculum_params['D_threshold_max'] += self.config['curriculum_D_threshold_max']
            self.discriminator.update_threshold(self.curriculum_params['D_threshold_min'], self.curriculum_params['D_threshold_max'])

        p_comp_max = 1.0  # TODO Hardcoded max value
        if self.curriculum_scheduler == 'fixed':  # No update, augmentaiton always active
            self.curriculum_params['p_comp'] = p_comp_max
        elif self.curriculum_scheduler == 'linear':  # Update p_comp linearly, every validation step
            update_every = self.config.logging_interval
            if (step % update_every == 0) and step != 0:
                update_curriculum_values()
        elif self.curriculum_scheduler == 'loss':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not val_loss < self.curriculum_loss:
                    update_curriculum_values()
                if val_loss > 0.0:
                    self.curriculum_loss = val_loss
        elif self.curriculum_scheduler == 'seld_metric':
            if (step % self.config.logging_interval == 0) and step != 0:
                if not seld_metric < self.curriculum_seld_metric:
                    update_curriculum_values()
                #else:  # REmermbers the best metric ever,
                #    self.curriculum_seld_metric = seld_metric
                if True:  # REmermbers the last validation step, so updates are smoother
                    self.curriculum_seld_metric = seld_metric
        if self.curriculum_params['p_comp'] > p_comp_max:
            self.curriculum_params['p_comp'] = p_comp_max

    def get_curriculum_params(self) -> List:
        """ Returns the current values for the params that are being update via curriculum learning. """
        p_comp = self.curriculum_params['p_comp']
        curr_w_adv = self.curriculum_params['w_adv']
        curr_d_threshold_min = self.curriculum_params['D_threshold_min']
        curr_d_threshold_max = self.curriculum_params['D_threshold_max']

        return [p_comp, curr_w_adv, curr_d_threshold_min, curr_d_threshold_max]

        curr_w_adv, curr_d_threshold_min, curr_d_threshold_max = None, None, None
        if self.config['curriculum_w_adv'] != 0.0:
            curr_w_adv = self.curriculum_params['w_adv']
        if self.config['curriculum_D_threshold_min'] != 0.0:
            curr_d_threshold_min = self.curriculum_params['D_threshold_min']
        if self.config['curriculum_D_threshold_max'] != 0.0:
            curr_d_threshold_max = self.curriculum_params['D_threshold_max']
        return [p_comp, curr_w_adv, curr_d_threshold_min, curr_d_threshold_max]

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
        grad_norm_predictor = grad_norm(self.predictor.parameters())
        grad_norm_disc = grad_norm(self.discriminator.parameters())
        return [grad_norm_predictor, grad_norm_disc]

    def get_conditioned_input(self, input: torch.Tensor, labels: torch.Tensor):
        # Prepare the conditioning input
        # Here input is the audio input of the generator [batch, channels, freqs, frames], e.g. (batch, 7, 257, 128)
        # And labels are the target or outputs of the generator [batch, coords, classes, frames], e.g. (batch, 3, 12, 128)
        if self.discriminator.conditioning == 'concat':
            # We upsample the labels across the -2 classes dimension to match the frequencies, and then concat across channels
            tmp = torch.cat((input, self.upsampler(labels)), 1)
        elif self.discriminator.conditioning == 'none':
            tmp = labels
        elif self.discriminator.conditioning == 'none-upsample':
            tmp = self.upsampler(labels)  # here we upsample the labels long frequency/class axis
        return tmp

    def forward(self):
        # For the predictor
        if self.config.use_mixup:  # mixup
            raise NotImplementedError
            self.data_gen['x'], self.data_gen['y_a'], self.data_gen['y_b'], self.lam = mixup_data(self.data_gen['x'], self.data_gen['y'],
                                                                                                  alpha=self.config.mixup_alpha,
                                                                                                  use_cuda=self.device == 'cuda')
        else:
            self.lam = 1.
        self.data_gen['y_hat'] = self.predictor(self.data_gen['x'])

        # Discriminator
        self.data_disc['y_hat'] = self.predictor(self.data_disc['x'])
        self.upsampler = nn.Upsample(size=(self.data_gen['x'].shape[-2], self.data_gen['x'].shape[-1]), mode='area')

    def backward_discriminator(self):
        """Calculate GAN loss for the discriminator"""
        tmp_loss_fake, tmp_loss_real = [], []
        # Fake
        fake_y = self.get_conditioned_input(self.data_disc['x'], self.data_disc['y_hat'])
        d_fake = self.discriminator(fake_y.detach())  # Detach to stop gradients for the predictor
        loss_D_fake = self.loss_fns['D_fake'](d_fake, False, self.config['D_crit'])
        # Real
        real_y = self.get_conditioned_input(self.data_disc['x'], self.data_disc['y'])
        d_real = self.discriminator(real_y)
        loss_D_real = self.loss_fns['D_fake'](d_real, True, self.config['D_crit'])

        # Backprop and log losses
        tmp_loss_fake.append(loss_D_fake)
        tmp_loss_real.append(loss_D_real)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()

        self.loss_values['D_fake'] = torch.mean(torch.stack(tmp_loss_fake))
        self.loss_values['D_real'] = torch.mean(torch.stack(tmp_loss_real))
        del loss_D

    def backward_predictor(self):
        """Calculate GAN and reconstruction loss for the generator"""
        # Adversarial loss: y_hat = pred(x) should fool the discriminator
        fake_y = self.get_conditioned_input(self.data_gen['x'], self.data_gen['y_hat'])
        d_fake = self.discriminator(fake_y)
        loss_G_adv = self.loss_fns['G_adv'](d_fake, loss_type=self.config['G_crit'])   ### TODO fix loss type
        if self.config.use_mixup:
            loss_func = mixup_criterion(self.data_gen['y_a'], self.data_gen['y_b'], self.lam)
            loss_G_rec = loss_func(self.loss_fns['G_rec'], self.data_gen['y_hat'])
        else:
            loss_G_rec = self.loss_fns['G_rec'](self.data_gen['y_hat'], self.data_gen['y'])

        # Total weighted loss
        self.loss_values['G_adv'] = loss_G_adv
        self.loss_values['G_rec'] = loss_G_rec
        loss_G = (self.config.w_adv * loss_G_adv) + (self.config.w_rec * loss_G_rec)
        loss_G.backward()
        del loss_G

    def train_step(self):
        """ Calculates losses, gradients, and updates the network parameters"""
        if self.config.disc_algorithm == 'dfgan':
            raise NotImplementedError
        elif self.config.disc_algorithm == 'soft_labels':
            raise NotImplementedError
        elif self.config.disc_algorithm == 'vanilla':
            self.predictor.train()
            self.discriminator.train()
            self.forward()
            # Update Discriminator
            set_requires_grad(self.discriminator, True)
            self.optimizer_discriminator.zero_grad()
            self.backward_discriminator()
            if self.config.D_grad_clip > 0:  # Optional gradient clipping to establize traning
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config.D_grad_clip)
            self.optimizer_discriminator.step()
            # Update Predictor
            set_requires_grad(self.discriminator, False)
            self.optimizer_predictor.zero_grad()
            self.backward_predictor()
            self.optimizer_predictor.step()
        else:
            raise NotImplementedError

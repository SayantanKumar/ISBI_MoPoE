import os
import numpy as np
import hydra
import re
import collections.abc

import torch
import pytorch_lightning as pl

from os.path import join, isdir, exists
from datetime import datetime
from abc import ABC, abstractmethod
from hydra import compose, initialize, initialize_config_dir
from schema import Schema, SchemaError, And, Or

from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf, open_dict

from .constants import *
from .validation import config_schema
from .datasets import MVDataset
from .exceptions import *

def update_dict(d, u, l):
    for k, v in u.items():
        if k in l:
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v, l=v.keys())
            else:
                d[k] = v
    return d

class BaseModelAE(ABC, pl.LightningModule):
    """Base class for autoencoder models.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    is_variational = False

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        assert (model_name is not None and model_name in MODELS), \
        "Model name is invalid"  # have to choose which model always

        try:
            # check input_dim
            Schema(And(list, lambda x: len(x) > 0), error="input_dim should not be empty").validate(input_dim)
            Schema([Or(int, tuple)], error="input_dim should be a list of int/tuple").validate(input_dim)
            for d in input_dim:
                if isinstance(d, int):
                    Schema(lambda x: x > 0, error="each dim should be > 0").validate(d)
                else:
                    Schema(lambda x: all(a > 0 for a in x), error="each dim should be > 0").validate(d)
                    # Schema(lambda x: len(x) in [1,2,3], error="each dim should be 1D or 2D").validate(d)

            # check z_dim
            if z_dim is not None:
                Schema(And(int, lambda x: x > 0), error="z_dim must be > 0").validate(z_dim)
        except SchemaError as se:
            raise ConfigError(se) from None

        super().__init__()
        self.model_name = model_name
        self.input_dim = input_dim
        self.n_views = len(self.input_dim)
        self.z_dim = z_dim

        with initialize(version_base=None, config_path="../configs"):
            def_cfg = compose(
                            config_name="default",
                            return_hydra_config=True,
                            overrides=[f"model_type={self.model_name}.yaml"]
                        )

        user_cfg = None
        if cfg is not None: # user overrides default config
            if os.path.isabs(cfg):
                cfgdir, cfg_file = os.path.split(cfg)
                with initialize_config_dir(version_base=None, config_dir=cfgdir):
                    user_cfg = compose(
                                config_name=cfg_file,
                                return_hydra_config=True
                            )
            else:
                workdir = os.getcwd()
                with initialize_config_dir(version_base=None, config_dir=workdir):
                    user_cfg = compose(
                                config_name=cfg,
                                return_hydra_config=True
                            )


        def_cfg = self.__updateconfig(def_cfg, user_cfg)

        if self.z_dim is not None:   # overrides hydra config... passed arg has precedence
            def_cfg.model.z_dim = self.z_dim


        # validate model configuration
        self.cfg = self.__checkconfig(def_cfg)

        self.__dict__.update(self.cfg.model)

        print("MODEL: ", self.model_name)
        self.print_config() #TODO: put this in debug mode logging

        if all(k in self.cfg.model for k in ["seed_everything", "seed"]):
            pl.seed_everything(self.cfg.model.seed, workers=True)

        self._setencoders()
        self._setdecoders()
        self._setprior()

        # TODO: should this be in the end of instance init()?
        self.save_hyperparameters()
  
    ################################            public methods
    def fit(self, *data, labels=None, max_epochs=None, batch_size=None):
        self.create_folder(self.cfg.out_dir)
        self.save_config()

       # data = list(data)

       # if not all(data_.shape[0] == data[0].shape[0] for data_ in data):
      #      raise InputError('All modalities must have the same number of entries')

      #  if not (len(data) == self.n_views):
      #      raise InputError("number of modalities must be equal to number of views")

       # for i in range(self.n_views):
       #     data_dim = data[i].shape[1:]
       #     if len(data_dim) == 1:
        #        data_dim = data_dim[0]
         #   if not (data_dim == self.input_dim[i]):
        #        raise InputError("modality's shape must be equal to corresponding input_dim's shape")

        self._training = True
        if max_epochs is not None:
            self.max_epochs = max_epochs
            self.cfg.trainer.max_epochs = max_epochs
        else:
            self.max_epochs = self.cfg.trainer.max_epochs

        if batch_size is not None:
            self.batch_size = batch_size
            self.cfg.datamodule.batch_size = batch_size
        else:
            self.batch_size = self.cfg.datamodule.batch_size

        callbacks = []
        if self.cfg.datamodule.is_validate:
            for _, cb_conf in self.cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(cb_conf))

        logger = hydra.utils.instantiate(self.cfg.logger)

        # NOTE: have to check file exists otherwise error raised
        if (self.cfg.trainer.resume_from_checkpoint is None) or \
            (not exists(self.cfg.trainer.resume_from_checkpoint)):
            self.cfg.trainer.resume_from_checkpoint = None

        py_trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=callbacks, logger=logger,
        )
        datamodule = hydra.utils.instantiate(
           self.cfg.datamodule, data=data, labels=labels, _recursive_=False
        )
        py_trainer.fit(self, datamodule)

    def predict_latents(self, *data, batch_size=None):
        return self.__predict(*data, batch_size=batch_size)

    def predict_reconstruction(self, *data, batch_size=None):
        return self.__predict(*data, batch_size=batch_size, is_recon=True)

    def print_config(self, cfg=None, keys=None):
        if cfg is None:
            cfg = self.cfg

        if keys is not None:
            print(f"{'model_name'}:\n  {cfg['model_name']}")
            for k in keys:
                if k in CONFIG_KEYS:
                    if cfg[k] is not None:
                        str = (OmegaConf.to_yaml(cfg[k])).replace("\n", "\n  ")
                    else:
                        str = "null\n"
                    print(f"{k}:\n  {str}")
        else:
            self.print_config(cfg=cfg, keys=CONFIG_KEYS)

    def save_config(self, keys=None):
        run_time = datetime.now().strftime("%Y-%m-%d_%H%M")
        save_cfg = {}
        if keys is not None:
            for k in keys:
                if k in CONFIG_KEYS:
                    save_cfg[k] = self.cfg[k]
            OmegaConf.save(save_cfg, join(self.cfg.out_dir, 'config_{0}.yaml'.format(run_time)))
        else:
            self.save_config(keys=CONFIG_KEYS)

    def create_folder(self, dir_path):
        check_folder = isdir(dir_path)
        if not check_folder:
            os.makedirs(dir_path)

    ################################            abstract methods

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    @abstractmethod
    def loss_function(self, x, fwd_rtn):
        raise NotImplementedError()

    ################################            LightningModule methods
    @abstractmethod
    def forward(self, x):
         raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="val")

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.cfg.out_dir, "model.ckpt"))
        torch.save(self, join(self.cfg.out_dir, "model.pkl"))

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()),
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
        return optimizers

    ################################            protected methods, can be overwritten by child
    def _setencoders(self):
        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    def _setdecoders(self):
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.decoder.dec{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    def _setprior(self):
        if self.model_name not in VARIATIONAL_MODELS or \
            (self.model_name in VARIATIONAL_MODELS and not self.sparse):
            self.prior = hydra.utils.instantiate(self.cfg.prior)

    ################################            private methods
    def __updateconfig(self, orig, update):
        OmegaConf.set_struct(orig, True)
        with open_dict(orig):
            # update default cfg with user config
            if update is not None:
                update_keys = list(set(update.keys()) & set(CONFIG_KEYS))
                orig = update_dict(orig, update, l=update_keys)

            # update encoder/decoder config
            for i, d in enumerate(self.input_dim):
                enc_key = f"enc{i}"
                if enc_key not in orig.encoder.keys():
                    if update is not None and "encoder" in update.keys() and \
                        enc_key in update.encoder.keys(): # use user-defined
                        orig.encoder[enc_key] = update.encoder[enc_key].copy()
                    else: # use default
                        orig.encoder[enc_key] = orig.encoder.default.copy()

                dec_key = f"dec{i}"
                if dec_key not in orig.decoder.keys():
                    if update is not None and "decoder" in update.keys() and \
                        dec_key in update.decoder.keys(): # use user-defined
                        orig.decoder[dec_key] = update.decoder[dec_key].copy()
                    else: # use default
                        orig.decoder[dec_key] = orig.decoder.default.copy()
        if update is not None and update.get('out_dir'):
            orig.out_dir = update.out_dir
        return orig

    def __checkconfig(self, cfg):

        cfg_dict = OmegaConf.to_container(cfg)

        try:
            cfg_dict = config_schema.validate(cfg_dict)
        except SchemaError as se:
            raise ConfigError(se) from None
            
        if self.model_name == MODEL_JMVAE and len(self.input_dim) != 2:
            raise InputError('JMVAE expects two len(input_dim) == 2')

        # pattern = re.compile(r'multiviewae\.architectures\..*\.Decoder') #TODO FIX THIS TO WORK WITH OTHER DECODERS OUTSIDE OF LIB
        # for k in cfg.decoder.keys():
        #     if eval(f"cfg.decoder.{k}.dec_dist._target_") in \
        #     ["multiviewae.base.distributions.Default", \
        #     "multiviewae.base.distributions.Bernoulli"]:
        #         if not bool(pattern.match(eval(f"cfg.decoder.{k}._target_"))):
        #             raise ConfigError(f"{k}: must use non-variational Decoder if decoder dist is Default/Bernoulli.")

        if self.model_name in [MODEL_AE] + ADVERSARIAL_MODELS:
            pattern1 = re.compile(r'multiviewae\.architectures\..*\.VariationalEncoder')
            pattern2 = re.compile(r'multiviewae\.base\.distributions\..*Normal')
            for k in cfg.encoder.keys():
                if bool(pattern1.match(eval(f"cfg.encoder.{k}._target_"))):
                    raise ConfigError(f"{k}: must use non-variational encoder for adversarial models.")

                if bool(pattern2.match(eval(f"cfg.encoder.{k}.enc_dist._target_"))):
                    raise ConfigError(f"{k}: must not use Normal/MultivariateNormal encoder dist for adversarial models")

        if self.model_name in VARIATIONAL_MODELS:
            self.is_variational = True

            pattern = re.compile(r'multiviewae\.architectures\..*\.VariationalEncoder')
            for k in cfg.encoder.keys():
             #   if not bool(pattern.match(eval(f"cfg.encoder.{k}._target_"))): #TODO FIX THIS TO WORK WITH OTHER ENCODERS OUTSIDE OF LIB
              #      raise ConfigError(f"{k}: must use variational encoder for variational models")

                if cfg.prior._target_ != eval(f"cfg.encoder.{k}.enc_dist._target_"):
                    raise ConfigError('Encoder and prior must have the same distribution for variational models')

        if cfg.prior._target_ == "multiviewae.base.distributions.Normal":
            if not isinstance(cfg.prior.loc, (int, float)):
                raise ConfigError("loc must be int/float for Normal prior dist")

            if not isinstance(cfg.prior.scale, (int, float)):
                raise ConfigError("scale must be int/float for Normal prior dist")

        else:   # MultivariateNormal
            if isinstance(cfg.prior.loc, (int, float)):
                cfg.prior.loc = [cfg.prior.loc] * cfg.model.z_dim

            if isinstance(cfg.prior.scale, (int, float)):
                cfg.prior.scale = [cfg.prior.scale] * cfg.model.z_dim

            if  len(cfg.prior.loc) != len(cfg.prior.scale):
                raise ConfigError("loc and scale must have the same length for MultivariateNormal prior dist")

            if len(cfg.prior.loc) != cfg.model.z_dim:
                raise ConfigError("loc and scale must have the same length as z_dim for MultivariateNormal prior dist")

        # should be always false for non-sparse models
        if self.model_name not in SPARSE_MODELS:
            cfg.model.sparse = False    # TODO: log warning if changing overriding value

        return cfg

    def __step(self, batch, batch_idx, stage):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def __predict(self, *data, batch_size=None, is_recon=False):
        self._training = False

        data = list(data)

     #   if not (len(data) == self.n_views):
       #     raise InputError("number of modalities must be equal to number of views")

       # for i in range(self.n_views):
       #     data_dim = data[i].shape[1:]
       #     if len(data_dim) == 1:
         #       data_dim = data_dim[0]
         #   if not (data_dim == self.input_dim[i]):
           #     raise InputError("modality's shape must be equal to corresponding input_dim's shape")
        if batch_size is None:
          #  batch_size = data[0].shape[0]
            batch_size = len(data)

        self.cfg.datamodule.batch_size = batch_size
        self.cfg.datamodule.is_validate = False

        datamodule = hydra.utils.instantiate(
           self.cfg.datamodule, data=data, labels=None, _recursive_=False
        )

        datamodule.setup(stage="test")
        dataset = datamodule.train_dataset
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            z_ = None
            for batch_idx, local_batch in enumerate(generator):

                local_batch = [
                    local_batch_.to(self.device) for local_batch_ in local_batch
                ]
                z = self.encode(local_batch)
                if self.sparse:
                    z = self.apply_threshold(z)
                if is_recon:
                    z = self.decode(z)

                z = [
                        [ d__._sample().cpu().detach().numpy() for d__ in d_ ]
                        if isinstance(d_, (list))
                        else
                        (d_.cpu().detach().numpy() if isinstance(d_, torch.Tensor)
                        else d_._sample().cpu().detach().numpy())
                        for d_ in z
                    ]

                if z_ is not None:
                    z_ = [
                            [ np.append(d_, d, axis=0) for d_, d in zip(p_,p) ]
                            if isinstance(p_, list) else np.append(p_, p, axis=0)
                            for p_, p in zip(z_, z)
                         ]
                else:
                    z_ = z
        return z_


################################################################################
class BaseModelVAE(BaseModelAE):
    """Base class for variational autoencoder models. Inherits from BaseModelAE.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=model_name,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)

    ################################            class methods
    def apply_threshold(self, z):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """

        keep = (self.__dropout() < self.threshold).squeeze().cpu()
        z_keep = []

        for _ in z:
            _ = _._sample()
            _[:, ~keep] = 0
            d = hydra.utils.instantiate(   
                self.cfg.encoder.default.enc_dist, loc=_, scale=1
            )
            z_keep.append(d)
            del _

        return z_keep

    ################################            protected methods
    def _setencoders(self):

        if self.sparse and self.threshold != 0.:
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    ################################            private methods
    def __dropout(self):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """
        alpha = torch.exp(self.log_alpha.detach())
        return alpha / (alpha + 1)

################################################################################
class BaseModelAAE(BaseModelAE):
    """Base class for adversarial autoencoder models. Inherits from BaseModelAE.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    is_wasserstein = False

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=model_name,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.automatic_optimization = False

        self.discriminator = hydra.utils.instantiate(
            self.cfg.discriminator,
            input_dim=self.z_dim,
            output_dim=1,
            is_wasserstein=self.is_wasserstein,
            _convert_="all"
        )

    ################################            abstract methods
    @abstractmethod
    def forward_recon(self, x):
        raise NotImplementedError()

    @abstractmethod
    def forward_discrim(self, x):
        raise NotImplementedError()

    @abstractmethod
    def forward_gen(self, x):
        raise NotImplementedError()

    @abstractmethod
    def recon_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    @abstractmethod
    def generator_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    @abstractmethod
    def discriminator_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    ################################        unused abstract methods
    def loss_function(self, x, fwd_rtn):
        pass

    def forward(self, x):
        pass

    ################################            LightningModule methods
    def training_step(self, batch, batch_idx):
        loss = self.__optimise_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"train_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.__validate_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"val_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def configure_optimizers(self):
        optimizers = []
        #Encoder optimizers
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        #Decoder optimizers
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        #Generator optimizers
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        #Discriminator optimizers
        optimizers.append(
            torch.optim.Adam(
                list(self.discriminator.parameters()), lr=self.learning_rate
            )
        )
        return optimizers

    ################################            private methods
    def __optimise_batch(self, local_batch):
        fwd_return = self.forward_recon(local_batch)
        loss_recon = self.recon_loss(local_batch, fwd_return)
        opts = self.optimizers()
        enc_opt = [opts.pop(0) for idx in range(self.n_views)]
        dec_opt = [opts.pop(0) for idx in range(self.n_views)]
        gen_opt = [opts.pop(0) for idx in range(self.n_views)]
        disc_opt = opts[0]
        [optimizer.zero_grad() for optimizer in enc_opt]
        [optimizer.zero_grad() for optimizer in dec_opt]
        self.manual_backward(loss_recon)
        [optimizer.step() for optimizer in enc_opt]
        [optimizer.step() for optimizer in dec_opt]

        fwd_return = self.forward_discrim(local_batch)
        loss_disc = self.discriminator_loss(fwd_return)
        disc_opt.zero_grad()
        self.manual_backward(loss_disc)
        disc_opt.step()
        if self.is_wasserstein:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        fwd_return = self.forward_gen(local_batch)
        loss_gen = self.generator_loss(fwd_return)
        [optimizer.zero_grad() for optimizer in gen_opt]
        self.manual_backward(loss_gen)
        [optimizer.step() for optimizer in gen_opt]
        loss_total = loss_recon + loss_disc + loss_gen
        loss = {
            "loss": loss_total,
            "recon": loss_recon,
            "disc": loss_disc,
            "gen": loss_gen,
        }
        return loss

    def __validate_batch(self, local_batch):
        with torch.no_grad():
            self.eval()
            fwd_return = self.forward_recon(local_batch)
            loss_recon = self.recon_loss(local_batch, fwd_return)
            fwd_return = self.forward_discrim(local_batch)
            loss_disc = self.discriminator_loss(fwd_return)
            fwd_return = self.forward_gen(local_batch)
            loss_gen = self.generator_loss(fwd_return)
            loss_total = loss_recon + loss_disc + loss_gen
            loss = {
                "loss": loss_total,
                "recon": loss_recon,
                "disc": loss_disc,
                "gen": loss_gen,
            }
        return loss

################################################################################
# TODO: use this
class BaseEncoder(pl.LightningModule):

    def __init__(self, **kwargs):
         super().__init__()
         self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        pass
        # raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

class BaseDecoder(pl.LightningModule):

    def __init__(self, **kwargs):
         super().__init__()
         self.model_name = 'BaseDecoder'
         self.save_hyperparameters()

    @abstractmethod
    def forward(self, z):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

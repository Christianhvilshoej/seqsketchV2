import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from seqsketch.utils.modules import get_class_from_string
from tqdm import tqdm


class SeqStrokeDiffusionModule2(pl.LightningModule):

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self._initialize_networks()
        self._intialize_scheduler()
        self.num_train_timesteps = params.scheduler.params.num_train_timesteps
        self.num_inference_timesteps = params.num_inference_timesteps
        self.single = params.denoising_network.params.single
        self.max_seq_length = params.denoising_network.params.max_seq_length
        self.max_strokes = params.denoising_network.params.max_strokes
        self.prediction_type = params.scheduler.params.prediction_type
        self.relative = params.relative
        self.threshold = 0.5

        assert self.prediction_type in ("epsilon","sample"), "Prediction type as to be either 'epsilon' or 'sample'"


    def load_pretrained_weights(self, path):
        # Load the checkpoint data (this includes more than just the model's state dict)
        checkpoint = torch.load(path, map_location=self.device)

        # Extract and load only the model's state dictionary
        self.load_state_dict(checkpoint["state_dict"])

    def _intialize_scheduler(self):
        cls = get_class_from_string(f"seqsketch.models.{self.params.scheduler.module}")
        self.scheduler = cls(self.params.scheduler.params)

    def _initialize_networks(self):
        cls = get_class_from_string(
            f"seqsketch.models.{self.params.denoising_network.module}"
        )
        self.denoising_network = cls(self.params.denoising_network.params)
        self.learnable_parameters = self.denoising_network.parameters()

    def calculate_loss(self, x0, x0_mask, c, c_mask):
        t = torch.randint(
            0, self.num_train_timesteps, (x0.size(0),), device=x0.device
        ).long()
        eps = torch.randn_like(x0)
        xt = self.scheduler.add_noise(x0, eps, t)
        pred = self.denoising_network(xt, t, c, x_mask=x0_mask, c_mask=c_mask)
        if self.prediction_type == "epsilon":
            loss = torch.mean((eps - pred) ** 2)
        elif self.prediction_type == "sample":
            loss = torch.mean((x0 - pred)**2)
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or "
            )
        print(loss)
        return loss

    def sampling(self, c, c_mask = None):
        # run full denoising process
        x0_pred = torch.randn_like(c[:,:1])
        for t in tqdm(self.scheduler.timesteps):
            t = t.repeat(x0_pred.size(0))
            eps_pred = self.denoising_network(x0_pred, t, c, c_mask=c_mask)
            x0_pred = self.scheduler.step(eps_pred, t[0], x0_pred).prev_sample
        return self.postprocess(x0_pred)

    def postprocess(self, x_hat):
        # clamp tp 0,256 and round to nearest integer
        if self.relative:
            x_hat = torch.clamp(x_hat, -1, 1)
        else:
            x_hat = torch.clamp(x_hat, 0, 1)
        return x_hat

    def forward(self, bs, iterations=10):
        padding_value = 0 if self.relative else -1
        """
        c_all = torch.zeros((bs, self.max_strokes, self.max_seq_length, 2), device=self.device) * (padding_value)
        c = c_all[:,0].clone() if self.single else c_all
        N = 1 if self.single else self.max_strokes
        c_mask = torch.zeros((bs, N, self.max_seq_length, 2), device=self.device) 
        
        for i in range(iterations):
            x_hat = self.sampling(c, c_mask)
            c_all[:,i:(i+1)] = x_hat
            if self.single:
                c = x_hat
                c_mask = torch.ones_like(c_mask)
            else:
                c_mask[:,i] = 1
                c = c_all
            
         return c_all[:,:(iterations + 1)]   

        """
        c = torch.zeros((bs, self.max_strokes, self.max_seq_length, 2), device=self.device) * (padding_value)
        if self.single:
            c_single = torch.zeros((bs, 1,self.max_seq_length, 2), device=self.device) * (padding_value)
            c_mask = torch.zeros((bs, 1, self.max_seq_length, 2), device=self.device) 
            for i in range(iterations):
                x_hat = self.sampling(c_single, c_mask)
                # Update c
                c_single = x_hat
                c[:,i:(i+1)] = x_hat
                c_mask = None
        else:
            c_mask = torch.zeros((bs, self.max_strokes, self.max_seq_length, 2), device=self.device)
            for i in range(iterations):
                x_hat = self.sampling(c, c_mask)
                # Update c
                c[:,i:(i+1)] = x_hat
                # Update mask
                c_mask[:,i] = 1
        return c[:,:(iterations + 1)]

    def prepare_batch(self, batch):
        # here we should prepare batch for the model
        x = batch["next_stroke"]  # shape: (batch_size, channels, height, width)
        x_mask = batch["next_stroke_mask"]
        # c = batch["current_strokes"]  # shape: (batch_size, channels, height, width)
        # n_strokes = batch["n_strokes"]
        #
        if "current_strokes" in self.params.conditioning:
            c = batch["current_strokes"]
            c_mask = batch["current_strokes_mask"]
            if self.single:
                # subtract 1 from all (non zero) indices
                step = batch["step"].clone()
                step[step > 0] = step[step > 0]  - 1
                # Chop chop (not great)
                step[(step > self.max_seq_length - 1)] = self.max_seq_length - 1
                B = c.shape[0]
                # Extract the "newest" current stroke as condition
                c = c[torch.arange(B), step].unsqueeze(1)
                c_mask = c_mask[torch.arange(B), step].unsqueeze(1)
        else:
            c = None
            c_mask = None
        return x, x_mask, c, c_mask

    def training_step(self, batch, batch_idx):
        x, x_mask, c, c_mask = self.prepare_batch(batch)
        loss = self.calculate_loss(x, x_mask, c, c_mask)
        # log the loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_mask, c, c_mask = self.prepare_batch(val_batch)
        loss = self.calculate_loss(x, x_mask, c, c_mask)
        # log the loss
        self.log("val_loss", loss)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.learnable_parameters, lr=self.params.lr)
        if self.params.lr_scheduler:
            if self.params.lr_scheduler == "exponential_decay_0.01":
                optimizer = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.01
                )
            else:
                raise NotImplementedError(
                    f"Learning rate scheduler {self.params.lr_scheduler} not implemented"
                )
        return optimizer

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.scheduler.set_timesteps(self.num_train_timesteps)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

    def eval(self):
        super().eval()
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

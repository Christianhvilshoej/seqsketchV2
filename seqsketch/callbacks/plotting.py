from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, Image


class ImageLogger(Callback):
    def __init__(self, model, data_module):
        super().__init__()

        dataloader = data_module.val_dataloader()
        batch = next(iter(dataloader))
        self.x, _, self.c, _ = model.prepare_batch(batch)

    def plot_inline(self, condition_image, sample_image, true_image):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.tight_layout()
        for im, title, a in zip(
            [condition_image, sample_image, true_image],
            ["Condition", "Sample", "True"],
            ax,
        ):
            if im is None:
                im = torch.zeros_like(true_image)
            im_display = a.imshow(im.squeeze().detach().cpu().numpy(), cmap="gray")
            a.set_title(title)
            plt.colorbar(im_display, ax=a)
            a.axis("off")
        # convert to wandb image
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def on_validation_epoch_end(self, trainer, pl_module):
        # Dataloader loads on CPU --> pass to GPU
        x0 = self.x.to(device=pl_module.device)
        c = self.c
        if c is not None:
            c = self.c.to(device=pl_module.device)
        zt = torch.randn_like(x0)
        x0_pred = pl_module.sampling(zt, c)

        # move arrays back to the CPU for plotting
        x0_pred = x0_pred.cpu()
        x0 = x0.cpu()
        if c is not None:
            c = c.cpu()
        else:
            # create list of None for condition images
            c = [None] * x0.size(0)

        # generate figures in a list
        figs = [
            self.plot_inline(im1, im2, im3) for im1, im2, im3 in zip(c, x0_pred, x0)
        ]

        # add to logger like so
        trainer.logger.experiment.log({"Sample images": figs})


class ImageLogger2(Callback):
    def __init__(self, model, data_module):
        super().__init__()

        dataloader = data_module.val_dataloader()
        batch = next(iter(dataloader))
        self.x,  _, self.c, self.c_mask = model.prepare_batch(batch)
        self.bs = self.x.size(0)

    def plot_inline(self, condition_image, sample_image, true_image):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.tight_layout()
        for im, title, a in zip(
            [condition_image, sample_image, true_image],
            ["Condition", "Sample", "True"],
            ax,
        ):
            a.imshow(im, cmap="gray")
            a.set_title(title)
            # a.axis("off")
        # convert to wandb image
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def draw_strokes(self, strokes):
        image = Image.new("RGB", (256, 256), "white")
        image_draw = ImageDraw.Draw(image)
        for stroke in strokes:
            points = self.scale_points(stroke)
            image_draw.line(points, fill=0)
        return image

    def scale_points(self, points):
        points = [(round(x * 255), round(y * 255)) for x, y in points]
        points = [point for point in points if point[0] > 1e-4 or point[1] > 1e-4]
        return points

    def on_validation_epoch_end(self, trainer, pl_module):
        # Dataloader loads on CPU --> pass to GPU
        x0 = self.x.to(device=pl_module.device)
        
        c = self.c.to(device=pl_module.device)
        c_mask = self.c_mask.to(device=pl_module.device)
        #c = [sc.to(device=pl_module.device) for sc in self.c]
        x0_pred = pl_module.sampling(c,c_mask)
        # replace nan with 0
        x0_pred = torch.nan_to_num(x0_pred, nan=0.0)

        # move arrays back to the CPU for plotting
        x0_pred = x0_pred.cpu()
        if len(x0_pred.shape) == 4:
            # B x 1 x L x 2 -> B x L x 2
            x0_pred = x0_pred.squeeze(1) 
        if len(x0.shape) == 4:
            x0 = x0.squeeze(1)
        x0 = x0.cpu()

        N = self.c.shape[1]
        c = [self.c[:, i, :, :].cpu() for i in range(N)]
        #c = [sc.cpu() for sc in c]

        figs = []
        for i in range(self.bs):
            # extract list of condition strokes
            condition_strokes = [sc[i].tolist() for sc in c]
            pred_stroke = [x0_pred[i].tolist()]
            next_stroke = [x0[i].tolist()]
            # draw images
            condition_image = self.draw_strokes(condition_strokes)
            sample_image = self.draw_strokes(pred_stroke)
            true_image = self.draw_strokes(next_stroke)
            # plot images
            figs.append(self.plot_inline(condition_image, sample_image, true_image))

        # add to logger like so
        trainer.logger.experiment.log({"Sample images": figs})

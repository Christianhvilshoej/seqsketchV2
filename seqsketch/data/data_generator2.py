#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import ndjson
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Grayscale, ToTensor, Compose, Lambda
from torchvision.transforms.functional import invert
from PIL import ImageDraw, Image
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, data_list, max_stroke_length = 32, max_strokes=32,
                 relative = False):
        super().__init__()
        self.data_list = data_list
        self.data_size = (256, 256)
        self.max_stroke_length = max_stroke_length
        self.max_strokes = max_strokes
        self.relative = relative
    def draw_strokes(self, strokes):
        image = Image.new("RGB", self.data_size, "white")
        image_draw = ImageDraw.Draw(image)
        for stroke in strokes:
            points = list(zip(stroke[0], stroke[1]))
            image_draw.line(points, fill=0)
        return image

    def format_strokes_for_model(self, strokes, max_stroke_length=32, max_strokes=32):
        """
        Convert strokes formatted as [[[x-coords], [y-coords]], ...]
        into a list of tensors where each tensor has shape (num_points, 2).
        """
        formatted_strokes = (
            torch.ones(max_strokes, max_stroke_length, 2, dtype=torch.float32) * -1
        )
        mask = torch.zeros(max_strokes, max_stroke_length, 2, dtype=torch.float32)
        # Shorten "long" drawings
        if len(strokes) > max_strokes:
            strokes = strokes[:max_strokes]
        for i, stroke in enumerate(strokes):
            x_coords, y_coords = stroke
            stroke_length = min(len(x_coords), max_stroke_length)
            stroke_torch = torch.tensor(
                list(zip(x_coords, y_coords)), dtype=torch.float32
            )
            # If the stroke is longer than max_length we chop
            if len(x_coords) > max_stroke_length:
                stroke_torch = stroke_torch[:max_stroke_length]
            mask[i, :stroke_length] = 1
            formatted_strokes[i, :stroke_length] = stroke_torch / 255
        return formatted_strokes, mask
    
    def format_strokes_relative(self, strokes, L=32, max_strokes = 24,):
        """
        Convert strokes formatted as [[[x-coords], [y-coords]], ...]
        into a list of tensors where each tensor has shape (num_points, 2).
        """
        formatted_strokes = torch.zeros((max_strokes,L,2))
        mask = torch.zeros(max_strokes, L,2, dtype=torch.float32)
        absolute_points = torch.tensor([[0],[0]])
        for i, stroke in enumerate(strokes):
            stroke = torch.tensor(stroke)
            relative_points = torch.diff(stroke, prepend = absolute_points)
            l = relative_points.shape[-1]
            # New abs points is where the last stroke "ended"
            absolute_points = relative_points.sum(1).view(-1,1)
            padded_points = F.pad(relative_points,pad = (0,L-l), value = 0)
            # 2 x L -> L x 2
            padded_points = padded_points.T
            formatted_strokes[i] = padded_points
            mask[i, :l] = 1
        ## Normalize abs point (but only of mask it not empty)
        if not mask.sum() == 0:
            formatted_strokes[0][0] = 2*formatted_strokes[0][0] - 255
        ## Normalize all points
        formatted_strokes = formatted_strokes / 255
        return formatted_strokes, mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx].copy() 
        if self.relative:
            sample["next_stroke"], sample["next_stroke_mask"] = (
            self.format_strokes_relative(
                sample["next_stroke"], self.max_stroke_length, 1)
                )
            sample["current_strokes"], sample["current_strokes_mask"] = (
            self.format_strokes_relative(
                sample["current_strokes"], self.max_stroke_length, self.max_strokes)
                )   
        else:
            sample["next_stroke"], sample["next_stroke_mask"] = (
                self.format_strokes_for_model(
                    sample["next_stroke"], self.max_stroke_length, 1
                )
            )
            sample["current_strokes"], sample["current_strokes_mask"] = (
                self.format_strokes_for_model(
                    sample["current_strokes"], self.max_stroke_length, self.max_strokes
                )
            )

        return sample


class QuickDrawDataModule2(pl.LightningDataModule):

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self.data_dir = Path(params.data_dir).joinpath(params.category)
        self.category = params.category
        self.val_size = params.val_size
        self.max_samples = params.max_samples
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.max_seq_length = params.max_seq_length
        self.max_strokes = params.max_strokes
        self.relative = params.relative

    def make_data_splitting(self):
        data_split_json = self.data_dir.joinpath("train_test_indices.json")
        with open(data_split_json, "r") as f:
            samples = json.load(f)
        train_samples, val_samples = train_test_split(
            samples["train"], test_size=self.val_size, random_state=42
        )
        self.sample_indices = {
            "train": train_samples,
            "val": val_samples,
            "test": samples["test"],
        }

    def prepare_samples(self, mode="train", seed=42):
        # prepare samples for the given mode
        data_file = self.data_dir.joinpath(f"{self.category}.ndjson")
        with open(data_file, "r") as f:
            data = ndjson.load(f)
        samples = [data[idx] for idx in self.sample_indices[mode]]
        prepared_samples = []
        print(f"Preparing {mode} samples")
        included = 0
        for sample in tqdm(samples):
            strokes = sample["drawing"]
            n_strokes = len(strokes)
            if n_strokes > self.max_strokes:
                continue
            include = True
            for stroke in strokes:
                if len(stroke[0]) > self.max_seq_length:
                    include = False
                    break
            if include:
                for i in range(0, n_strokes):
                    sample = {
                        "n_strokes": n_strokes,
                        "step": i,
                        "next_stroke": strokes[i : i + 1],
                        "current_strokes": strokes[:i],
                    }
                    prepared_samples.append(sample)
                included += 1
        print(f"Included {included} samples")
        # shuffle the samples
        random.seed(seed)
        random.shuffle(prepared_samples)
        return prepared_samples

    def prepare_data(self):
        if self.data_dir.joinpath(f"datasplit.json").exists():
            with open(self.data_dir.joinpath(f"datasplit.json"), "r") as f:
                samples = json.load(f)
            self.train_samples = samples["train"]
            self.val_samples = samples["val"]
        else:
            self.make_data_splitting()
            self.train_samples = self.prepare_samples("train", seed=42)
            self.val_samples = self.prepare_samples("val", seed=43)
            with open(self.data_dir.joinpath(f"datasplit.json"), "w") as f:
                json.dump({"train": self.train_samples, "val": self.val_samples}, f)
        if self.max_samples:
            self.train_samples = self.train_samples[: self.max_samples]
            self.val_samples = self.val_samples[: self.max_samples]

    def setup(self, stage=None):
        # setup for trainer.fit()
        if stage in (None, "fit"):
            self.train_set = ImageDataset(self.train_samples,
                                          relative=self.relative,
                                          max_stroke_length=self.max_seq_length,
                                          max_strokes=self.max_strokes)
            self.val_set = ImageDataset(self.val_samples,
                                        relative=self.relative,
                                        max_stroke_length=self.max_seq_length, 
                                        max_strokes=self.max_strokes)

        # setup for trainer.test()
        if stage == "test":
            self.test_set = ImageDataset(self.train_samples, 
                                         relative=self.relative,
                                         max_stroke_length=self.max_seq_length,
                                         max_strokes=self.max_strokes)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

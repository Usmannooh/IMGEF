import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, IuxrayDatasetProgressive


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)

    # @staticmethod
    # def collate_fn(data):
    #     image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
    #     image_batch = torch.stack(image_batch, 0)
    #     max_seq_length = max(seq_lengths_batch)  # Maximum sequence length in the batch
    #
    #     # Initialize padded target and mask batches
    #     target_batch = torch.zeros((len(report_ids_batch), max_seq_length), dtype=torch.long)
    #     target_masks_batch = torch.zeros((len(report_masks_batch), max_seq_length), dtype=torch.float)
    #
    #     # Populate each entry up to the length of the sequence
    #     for i, report_ids in enumerate(report_ids_batch):
    #         # Ensure report_ids is converted to a flat tensor
    #         report_ids = torch.tensor(report_ids, dtype=torch.long).flatten()
    #         seq_len = len(report_ids)
    #         target_batch[i, :seq_len] = report_ids  # Assign the report_ids to the appropriate row
    #
    #     for i, report_masks in enumerate(report_masks_batch):
    #         # Ensure report_masks is converted to a flat tensor
    #         report_masks = torch.tensor(report_masks, dtype=torch.float).flatten()
    #         seq_len = len(report_masks)
    #         target_masks_batch[i, :seq_len] = report_masks  # Assign the report_masks to the appropriate row
    #
    #     return image_id_batch, image_batch, target_batch, target_masks_batch


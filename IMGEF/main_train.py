# main_train.py

import argparse
import os
import pickle
import numpy as np
import torch
from pytorch_metric_learning.losses import ContrastiveLoss
from models.imgef import imgef
from modules.contrastive_loss import SupConLoss
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores, compute_mlc
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.process_emb import read_word_embeddings
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from utils import *


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default=r'E:\MKCL_usman\usman\R2Gen\data\iu_xray\images', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default=r'E:\MKCL_usman\usman\R2Gen\data\iu_xray\annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--vocab-path', type=str, default=r'E:\MKCL_usman\usman\R2Gen\data\vocab.pkl')
    parser.add_argument('--emb_path', type=str, default=r'E:\MKCL_usman\usman\R2Gen\data\embedding\glove.6B.200d-relativized.txt')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60,help='the maximum sequence length of the reports.')  #60
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')  #16

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')  #101
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,  help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')  #NUM_LYERS
    parser.add_argument('--dropout', type=float, default=0.3, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the number of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')  # 1.0
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')  # EPOCHS
    parser.add_argument('--save_dir', type=str, default=r'E:\MKCL_usman\usman\R2Gen\results\iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default=r'E:\MKCL_usman\usman\R2Gen\records', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=100, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')  # Adam,SGD
    parser.add_argument('--lr_ve', type=float, default=0.0001 , help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='betas for Adam optimizer (momentum and variance tracking).')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='Epsilon value for the Adam optimizer for numerical stability.(Gradients vanishing)')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=4000, help='Zyada warmup steps se model ko initial training phase mein zyada stability milti hai')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.') # 1 se 1.2 or 1.5
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD optimizer.')
    parser.add_argument('--nesterov', type=bool, default=True, help='Use Nesterov momentum in SGD optimizer.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of the learning rate scheduler.') #0.1 se 0.01, 0.05,

    # CyclicLR settings
    # parser.add_argument('--lr_scheduler', type=str, default='CyclicLR', help='the type of the learning rate scheduler.')
    # parser.add_argument('--base_lr', type=float, default=1e-5, help='the minimum learning rate of the cyclic scheduler.')
    # parser.add_argument('--max_lr', type=float, default=7e-4, help='the maximum learning rate of the cyclic scheduler.')
    # parser.add_argument('--step_size_up', type=int, default=2000,help='the number of iterations in the increasing half of a cycle.')
    # parser.add_argument('--mode', type=str, default='triangular2',help='cyclic mode: {triangular, triangular2, exp_range}.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='Set seed for reproducibility of results.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--num_classes', type=int, default=31)  # 21,31,41

    args = parser.parse_args()
    return args


def main():

    args = parse_agrs()


    file_path = r'E:\MKCL_usman\usman\R2Gen\data\embedding\glove.6B.200d-relativized.txt'


    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)



    model = imgef(args, tokenizer, args.num_classe )



    criterion = compute_loss
    criterion_c = SupConLoss(temperature=0.07)
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)


    #build trainer and start to train
    trainer = Trainer(model,  criterion, criterion_c, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, val_dataloader, test_dataloader)

    # # build NEW trainer and start to tr
    # trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
    #                   test_dataloader)

    trainer.train()


if __name__ == '__main__':
    main()


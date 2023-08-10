import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from model import BERTLM, BERT
from .optim_schedule import ScheduledOptim
import os

import tqdm
import logging
logging.basicConfig(filename='train.log', level=logging.DEBUG)

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 ckpt_path : str = None, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, export_path: str = "output/bert_trained.model"):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        print(cuda_devices)
        self.device = torch.device(f"cuda:{cuda_devices[0]}" if cuda_condition else "cpu")
        
        #self.device = torch.device("cuda:0" if cuda_condition else "cpu")
      
        # This BERT model will be saved every epoch
        self.bert = bert
            # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=warmup_steps, 
                                                         num_training_steps=1e6)
       
        self.avg_loss = 0
        self.step = -1
        self.total_correct = 0
        self.total_element = 0
        if ckpt_path != None:
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optim.load_state_dict(ckpt['optim_state_dict'])
                self.optim_schedule.load_state_dict(ckpt['sched_state_dict'])
                self.avg_loss = ckpt['loss']
                self.step = ckpt['step']
                self.total_correct = ckpt['total_correct']
                self.total_element = ckpt['total_element']

            else:
                print("start to train with init model...")
        self.model.to(self.device)
        # else:
        #     self.model = torch.load(model_path).to(self.device)
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
    

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
     
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion1 = nn.NLLLoss()
        self.criterion2 = nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        
        self.export_path = export_path

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.fix_seed(1234)
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.fix_seed(1234)
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        
       

        for i, data in data_iter:
            if i <= self.step:
                continue
            # 0. batch_data will be sent into the device(GPU or cpu)
        
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion1(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion2(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            # 3. backward and optimization only in train
            if train:
                #self.optim_schedule.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.optim_schedule.step()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            self.avg_loss += loss.item()
            self.total_correct += correct
            self.total_element += data["is_next"].nelement()

            
            if i % self.log_freq == 0:
                
                logging.info('Epoch [{}], Step [{}], next_loss[{}], mask_loss[{}], avg_loss: {:.4f}, next_avg_acc: {}, loss: {}, lr: {}'
                     .format(epoch, i, next_loss.item(),mask_loss.item(), self.avg_loss / (i + 1),self.total_correct * 100.0 / self.total_element, loss.item(), 
                             self.optim.param_groups[0]["lr"]))
                self.step = i
                # self.save(self.export_path)
            

        print("EP%d_%s, avg_loss=" % (epoch, str_code), self.avg_loss / len(data_iter), "total_acc=",
              self.total_correct * 100.0 / self.total_element)
            
            # logging.info('Epoch [{}], Step [{}], avg_loss: {:.4f}, total_acc: {}'
            #          .format(epoch, i, avg_loss / len(data_iter) ,total_correct * 100.0 / total_element ))
      

    def save(self, output_path="output/bert.ckpt"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        
        state_dict = {
                    'model_state_dict': self.model.module.cpu().state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'sched_state_dict': self.optim_schedule.state_dict(),
                    'total_correct': self.total_correct,
                    'total_element': self.total_element,
                    'loss': self.avg_loss,
                    'step': self.step,
                }
        torch.save(state_dict, output_path)
        #self.model.module.bert.to(self.device)
        self.model.module.to(self.device)
        print("Model Saved on:" , output_path)
        return output_path
    
    def fix_seed(self, seed):
        import numpy as np
        import random
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

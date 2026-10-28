import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clip_models import CLIPModel
from networks.base_model import BaseModel, init_weights
from networks.SupervisedContrastiveLoss import SupervisedContrastiveLoss


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = CLIPModel(opt)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.SupervisedContrastiveLoss = SupervisedContrastiveLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.class_text = input[1]
        self.label = input[2].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input, self.class_text)

    def get_loss(self):
        return 0.2 * self.loss_fn(self.output["label"].squeeze(1), self.label) + 0.8 * self.SupervisedContrastiveLoss(self.output["features_artifact"], self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

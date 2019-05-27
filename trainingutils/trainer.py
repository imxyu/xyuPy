import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
# from utils.real_time_push import Push

class trainer(object):
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, dataloaders, comments, verbose_train, verbose_val,
                 ckpt_frequency,  max_epochs, checkpoint_dir='ckpt', is_ReduceLRonPlateau=False,
                 max_iter=1e99, start_epoch=0, start_iter=0, device=torch.device('cuda:0'), push=False):
        self.model = model 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_ReduceLRonPlateau = is_ReduceLRonPlateau
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_epochs = max_epochs
        self.max_iter = max_iter
        self.verbose_train = verbose_train
        self.verbose_val = verbose_val
        self.ckpt_frequency = ckpt_frequency
        self.start_epoch = start_epoch
        self.epoch = 0
        self.iter = start_iter
        self.comments = comments
        self.current_val_loss = 0.0
        self.push = push
        # print(self.comments)
        self.writer = SummaryWriter(comment=self.comments)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if push:
            self.p = Push(comment=self.comments)
    
    def train(self):
        for self.epoch in range(self.start_epoch, self.max_epochs):
            if not self.is_ReduceLRonPlateau:
                self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()
            print('Epoch: {}'.format(self.epoch+1))
            print('learning rate: {}'.format(current_lr[-1]))
            should_terminate = self.training_phase(self.dataloaders['train'])
            if should_terminate:
                print('Maximum number of iterations {} exceeded. Finishing training...'.format(self.max_iter))
                break
            # epoch_training_loss, epoch_training_IoU = self.validating_phase(self.dataloaders['train'])
            epoch_val_loss, epoch_val_IoU = self.validating_phase(self.dataloaders['val'])
            self.writer.add_scalar('Val/Loss(end of epoch)', epoch_val_loss, self.epoch+1)
            self.writer.add_scalar('Val/Loss', epoch_val_loss, self.iter)
            self.writer.add_scalar('Val/IoU', epoch_val_IoU, self.iter)
            print('End of the epoch:')
            # print('training loss: {:.16f}'.format(epoch_training_loss))
            # print('training IoU: {:.16f}'.format(epoch_training_IoU))
            print('val loss: {:.16f}'.format(epoch_val_loss))
            print('val IoU: {:.16f}'.format(epoch_val_IoU))
            if self.is_ReduceLRonPlateau:
                self.lr_scheduler.step(self.current_val_loss)
        self.writer.close()

    def training_phase(self, dataloader):
        self.model.train()
        for inputs, targets in dataloader:
            self.iter += 1
            outputs = self.model(inputs.to(self.device))
            loss = self.loss_criterion(outputs, targets.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (self.iter % self.verbose_train) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch+1, self.iter))
                print('training loss: {:.16f}'.format(loss.item()))
                if self.push:
                    self.p.send(self.epoch+1, self.iter, loss.item())

            if (self.iter % self.verbose_val) == 0:
                self.current_val_loss, IoU_val = self.validating_phase(self.dataloaders['val'])
                self.writer.add_scalar('Val/Loss', self.current_val_loss, self.iter)
                self.writer.add_scalar('Val/IoU', IoU_val, self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch+1, self.iter))
                print('val loss: {:.16f}'.format(self.current_val_loss))
                print('val IoU: {:.16f}'.format(IoU_val))
                if self.push:
                    self.p.send(self.epoch+1, self.iter, loss.item(), self.current_val_loss, IoU_val)
            
            if (self.iter % self.ckpt_frequency) == 0:
                checkpoint_name = os.path.join(self.checkpoint_dir, self.comments + 'iter_' + str(self.iter) + '.pth')
                torch.save(self.model.state_dict(), checkpoint_name)

        if self.iter >= self.max_iter:
            return True 
        return False
        
    def validating_phase(self, dataloader):
        self.model.eval()
        sm = nn.Softmax(dim=1)
        loss_total = 0.0
        IoU_total = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs.to(self.device))
                loss = self.loss_criterion(outputs, targets.to(self.device))
                loss_total += loss.item() * inputs.size(0)
                outputs_sm = sm(outputs)
                _, results = torch.max(outputs_sm, dim=1)
                IoU_total += self.eval_criterion(results, targets.to(self.device), is_training=True) * inputs.size(0)
        loss_output = loss_total / dataloader.dataset.__len__()
        IoU_output = IoU_total / dataloader.dataset.__len__()
        self.model.train()
        return loss_output, IoU_output


class trainer_iou(object):
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, dataloaders, comments, verbose_train, verbose_val,
                 ckpt_frequency, max_epochs, checkpoint_dir='ckpt', is_ReduceLRonPlateau=False,
                 max_iter=1e99, start_epoch=0, start_iter=0, device=torch.device('cuda:0')):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_ReduceLRonPlateau = is_ReduceLRonPlateau
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_epochs = max_epochs
        self.max_iter = max_iter
        self.verbose_train = verbose_train
        self.verbose_val = verbose_val
        self.ckpt_frequency = ckpt_frequency
        self.start_epoch = start_epoch
        self.epoch = 0
        self.iter = start_iter
        self.comments = comments
        self.current_val_loss = 0.0
        self.sigmoid = nn.Sigmoid()
        # print(self.comments)
        self.writer = SummaryWriter(comment=self.comments)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train(self):
        for self.epoch in range(self.start_epoch, self.max_epochs):
            if not self.is_ReduceLRonPlateau:
                self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()
            print('Epoch: {}'.format(self.epoch + 1))
            print('learning rate: {}'.format(current_lr[-1]))
            should_terminate = self.training_phase(self.dataloaders['train'])
            if should_terminate:
                print('Maximum number of iterations {} exceeded. Finishing training...'.format(self.max_iter))
                break
            # epoch_training_loss, epoch_training_IoU = self.validating_phase(self.dataloaders['train'])
            epoch_val_loss, epoch_val_IoU = self.validating_phase(self.dataloaders['val'])
            self.writer.add_scalar('Val/Loss(end of epoch)', epoch_val_loss, self.epoch + 1)
            print('End of the epoch:')
            # print('training loss: {:.16f}'.format(epoch_training_loss))
            # print('training IoU: {:.16f}'.format(epoch_training_IoU))
            print('val loss: {:.16f}'.format(epoch_val_loss))
            print('val IoU: {:.16f}'.format(epoch_val_IoU))
            if self.is_ReduceLRonPlateau:
                self.lr_scheduler.step(self.current_val_loss)
        self.writer.close()

    def training_phase(self, dataloader):
        self.model.train()
        for inputs, targets in dataloader:
            self.iter += 1
            outputs = self.model(inputs.to(self.device))
            outputs = self.sigmoid(outputs)
            _, results = torch.max(outputs, dim=1)

            loss = self.loss_criterion(outputs, targets.to(self.device).float())
            # print(outputs)
            # print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (self.iter % self.verbose_train) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch + 1, self.iter))
                print('training loss: {:.16f}'.format(loss.item()))

            if (self.iter % self.verbose_val) == 0:
                self.current_val_loss, IoU_val = self.validating_phase(self.dataloaders['val'])
                self.writer.add_scalar('Val/Loss', self.current_val_loss, self.iter)
                self.writer.add_scalar('Val/IoU', IoU_val, self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch + 1, self.iter))
                print('val loss: {:.16f}'.format(self.current_val_loss))
                print('val IoU: {:.16f}'.format(IoU_val))

            if (self.iter % self.ckpt_frequency) == 0:
                checkpoint_name = os.path.join(self.checkpoint_dir, self.comments + 'iter_' + str(self.iter) + '.pth')
                torch.save(self.model.state_dict(), checkpoint_name)

        if self.iter >= self.max_iter:
            return True
        return False

    def validating_phase(self, dataloader):
        self.model.eval()
        loss_total = 0.0
        IoU_total = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs.to(self.device))
                outputs = self.sigmoid(outputs)
                results = torch.round(outputs.cpu().detach())
                IoU_total += self.eval_criterion(results, targets, is_training=True) * inputs.size(0)
                loss = self.loss_criterion(outputs, targets.to(self.device).float())
                loss_total += loss.item() * inputs.size(0)

        loss_output = loss_total / dataloader.dataset.__len__()
        IoU_output = IoU_total / dataloader.dataset.__len__()
        self.model.train()
        return loss_output, IoU_output







            


            



    


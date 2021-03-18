import torch.nn as nn
import torch


class StarGANLoss():
    def __init__(self, device, lmbd_gp=10, lmbd_cls=5, lmbd_rec=10):
        self.L1 = nn.L1Loss().to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.lmbd_gp = lmbd_gp
        self.lmbd_cls = lmbd_cls
        self.lmbd_rec = lmbd_rec

    def __reconstruction_loss(self, real, reconstructed):
        return self.L1(real, reconstructed)

    def __cls_loss(self, preds, labels):
        return self.BCE(preds, labels.to(torch.float32))

    def __gen_adv_loss(self, d_gen):
        return -d_gen.mean()

    def __disc_adv_loss(self, d_gen, d_real):
        return d_gen.mean() - d_real.mean()

    def generator_loss(self, real, reconstructed, d_gen, d_cls, labels):
        adv_loss = self.__gen_adv_loss(d_gen)
        rec_loss = self.__reconstruction_loss(real, reconstructed)
        cls_loss = self.__cls_loss(d_cls, labels)
        loss_dict = {'Gen/adv_loss': adv_loss.item(), 'Gen/rec_loss': rec_loss, 'Gen/cls_loss': cls_loss}
        return loss_dict, adv_loss + self.lmbd_rec * rec_loss + self.lmbd_cls * cls_loss

    def discriminator_loss(self, real, d_gen, d_real, d_cls, labels, gp):
        adv_loss = self.__disc_adv_loss(d_gen, d_real) + self.lmbd_gp * gp
        cls_loss = self.__cls_loss(d_cls, labels)
        loss_dict = {'Disc/adv_loss': adv_loss.item(), 'Disc/cls_loss': cls_loss.item(), 'Disc/gp_loss': gp.item()}
        return loss_dict, adv_loss + self.lmbd_cls * cls_loss
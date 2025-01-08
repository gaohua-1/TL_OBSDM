import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
from ignite.engine import Engine
import ignite.distributed as idist
from torch import nn, einsum
from einops import rearrange
import cv2
import sys

from transforms import extract_diff
def crop9(img): #crop image
    imgclips = []
    for i in range(3):
        for j in range(3):
            clip = img[:,:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]

            randomx = random.randint(0, 10)

            randomy = random.randint(0, 10)

            clip = clip[:, :,randomx: randomx+64, randomy:randomy+64]

            imgclips.append(clip)
    return imgclips

def crop16(img):

    embx, emby, embz, emba = img.chunk(4, dim=2)

    emb1, emb2, emb3, emb4 = embx.chunk(4, dim=3)

    emb5, emb6, emb7, emb8 = emby.chunk(4, dim=3)
    emb9, emb10, emb11, emb12 = embz.chunk(4, dim=3)
    emb13, emb14, emb15, emb16 = emba.chunk(4, dim=3)
    emb = [emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8, emb9, emb10, emb11, emb12, emb13, emb14, emb15, emb16]
    resultlist = random.sample(range(1, 17), 16)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(resultlist)):

        #print(int(resultlist[i] / 4), int(resultlist[i] % 4))
        x = int(resultlist[i] / 4)
        y = int(resultlist[i] % 4)
        if x == 0:
            if y == 1:
                x1 = i
                continue
            elif y == 2:
                x2 = i
                continue
            elif y == 3:
                x3 = i
                continue
        elif x == 1:
            if y == 0:
                x4 = i
                continue
            elif y == 1:
                x5 = i
                continue
            elif y == 2:
                x6 = i
                continue
            elif y == 3:
                x7 = i
                continue
        elif x == 2:
            if y == 0:
                x8 = i
                continue
            elif y == 1:
                x9 = i
                continue
            elif y == 2:
                x10 = i
                continue
            elif y == 3:
                x11 = i
                continue
        elif x == 3:
            if y == 0:
                x12 = i
                continue
            elif y == 1:
                x13 = i
                continue
            elif y == 2:
                x14 = i
                continue
            elif y == 3:
                x15 = i
                continue
        elif x == 4:
            if y == 0:
                x16 = i
                continue


    c1 = torch.cat((emb[x1], emb[x2], emb[x3], emb[x4]), dim=3)
    c2 = torch.cat((emb[x5], emb[x6], emb[x7], emb[x8]), dim=3)
    c3 = torch.cat((emb[x9], emb[x10], emb[x11], emb[x12]), dim=3)
    c4 = torch.cat((emb[x13], emb[x14], emb[x15], emb[x16]), dim=3)
    c = torch.cat((c1, c2, c3, c4), dim=2)
    return c
class SSObjective:
    def __init__(self, crop=-1, color=-1, flip=-1, blur=-1, rot=-1, sol=-1, only=False):
        self.only = only
        self.params = [
            ('crop',  crop,  4, 'regression'),
            ('color', color, 4, 'regression'),
            ('flip',  flip,  1, 'binary_classification'),
            ('blur',  blur,  1, 'regression'),
            ('rot',    rot,  4, 'classification'),
            ('sol',    sol,  1, 'regression'),
        ]

    def __call__(self, ss_predictor, z1, z2, d1, d2, symmetric=True):
        if symmetric:
            z = torch.cat([torch.cat([z1, z2], 1),
                           torch.cat([z2, z1], 1)], 0)
            d = { k: torch.cat([d1[k], d2[k]], 0) for k in d1.keys() }

        else:
            z = torch.cat([z1, z2], 1)
            d = d1


        losses = { 'total': 0 }
        for name, weight, n_out, loss_type in self.params:
            if weight <= 0:
                continue

            p = ss_predictor[name](z)

            if loss_type == 'regression':
                losses[name] = F.mse_loss(torch.tanh(p), d[name])
            elif loss_type == 'binary_classification':
                losses[name] = F.binary_cross_entropy_with_logits(p, d[name])
            elif loss_type == 'classification':
                losses[name] = F.cross_entropy(p, d[name])
            losses['total'] += losses[name] * weight

        return losses


def prepare_training_batch(batch, t1, t2,t3,t4, device):
    ((x1, w1), (x2, w2),(x3, w3),(x4, w4)), _ = batch
    with torch.no_grad():
        x1 = t1(x1.to(device)).detach()
        x2 = t2(x2.to(device)).detach()
        #two new views
        x3 = t3(x3.to(device)).detach()
        x4 = t4(x4.to(device)).detach()

        x3=crop9(x3)
        x4=crop9(x4)

        diff1 = { k: v.to(device) for k, v in extract_diff(t1, t2, w1, w2).items() }
        diff2 = { k: v.to(device) for k, v in extract_diff(t2, t1, w2, w1).items() }
    return x1, x2,x3,x4, diff1, diff2


def detsim(backbone,
            projector,
            pro1,
            pro2,
            pro3,
            predictor,
            ss_predictor,
            ss_predictor1,
            ss_predictor2,
            ss_predictor3,
            pre1,
            pre2,
            pre3,
            t1,
            t2,
            t3,
            t4,
            optimizers,
            device,
            ss_objective
            ):

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        pro1.train()
        pro2.train()
        pro3.train()
        predictor.train()
        pre1.train()
        pre2.train()
        pre3.train()

        for o in optimizers:
            o.zero_grad()

        x1, x2 ,x3,x4,d1,d2= prepare_training_batch(batch, t1, t2,t3,t4, device)
        y1,y2,y3,y=backbone(x1)

        m1,m2,m3,m=backbone(x2)
        z1 = projector(y)

        zy1,zy2,zy3=pro1(y1),pro2(y2),pro3(y3)

        z2 = projector(m)
        zm1, zm2, zm3 = pro1(m1), pro2(m2), pro3(m3)

        p1=predictor(z1)
        py1, py2, py3 = pre1(zy1), pre2(zy2), pre3(zy3)

        p2=predictor(z2)
        pm1, pm2, pm3 = pre1(zm1), pre2(zm2), pre3(zm3)

        lossx=F.cosine_similarity(p1,z2.detach(), dim=-1).mean().mul(-1)+F.cosine_similarity(p2,z1.detach(), dim=-1).mean().mul(-1)
        lossx1=F.cosine_similarity(pm1,zy1.detach(), dim=-1).mean().mul(-1)+F.cosine_similarity(py1,zm1.detach(), dim=-1).mean().mul(-1)
        lossx2=F.cosine_similarity(pm2,zy2.detach(), dim=-1).mean().mul(-1)+F.cosine_similarity(py2,zm2.detach(), dim=-1).mean().mul(-1)
        lossx3=F.cosine_similarity(pm3,zy3.detach(), dim=-1).mean().mul(-1)+F.cosine_similarity(py3,zm3.detach(), dim=-1).mean().mul(-1)
        loss = 0.4 * lossx + 0.3 * lossx3 + 0.2 * lossx2 + 0.1 * lossx1

        outputs = dict(loss=loss)
        if not ss_objective.only:
            outputs['z1'] = z1
            outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y, m, d1, d2)

        ss_losses1 = ss_objective(ss_predictor1, y1, m1, d1, d2)
        ss_losses2 = ss_objective(ss_predictor2, y2, m2, d1, d2)
        ss_losses3 = ss_objective(ss_predictor3, y3, m3, d1, d2)
        ss_lossesx=0.4*ss_losses['total']+0.3*ss_losses3['total']+0.2*ss_losses2['total']+0.1*ss_losses1['total']

        (loss+ss_lossesx).backward()
        outputs['ss'] = ss_lossesx

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def moco(backbone,
         projector,
         ss_predictor,
         t1,
         t2,
         optimizers,
         device,
         ss_objective,
         momentum=0.999,
         K=65536,
         T=0.2,
         ):

    target_backbone  = deepcopy(backbone)
    target_projector = deepcopy(projector)
    for p in list(target_backbone.parameters())+list(target_projector.parameters()):
        p.requires_grad = False

    queue = F.normalize(torch.randn(K, 128).to(device)).detach()
    queue.requires_grad = False
    queue.ptr = 0

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        target_backbone.train()
        target_projector.train()

        for o in optimizers:
            o.zero_grad()

        x1, x2, d1, d2 = prepare_training_batch(batch, t1, t2, device)
        y1 = backbone(x1)
        z1 = F.normalize(projector(y1))
        with torch.no_grad():
            y2 = target_backbone(x2)
            z2 = F.normalize(target_projector(y2))

        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1).div(T)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)
        outputs = dict(loss=loss, z1=z1, z2=z2)

        ss_losses = ss_objective(ss_predictor, y1, y2, d1, d2)
        (loss+ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        # momentum network update
        for online, target in [(backbone, target_backbone), (projector, target_projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data.mul_(momentum).add_(p1.data, alpha=1-momentum)

        # queue update
        keys = idist.utils.all_gather(z1)
        queue[queue.ptr:queue.ptr+keys.shape[0]] = keys
        queue.ptr = (queue.ptr+keys.shape[0]) % K

        return outputs

    engine = Engine(training_step)
    return engine


def simclr(backbone,
           projector,
           ss_predictor,
           t1,
           t2,
           optimizers,
           device,
           ss_objective,
           T=0.2,
           ):

    def training_step(engine, batch):
        backbone.train()
        projector.train()

        for o in optimizers:
            o.zero_grad()

        x1, x2, d1, d2 = prepare_training_batch(batch, t1, t2, device)
        y1 = backbone(x1)
        y2 = backbone(x2)
        z1 = F.normalize(projector(y1))
        z2 = F.normalize(projector(y2))

        z = torch.cat([z1, z2], 0)
        scores = torch.einsum('ik, jk -> ij', z, z).div(T)
        n = z1.shape[0]
        labels = torch.tensor(list(range(n, 2*n)) + list(range(0, n)), device=scores.device)
        masks = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(2*n):
            masks[i, i] = True
        scores = scores.masked_fill(masks, float('-inf'))
        loss = F.cross_entropy(scores, labels)
        outputs = dict(loss=loss, z1=z1, z2=z2)

        ss_losses = ss_objective(ss_predictor, y1, y2, d1, d2)
        (loss+ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        return outputs

    engine = Engine(training_step)
    return engine




def collect_features(backbone,
                     dataloader,
                     device,
                     normalize=True,
                     dst=None,
                     verbose=False):

    if dst is None:
        dst = device

    backbone.eval()
    with torch.no_grad():
        features = []
        labels   = []
        for i, (x, y) in enumerate(dataloader):
            if x.ndim == 5:
                _, n, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                y = y.view(-1, 1).repeat(1, n).view(-1)
            z = backbone(x.to(device))
            #print('len(z)', len(z))


            if len(z) != 1:
                z = z[3]
            if normalize:
                z = F.normalize(z, dim=-1)
            features.append(z.to(dst).detach())
            labels.append(y.to(dst).detach())
            if verbose and (i+1) % 10 == 0:
                print(i+1)
        features = idist.utils.all_gather(torch.cat(features, 0).detach())
        labels   = idist.utils.all_gather(torch.cat(labels, 0).detach())
        print(features.shape)
        print(labels.shape)

    return features, labels


def nn_evaluator(backbone,
                 trainloader,
                 testloader,
                 device):

    def evaluator():
        backbone.eval()
        with torch.no_grad():
            features, labels = collect_features(backbone, trainloader, device)
            corrects, total = 0, 0
            for x, y in testloader:
                # z = F.normalize(backbone(x.to(device)), dim=-1)
                z = backbone(x.to(device))
                if len(z) != 1:
                    z = z[3]
                z = F.normalize(z, dim=-1)
                scores = torch.einsum('ik, jk -> ij', z, features)
                preds = labels[scores.argmax(1)]

                corrects += (preds.cpu() == y).long().sum().item()
                total += y.shape[0]
            corrects = idist.utils.all_reduce(corrects)
            total = idist.utils.all_reduce(total)

        return corrects / total

    return evaluator


import math

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (hamming_loss, coverage_error,label_ranking_loss,
                             label_ranking_average_precision_score, accuracy_score)

import gcn
from params import *
from dataloader import *


def label_dependency_capture(label_dependency, labels):
    posterior_pro_y_y = 0.
    for j in range(labels.size(0)):
        for k in range(labels.size(0)):
            if int(labels[k]) != int(labels[j]):
                t = label_dependency[int(labels[j]), int(labels[k])]
                posterior_pro_y_y += t
    return posterior_pro_y_y


def train_one_step(net, data, label, optimizer, criterion):
    net.train()
    pred, _ = net(data)
    loss = criterion(torch.sigmoid(pred), label)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss


def train_one_step_correction(device, net, data, labels, optimizer, criterion, delta, beta):
    net.train()
    pred, label_dependency = net(data)
    pred = torch.sigmoid(pred)
    corrected_labels_batch = torch.zeros((labels.size(0), labels.size(1)))

    for j in range(pred.size(0)):
        t_pred = pred[j]
        t_number_labels = torch.nonzero(labels[j]).size(0)
        t_noisy_labels = torch.nonzero(labels[j])
        t_predicted_labels = torch.topk(t_pred, int(t_number_labels)).indices
        original_sc = beta * torch.sum(t_pred[t_noisy_labels]) + (1 - beta) * label_dependency_capture(
            label_dependency[j], t_noisy_labels)
        predicted_sc = beta * torch.sum(t_pred[t_predicted_labels]) + (1 - beta) * label_dependency_capture(
            label_dependency[j], t_predicted_labels)
        sr = original_sc / predicted_sc

        if sr <= delta:
            corrected_labels_batch[j, t_predicted_labels] = 1.
        else:
            corrected_labels_batch[j, t_noisy_labels] = 1.

    loss = criterion(pred.to(device), corrected_labels_batch.to(device))
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss, corrected_labels_batch


def train(device, model, train_loader, optimizer1):
    model.train()
    train_total = 0
    train_loss = 0.
    orginal = []
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        orginal.append(labels)
        train_total += 1
        loss = train_one_step(model, data.float(), labels.float(), optimizer1, nn.BCELoss())
        train_loss += loss
    train_loss_ = train_loss / float(train_total)

    return train_loss_, orginal


def train_after_correction(device, model, train_loader, new_labels, optimizer1, delta, beta):
    model.train()
    train_total = 0
    train_loss = 0.
    corrected_labels = []
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = new_labels[i].to(device)

        train_total += 1
        loss, corrected_labels_batch = train_one_step_correction(device, model, data.float(), labels.float(), optimizer1,
                                                                 nn.BCELoss(), delta, beta)
        train_loss += loss
        corrected_labels.append(corrected_labels_batch)
    train_loss_ = train_loss / float(train_total)

    return train_loss_, corrected_labels


def evaluate(device, test_loader, model1):
    model1.eval()
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)

        if target.dim() == 1:
            target = target.view(-1, 1)

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        for k in range(self.scores.size(1)):
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        sorted, indices = torch.sort(output, dim=0, descending=True)

        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP * 100, OR * 100, OF1 * 100, CP * 100, CR * 100, CF1 * 100


def compute_cover(labels, outputs):
    n_labels = labels.shape[1]
    loss = coverage_error(labels, outputs)

    return (loss-1)/n_labels


def test(device, net, loader, criterion=torch.nn.BCELoss(), return_map=False):
    running_loss = 0

    net.eval()
    target = []
    pred_list = []
    output = []

    ap_meter = AveragePrecisionMeter()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device).float(), y.to(device).float()

        with torch.no_grad():
            out_1, _ = net(X)

            ap_meter.add(out_1.cpu().detach(), y.cpu())

            out = torch.sigmoid(out_1)

            y[y == 0] = 1
            y[y == -1] = 0
            loss = criterion(out, y)

            pred_list.append((out > 0.5).cpu().detach().numpy())
            output.append(out.cpu().detach().numpy())
            target.append(y.cpu().detach().numpy())

        running_loss += loss.item()

    target = np.concatenate(target)
    output = np.concatenate(output)
    preds = np.concatenate(pred_list)

    learn_loss = running_loss / len(loader)

    hloss = hamming_loss(target, preds)
    rloss = label_ranking_loss(target, output)
    cover = compute_cover(target, output)
    avgpre = label_ranking_average_precision_score(target, output)

    top_label = np.argmax(output, axis=-1)
    oneerror = sum(1 - target[range(len(loader.dataset)), top_label]) / len(loader.dataset)

    acc = accuracy_score(target, preds)

    map = 100 * ap_meter.value().mean().cpu().detach().numpy()
    op, or_, of1, cp, cr, cf1 = ap_meter.overall()

    if return_map:
        return (learn_loss, hloss, rloss, cover, avgpre, oneerror, acc), (map, op, or_, of1, cp, cr, cf1)
    else:
        return learn_loss, hloss, rloss, cover, avgpre, oneerror, acc


def main():
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'>>> Using {device} device <<<')

    print('>>> Loading datasets <<<')
    train_dataset, val_dataset, test_dataset = load_data(image_size)
    train_loader = dataloader(train_dataset, batch_size)
    val_loader = dataloader(val_dataset, batch_size)
    test_loader = dataloader(test_dataset, batch_size)

    print('>>> Building model <<<')
    clf1 = gcn.get_model(num_classes)
    clf1 = nn.DataParallel(clf1)
    clf1.to(device)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    corrected_labels = []
    for e in range(epoch):
        clf1.train()

        delta = hyper_parameter_delta * max(0, 0.2 * (10 - e))
        print('>>> Training <<<')
        if e < epoch_update_start:
            train_loss, corrected_labels = train(device, clf1, train_loader, optimizer1)
        else:
            train_loss, corrected_labels = train_after_correction(device, clf1, train_loader, corrected_labels,
                                                                  optimizer1, delta, hyper_parameter_beta)

        val_1, val_2 = test(clf1, val_loader, return_map=True)
        val_loss, v_hloss, v_rloss, cover, avgpre, oneerror, acc = val_1
        map, op, or_, of1, cp, cr, cf1 = val_2

        test_1, test_2 = test(clf1, test_loader, return_map=True)

        test_loss, t_hloss, t_rloss, cover_, avgpre_, oneerror_, acc_ = test_1
        map_, op_, or__, of1_, cp_, cr_, cf1_ = test_2

        print(f'EPOCH: {e} | val_loss: {val_loss, 5} | hloss: {round(v_hloss, 5)} | rloss: {round(v_rloss, 5)} | '
              f'cover: {round(cover, 5)} | oneerror: {round(oneerror, 5)} | avgpre: {round(avgpre, 5)} | acc: {round(acc, 5)}')
        print(f'EPOCH: {e} | val_map: {map, 5} | op: {round(op, 5)} | or: {round(or_, 5)} | '
              f'of1: {round(of1, 5)} | cp: {round(cp, 3)} | cr: {round(cr, 5)} | cf1: {round(cf1, 5)}')
        print(f'EPOCH: {e} | test_loss: {test_loss, 5} | hloss: {round(t_hloss, 5)} | rloss: {round(t_rloss, 5)} | '
              f'cover: {round(cover_, 5)} | oneerror: {round(oneerror, 5)} | avgpre: {round(avgpre, 5)} | acc: {round(acc, 5)}')
        print(f'EPOCH: {e} | test_map: {map_, 5} | op: {round(op_, 5)} | or: {round(or__, 5)} | '
              f'of1: {round(of1_, 5)} | cp: {round(cp_, 3)} | cr: {round(cr_, 5)} | cf1: {round(cf1_, 5)}')


if __name__ == '__main__':
    main()

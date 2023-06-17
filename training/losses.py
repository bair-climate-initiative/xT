from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn, topk
from torch.nn import NLLLoss2d, MSELoss, BCEWithLogitsLoss


class LossCalculator(ABC):

    @abstractmethod
    def calculate_loss(self, outputs, sample):
        pass


def r2_loss(output, target):
    eps = 1e-6
    ss_tot = torch.sum((target - target.mean()) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = (ss_res + eps) / (ss_tot.clamp(1.) + eps)
    return r2


class R2Loss(LossCalculator):

    def __init__(self, field):
        super().__init__()
        self.field = field
        self.mse = MSELoss()

    def calculate_loss(self, outputs, sample):
        targets = sample[self.field].cuda().float()
        mask = targets > 0
        pred = outputs[self.field]
        mse = self.mse(pred[targets >= 0], targets[targets >= 0])
        if torch.sum(mask).item() > 0:
            return r2_loss(pred[mask], targets[mask]) + 0.001 * mse
        else:
            return 0


class ComboLossCalculator(LossCalculator):
    def __init__(self, field: str, **kwargs):
        super().__init__()
        self.field = field
        self.loss = ComboLoss(**kwargs)

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = outputs[self.field].float()
            targets = sample[self.field].cuda().float()
            return self.loss(outputs, targets)


class BceMaskLoss(LossCalculator):
    def __init__(self, field: str):
        super().__init__()
        self.field = field
        self.loss = BCEWithLogitsLoss()

    def calculate_loss(self, outputs, sample):
        outputs = outputs[self.field].view(-1)
        targets = sample[self.field].cuda().float().view(-1)
        non_ignored = targets < 200
        outputs = outputs[non_ignored]
        targets = targets[non_ignored]
        mask = targets > 0
        if torch.sum(mask).item() > 0:
            return self.loss(outputs[mask], targets[mask])
        return self.loss(outputs, targets)


class CenterLossCalculator(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = MSELoss(**kwargs, reduction='none')

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            targets = sample["center_mask"].cuda().float().view(-1)

            pred = outputs["center_mask"].float().view(-1)
            mse = self.mse(pred[targets >= 0], targets[targets >= 0])

            if torch.sum(mse > 0.1).item() > 0:
                res = mse[mse > 0.1].sum()
            else:
                res = mse.mean()
            return res


class LengthLoss(LossCalculator):

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = MSELoss(**kwargs)

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            targets = sample["length_mask"].cuda().float()
            mask = targets > 0
            pred = outputs["length_mask"].float()
            if torch.sum(targets >= 0).item() == 0:
                return 0 * pred.mean()
            return (torch.abs(pred[mask] - targets[mask]) / targets[mask]).mean()

def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues, reduce=False)


def soft_dice_loss(outputs, targets):
    eps = 1e-5
    dice_target = targets.contiguous().float()
    dice_output = outputs.contiguous().float()
    intersection = torch.sum(dice_output * dice_target)
    union = torch.sum(dice_output) + torch.sum(dice_target) + eps
    if union < 5:
        return 0
    loss = (1 - (2 * intersection + eps) / union)
    return loss.mean()


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5, reduce=True):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images
    if reduce:
        return losses.mean()
    else:
        return losses


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return soft_dice_loss(input, target)


class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        x = soft_dice_loss(input, target)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)


class NoiseRobustDice(nn.Module):
    def __init__(self, beta=1.5):
        super().__init__()
        self.beta = beta

    def forward(self, input, target):
        eps = 1e-4
        input = input.view(-1)
        target = target.view(-1)
        numerator = torch.sum(torch.pow(torch.abs(target - input), self.beta))
        denominator = torch.sum(torch.square(target) + torch.square(input)) + eps
        loss = numerator / denominator
        return loss.mean()


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, input, target):
        sigmoid_input = torch.sigmoid(input)
        return self.bce(input, target) + soft_dice_loss(sigmoid_input, target)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, skip_empty=False, channel_weights=[1] * 15, channel_losses=None):
        super().__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lcdice = LogCoshDiceLoss()
        self.nrdice = NoiseRobustDice()
        self.jaccard = JaccardLoss(per_image=per_image)
        self.focal = BinaryFocalLoss()
        self.weights = weights
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'lcdice': self.lcdice,
                        'nrdice': self.nrdice,
                        'focal': self.focal,
                        'jaccard': self.jaccard}
        self.expect_sigmoid = {'dice', 'jaccard', 'lcdice', "nrdice"}
        self.per_channel = {'dice', 'jaccard', 'lcdice', "nrdice"}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses
        self.skip_empty = skip_empty

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = F.logsigmoid(outputs).exp()
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        if self.skip_empty and torch.sum(targets[:, c, ...]) < 50:
                            continue
                        c_sigmoid_input = sigmoid_input[:, c, ...].view(-1)
                        c_targets = targets[:, c, ...].view(-1)
                        c_outputs = outputs[:, c, ...].view(-1)
                        non_ignored = c_targets.view(-1) < 200
                        c_sigmoid_input = c_sigmoid_input[non_ignored]
                        c_targets = c_targets[non_ignored]
                        c_outputs = c_outputs[non_ignored]
                        val += self.channel_weights[c] * self.mapping[k](
                            c_sigmoid_input if k in self.expect_sigmoid else c_outputs, c_targets)

            else:
                non_ignored = targets.view(-1) < 200

                val = self.mapping[k](
                    sigmoid_input.view(-1)[non_ignored] if k in self.expect_sigmoid else outputs.view(-1)[non_ignored],
                    targets.view(-1)[non_ignored])

            self.values[k] = val
            loss += self.weights[k] * val
        return loss


class FocalDiceLossCalculator(ABC):
    def __init__(self, **kwargs):
        super().__init__()

        self.loss = FocalLossWithDice(**kwargs)

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = outputs["conf_mask"].float()
            targets = sample["conf_mask"].cuda().long()
            return self.loss(outputs, targets)


class FocalLossWithDice(nn.Module):
    def __init__(self, num_classes, ignore_index=255, gamma=2, ce_weight=10., d_weight=1., weight=None,
                 size_average=True):
        super().__init__()
        self.num_classes = num_classes
        self.d_weight = d_weight
        self.ce_w = ce_weight
        self.gamma = gamma
        if weight is not None:
            weight = torch.Tensor(weight).float()
        self.nll_loss = NLLLoss2d(weight, size_average, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        probas = F.softmax(outputs, dim=1)
        ce_loss = self.nll_loss((1 - probas) ** self.gamma * F.log_softmax(outputs, dim=1), targets)
        d_loss = soft_dice_loss_mc(outputs, targets, self.num_classes, ignore_index=self.ignore_index)
        return self.ce_w * ce_loss + self.d_weight * d_loss


def soft_dice_loss_mc(outputs, targets, num_classes, per_image=False, only_existing_classes=False, ignore_index=255,
                      minimum_class_pixels=3, reduce_batch=True):
    batch_size = outputs.size()[0]
    eps = 1e-5
    outputs = F.softmax(outputs, dim=1)

    def _soft_dice_loss(outputs, targets):
        loss = 0
        non_empty_classes = 0
        for cls in range(1, num_classes):
            non_ignored = targets.view(-1) != ignore_index
            dice_target = (targets.view(-1)[non_ignored] == cls).float()
            dice_output = outputs[:, cls].contiguous().view(-1)[non_ignored]

            intersection = (dice_output * dice_target).sum()
            if dice_target.sum() > minimum_class_pixels:
                union = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - (2 * intersection + eps) / union)
                non_empty_classes += 1
        if only_existing_classes:
            loss /= (non_empty_classes + eps)
        else:
            loss /= (num_classes - 1)
        return loss

    if per_image:
        if reduce_batch:
            loss = 0
            for i in range(batch_size):
                loss += _soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0))
            loss /= batch_size
        else:
            loss = torch.Tensor(
                [_soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0)) for i in
                 range(batch_size)])
    else:
        loss = _soft_dice_loss(outputs, targets)

    return loss


class CropJaccardLossCalculator(LossCalculator):
    def __init__(self, ohem_fraction=0.5, bce_weight=2):
        super().__init__()
        self.ohem_fraction = ohem_fraction
        self.bce = BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def calculate_loss(self, outputs, sample):
        outputs = outputs["object_mask"]
        targets = sample["object_mask"].cuda().float()
        crops = sample["crop_coords"].cpu().numpy()
        bs = targets.size(0)
        out_crops = []
        all_crops = []
        for i in range(bs):
            for crop in crops[i]:
                x1, y1, x2, y2 = crop
                if x1 == 0:
                    break
                all_crops.append(targets[i, :, y1:y2, x1:x2])
                out_crops.append(outputs[i, :, y1:y2, x1:x2])

        if len(all_crops) < 2:
            return 0

        out_crops_raw = torch.cat(out_crops, dim=0)

        out_crops = torch.sigmoid(out_crops_raw)
        all_crops = torch.cat(all_crops, dim=0)
        bce_loss = self.bce_weight * self.bce(out_crops_raw, all_crops)
        jac_losses = jaccard(out_crops, all_crops, per_image=True, reduce=False)
        k = int(len(all_crops) * self.ohem_fraction)
        jac_loss = topk(jac_losses, k=k, sorted=False)[0].mean()
        return jac_loss + bce_loss


class CropCenterLossCalculator(LossCalculator):
    def __init__(self, ohem_fraction=0.5, ):
        super().__init__()
        self.ohem_fraction = ohem_fraction
        self.mse = MSELoss(reduction='none')

    def calculate_loss(self, outputs, sample):
        outputs = outputs["center_mask"]
        targets = sample["center_mask"].cuda().float()
        crops = sample["crop_coords"].cpu().numpy()
        bs = targets.size(0)
        out_crops = []
        all_crops = []
        for i in range(bs):
            for crop in crops[i]:
                x1, y1, x2, y2 = crop
                if x1 == 0:
                    break
                all_crops.append(targets[i, :, y1:y2, x1:x2])
                out_crops.append(outputs[i, :, y1:y2, x1:x2])

        if len(all_crops) < 2:
            return 0

        out_crops = torch.cat(out_crops, dim=0)

        all_crops = torch.cat(all_crops, dim=0)
        k = int(len(all_crops) * self.ohem_fraction)
        mse_loss = topk(self.mse(out_crops.flatten(1), all_crops.flatten(1)).mean(1), k=k, sorted=False)[0].mean()
        return mse_loss


class CrossEntrophy(LossCalculator):

    def __init__(self, field):
        super().__init__()
        self.field = field
        self.ce = nn.CrossEntropyLoss()

    def calculate_loss(self, outputs, sample):
        targets = sample[self.field].cuda().long() # Label map
        pred = outputs[self.field]
        return self.ce(pred,targets)
"""Train a small U-Net on the supplied ZIP. CPU training is sufficient."""
import argparse
import copy
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from data import frames


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs),
                         nn.ReLU(), nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.BatchNorm2d(outputs), nn.ReLU())


class YellowUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = block(3, 8)
        self.mid = block(8, 12)
        self.bottom = block(12, 16)
        self.dec2 = block(28, 12)
        self.dec1 = block(20, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        a = self.enc(x)
        b = self.mid(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, scale_factor=2), b), 1))
        return self.out(self.dec1(torch.cat((F.interpolate(d, scale_factor=2), a), 1)))


def iou(pred, target):
    return (pred & target).sum().item() / max((pred | target).sum().item(), 1)


def load_split(archive, split):
    rgb, mask = zip(*frames(archive, split))
    return (torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2),
            torch.from_numpy(np.stack(mask))[:, None])


@torch.no_grad()
def scores(net, x):
    net.eval()
    return torch.cat([net(batch.float()/255).sigmoid() for batch in x.split(8)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--out', type=Path, default=Path(__file__).parent/'models')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    args.out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with ZipFile(args.archive) as z:
        x, y = load_split(z, 'train')
        vx, vy = load_split(z, 'validation')
        tx, ty = load_split(z, 'test')
    net = YellowUNet().to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    best = -1.
    for epoch in range(args.epochs):
        net.train()
        for ids in torch.randperm(len(x)).split(8):
            images, labels = x[ids].float()/255, y[ids].float()
            if torch.rand(()).item() < .5:
                images, labels = images.flip([-1]), labels.flip([-1])
            images = (images * (.85 + .3*torch.rand(len(ids), 1, 1, 1))).clamp(0, 1)
            logits = net(images)
            probs = logits.sigmoid()
            dice = (2*(probs*labels).sum()+1) / (probs.sum()+labels.sum()+1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(8.)) + 1-dice
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = scores(net, vx)
        value, cutoff = max((iou(val >= t, vy), t) for t in (.4,.5,.6,.7,.8,.9))
        print(f'Epoch {epoch+1}: validation IoU {value:.3f}', flush=True)
        if value > best:
            best, best_epoch, threshold = value, epoch+1, cutoff
            weights = copy.deepcopy(net.state_dict())
    net.load_state_dict(weights)
    net.eval()
    # Fold the selected probability threshold into the exported binary output.
    class Segmenter(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, rgb):
            x = rgb.permute(2,0,1).unsqueeze(0).float()/255.
            return (self.model(x)[0,0].sigmoid() >= threshold).to(torch.uint8)

    model = Segmenter(net).eval()
    filename = args.out/'white_line.onnx'
    torch.onnx.export(model, tx[0].permute(1,2,0).contiguous(), str(filename),
                      opset_version=17, input_names=['rgb'], output_names=['white_mask'])
    onnx.checker.check_model(onnx.load(filename))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(filename), options, providers=['CPUExecutionProvider'])
    expected = scores(net, tx)[:,0] >= threshold
    masks = torch.from_numpy(np.stack([session.run(None, {'rgb': im.permute(1,2,0).numpy()})[0]
                                       for im in tx])).bool()
    diff = int((expected != masks).sum())
    if diff / masks.numel() > 0.0001:
        raise RuntimeError('ONNX masks differ from PyTorch by more than 0.01% of pixels')
    metrics = dict(train=len(x), validation=len(vx), test=len(tx), epochs=args.epochs,
                   selected_epoch=best_epoch, threshold=threshold, validation_iou=best,
                   test_onnx_iou=iou(masks, ty[:,0]), onnx_pytorch_different_pixels=diff,
                   compared_pixels=masks.numel())
    (args.out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(metrics)
    # First three held-out frames: RGB, reference mask, predicted mask over image.
    rows = []
    for im, ref, pred in zip(tx[:3], ty[:3,0], masks[:3]):
        rgb = im.permute(1,2,0).numpy()
        overlay = rgb.copy()
        overlay[pred.numpy()] = (overlay[pred.numpy()]*.4 + np.array([255,0,180])*.6).astype('uint8')
        rows.append(np.concatenate((rgb, np.repeat(ref.numpy()[...,None],3,axis=2).astype('uint8')*255, overlay),axis=1))
    Image.fromarray(np.concatenate(rows,axis=0)).save(args.out/'test_examples.png')


if __name__ == '__main__':
    main()

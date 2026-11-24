"""Compute ALL segmentation and lane-centre metrics for both U-Nets from the labelled ZIP.

No training, no PyTorch needed (NumPy, Pillow, ONNX Runtime).

    python3 evaluate_all.py /path/turtlebot3_autorace_dataset_to_fill.zip
    python3 evaluate_all.py /path/dataset.zip --out results_all

Writes <out>/metrics_all.json and <out>/metrics_all.md (tables to paste in the report).
"""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from data import frames  # noqa: E402
from tb3_lane_center.vision import lane_geometry  # noqa: E402

SPLITS = ('train', 'validation', 'test')
COLOURS = ('yellow', 'white')


def scores(c):
    tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
    d = lambda a, b: a / b if b else float('nan')
    return dict(
        images=c['images'], iou=d(tp, tp + fp + fn), dice_f1=d(2 * tp, 2 * tp + fp + fn),
        precision=d(tp, tp + fp), recall=d(tp, tp + fn), specificity=d(tn, tn + fp),
        pixel_accuracy=d(tp + tn, tp + fp + fn + tn),
        mean_iou_line_and_background=(d(tp, tp + fp + fn) + d(tn, tn + fp + fn)) / 2,
        confusion_matrix=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--out', type=Path, default=Path('results_all'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = Path(__file__).parent / 'models'
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    nets = {c: ort.InferenceSession(str(models / f'{c}_line.onnx'), opt, providers=['CPUExecutionProvider'])
            for c in COLOURS}

    zero = lambda: dict(tp=0, fp=0, fn=0, tn=0, images=0)
    total = {c: {s: zero() for s in SPLITS} for c in COLOURS}
    centre = {s: dict(frames=0, label_pairs=0, predicted_pairs=0, matched=0, predicted_only=0,
                      missed=0, errors_px=[]) for s in SPLITS}

    with ZipFile(args.archive) as z:
        for split in SPLITS:
            ys = frames(z, split, 'yellow_line')
            ws = frames(z, split, 'white_line')
            for (im, ylab), (_, wlab) in zip(ys, ws):
                pred = {}
                for c, lab in (('yellow', ylab), ('white', wlab)):
                    m = nets[c].run(None, {'rgb': im})[0].astype(bool)
                    pred[c] = m
                    t = total[c][split]
                    t['tp'] += int((m & lab).sum()); t['fp'] += int((m & ~lab).sum())
                    t['fn'] += int((~m & lab).sum()); t['tn'] += int((~m & ~lab).sum())
                    t['images'] += 1
                ref = lane_geometry(ylab, wlab)
                got = lane_geometry(pred['yellow'], pred['white'])
                cs = centre[split]
                cs['frames'] += 1
                cs['label_pairs'] += ref is not None
                cs['predicted_pairs'] += got is not None
                if ref and got:
                    cs['matched'] += 1
                    cs['errors_px'].append(abs(sum(got[0][1:]) - sum(ref[0][1:])) / 2)
                elif got and not ref:
                    cs['predicted_only'] += 1
                elif ref and not got:
                    cs['missed'] += 1

    result = {'segmentation': {c: {s: scores(total[c][s]) for s in SPLITS} for c in COLOURS},
              'lane_centre': {}}
    for s, cs in centre.items():
        e = np.array(cs.pop('errors_px'))
        cs.update(mae_px=float(e.mean()) if len(e) else None,
                  rmse_px=float(np.sqrt((e ** 2).mean())) if len(e) else None,
                  median_px=float(np.median(e)) if len(e) else None,
                  max_px=float(e.max()) if len(e) else None,
                  mae_percent_of_width=float(e.mean() / 256 * 100) if len(e) else None)
        result['lane_centre'][s] = cs
    (args.out / 'metrics_all.json').write_text(json.dumps(result, indent=2))

    f = lambda v: '-' if v is None or v != v else f'{v:.3f}'
    md = ['## Segmentation (pixel level, threshold folded into the ONNX model)', '',
          '| Network | Split | Images | IoU | Dice/F1 | Precision | Recall | Specificity | Pixel acc. | mIoU (line+bg) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in COLOURS:
        for s in SPLITS:
            r = result['segmentation'][c][s]
            md.append(f"| {c} | {s} | {r['images']} | {f(r['iou'])} | {f(r['dice_f1'])} | {f(r['precision'])} | "
                      f"{f(r['recall'])} | {f(r['specificity'])} | {f(r['pixel_accuracy'])} | {f(r['mean_iou_line_and_background'])} |")
    md += ['', '## Confusion matrices (pixels)', '', '| Network | Split | TP | FP | FN | TN |', '| --- | --- | --- | --- | --- | --- |']
    for c in COLOURS:
        for s in SPLITS:
            m = result['segmentation'][c][s]['confusion_matrix']
            md.append(f"| {c} | {s} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
    md += ['', '## Lane centre (near band, 256 px wide image)', '',
           '| Split | Frames | Label pairs | Predicted pairs | Matched | Missed | Predicted only | MAE px | RMSE px | Median px | Max px | MAE % width |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in SPLITS:
        r = result['lane_centre'][s]
        md.append(f"| {s} | {r['frames']} | {r['label_pairs']} | {r['predicted_pairs']} | {r['matched']} | {r['missed']} | "
                  f"{r['predicted_only']} | {f(r['mae_px'])} | {f(r['rmse_px'])} | {f(r['median_px'])} | {f(r['max_px'])} | {f(r['mae_percent_of_width'])} |")
    (args.out / 'metrics_all.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nSaved {args.out}/metrics_all.json and {args.out}/metrics_all.md')


if __name__ == '__main__':
    main()

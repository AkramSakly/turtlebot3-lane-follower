"""Check lane midpoint against same-row ground-truth polygons in held-out frames."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import onnxruntime as ort
from data import frames
from tb3_lane_center.vision import lane_geometry

models = Path(__file__).parent/'models'
opt = ort.SessionOptions()
opt.intra_op_num_threads = 2
nets = [ort.InferenceSession(str(models/(color+'_line.onnx')), opt, providers=['CPUExecutionProvider'])
        for color in ('yellow','white')]
errors = []
eligible = detected = unexpected = total = 0
with ZipFile(sys.argv[1]) as archive:
    for (im, y), (_, w) in zip(frames(archive,'test','yellow_line'), frames(archive,'test','white_line')):
        ref = lane_geometry(y,w)
        pred = lane_geometry(*(net.run(None,{'rgb':im})[0].astype(bool) for net in nets))
        total += 1
        eligible += ref is not None
        detected += pred is not None
        if pred and ref:
            errors.append(abs(sum(pred[0][1:])-sum(ref[0][1:]))/2)
        if pred and not ref:
            unexpected += 1
result = dict(frames=total, reference_pairs=eligible, predicted_pairs=detected, matched=len(errors),
              predicted_only=unexpected, near_center_mae_pixels=float(np.mean(errors)) if errors else None)
print(json.dumps(result,indent=2))

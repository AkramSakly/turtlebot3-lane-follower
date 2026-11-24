import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from tb3_lane_center.vision import SIZE


def frames(archive, split, label_name="white_line"):
    files = set(archive.namelist())
    held_out = set()
    if split == "train":
        for name in files:
            if name.endswith(".png") and ("/test/images/" in name or "/validation/images/" in name):
                held_out.add(hashlib.sha256(archive.read(name)).digest())
    for path in sorted(files):
        if f'/{split}/images/' not in path or not path.endswith('.png'):
            continue
        root, rel = path.split('/images/')
        label = root + '/labels/' + rel[:-4] + '.json'
        if label not in files and split == 'test':
            label = root + '/labels/' + '/'.join(rel.split('/')[-2:])[:-4] + '.json'
        if label not in files:
            continue
        payload = archive.read(path)
        if held_out and hashlib.sha256(payload).digest() in held_out:
            continue
        ann = json.loads(archive.read(label))
        im = Image.open(io.BytesIO(payload)).convert('RGB')
        assert im.size == (ann['imageWidth'], ann['imageHeight']), path
        mask = Image.new('L', im.size)
        for shape in ann['shapes']:
            if shape['label'] == label_name:
                assert shape['shape_type'] == 'polygon'
                ImageDraw.Draw(mask).polygon([tuple(p) for p in shape['points']], fill=1)
        yield np.asarray(im.resize(SIZE, getattr(Image, "Resampling", Image).BILINEAR)), np.array(mask.resize(SIZE, getattr(Image, 'Resampling', Image).NEAREST), dtype=bool)



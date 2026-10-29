#!/usr/bin/env python3
"""Export checked NudeNet weights to Core ML with built-in NMS.

Explicit two-output contract: confidence [N,18-or-80] and coordinates [N,4].
Different Ultralytics versions can change the export; fail the build if so.
"""
from pathlib import Path
from ultralytics import YOLO
import coremltools as ct

checkpoint = Path('320n.pt')
if not checkpoint.is_file():
    raise SystemExit('Missing 320n.pt')
model = YOLO(str(checkpoint))
names = model.names
if not isinstance(names, dict):
    names = dict(enumerate(names))
expected = {2: 'BUTTOCKS_EXPOSED', 3: 'FEMALE_BREAST_EXPOSED',
            4: 'FEMALE_GENITALIA_EXPOSED', 6: 'ANUS_EXPOSED',
            14: 'MALE_GENITALIA_EXPOSED'}
assert len(names) == 18, f'Unexpected NudeNet class count: {len(names)}'
for index, name in expected.items():
    assert names[index] == name, f'Unexpected label at {index}: {names[index]!r}'

converted = Path(model.export(format='coreml', imgsz=320, nms=True,
                              dynamic=False, batch=1))
assert converted.exists(), f'Missing exported Core ML model: {converted}'
model_spec = ct.models.MLModel(str(converted), skip_model_load=True).get_spec()
outputs = {out.name: out for out in model_spec.description.output}
assert set(outputs) == {'confidence', 'coordinates'}, \
    f'Expected NMS confidence and coordinates, got {list(outputs)}'
for name, expected_widths in [('coordinates', {4}), ('confidence', {18, 80})]:
    out = outputs[name]
    assert out.type.WhichOneof('Type') == 'multiArrayType', f'{name} not MLMultiArray'
    dims = list(out.type.multiArrayType.shape)
    ranges = out.type.multiArrayType.shapeRange.sizeRanges
    width = dims[-1] if dims else (ranges[-1].lowerBound if ranges else 0)
    assert width in expected_widths, f'{name} output channel size {width} unexpected (shape={dims})'
    print(f'PASS {name}: output width {width}, shape {dims}')
print('PASS: NudeNet labels and Core ML built-in NMS output contract')

import sys,json
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/cryoem_datasets/EMPIAR-10028/experiments/ann_yolo26m_cryo_tiles_v3/code')
from common import *
from ultralytics import YOLO
sys.path.insert(0,str(DATA/'metadata/three_method_evaluation'))
from evaluate_empiar10028_v3_pretrained_val import Point
from common_test import frozen,sha as file_sha,dump as save_json,E as TEST_EXP,M as TEST_MAN,names as test_names,prediction_rows
TEST_NAMES=test_names()

def write_csv(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['micrograph_name'])
        w.writeheader()
        w.writerows(rows)
def infer_ann_test(weight, out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    m = YOLO(str(weight))
    m.model.model[-1].max_det = 2000
    ns = TEST_NAMES
    preds = {}
    allrows = []
    info = []
    for name in ns:
        image, meta = normalize_mrc(DATA / 'micrographs' / name)
        candidates = []
        tile_counts = []
        for iy, y0 in enumerate(STARTS):
            for ix, x0 in enumerate(STARTS):
                tile = image[y0:y0 + SIZE, x0:x0 + SIZE]
                tensor = torch.from_numpy(np.repeat(tile[None, None], 3, axis=1)).to('cuda')
                r = m.predict(tensor, imgsz=SIZE, device=0, conf=1e-05, max_det=2000, verbose=False)[0]
                xywh = r.boxes.xywh.detach().cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                tile_counts.append({'x0': x0, 'y0': y0, 'candidates': len(scores), 'at_max_det': len(scores) >= 2000})
                for (x, y, w, h), score in zip(xywh, scores):
                    xx, yy = (float(x) + x0, float(y) + y0)
                    if BOUNDS[ix] <= xx < BOUNDS[ix + 1] and BOUNDS[iy] <= yy < BOUNDS[iy + 1]:
                        candidates.append((xx, yy, float(score)))
        keep = []
        grid = {}
        for x, y, s in sorted(candidates, key=lambda v: (-v[2], v[0], v[1])):
            gx, gy = (int(x // 28), int(y // 28))
            nearby = [v for xx in range(gx - 1, gx + 2) for yy in range(gy - 1, gy + 2) for v in grid.get((xx, yy), [])]
            if all(((x - a) ** 2 + (y - b) ** 2 > 28 ** 2 for a, b in nearby)):
                keep.append((x, y, s))
                grid.setdefault((gx, gy), []).append((x, y))
        preds[name] = [Point(x, y, s, j) for j, (x, y, s) in enumerate(keep)]
        allrows += [dict(method='ANN-YOLO26', split='test', micrograph_name=name, x_original_px=x, y_original_px=y, score=s) for x, y, s in keep]
        info.append(dict(micrograph_name=name, before_center_merge=len(candidates), after_center_merge=len(keep), normalization=meta, tile_counts=tile_counts))
        print('INFERENCE', name, len(keep), flush=True)
    write_csv(out / 'predictions.csv', allrows)
    write_json(out / 'inference.json', {'checkpoint': str(weight), 'checkpoint_sha256': sha(weight), 'score_floor': 1e-05, 'tile_max_det': 2000, 'ownership_regions': BOUNDS, 'center_merge_radius': 28, 'tile_size': SIZE, 'original_gt_coordinates': 'tile center + crop offset; input image already in CryoPPP orientation', 'micrographs': info, 'test_accessed': True})

if __name__=='__main__':
    f=frozen();weight=Path(f['methods']['ANN-YOLO26']['checkpoint']);out=TEST_EXP/'ann_test_predictions'
    torch.set_num_threads(4)
    infer_ann_test(weight,out)
    info=json.loads((out/'inference.json').read_text())
    if [r['micrograph_name'] for r in info['micrographs']]!=TEST_NAMES:raise ValueError('ANN test coverage mismatch')
    with (TEST_MAN/'test_manifest.csv').open(newline='') as h:gold={r['micrograph_name']:r['mrc_sha256'] for r in csv.DictReader(h)}
    for r in info['micrographs']:
        r['source_sha256']=file_sha(DATA/'micrographs'/r['micrograph_name'])
        if r['source_sha256']!=gold[r['micrograph_name']]:raise ValueError('ANN source checksum changed')
    save_json(out/'inference.json',info)
    rows=prediction_rows(out/'predictions.csv','ANN-YOLO26')
    save_json(out/'done.json',{'status':'complete','method':'ANN-YOLO26','split':'test','images':100,'predictions':len(rows),'predictions_sha256':file_sha(out/'predictions.csv'),'inference_sha256':file_sha(out/'inference.json'),'frozen_config_sha256':file_sha(TEST_EXP/'frozen_config.json')})

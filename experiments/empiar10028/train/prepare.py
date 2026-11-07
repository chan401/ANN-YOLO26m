"""Prepare only the existing v3 Train/Validation float32 tiles, without original-data copies."""
import csv,json
from PIL import Image
from common import *

def main():
    EXP.mkdir(parents=True,exist_ok=True)
    rows=[];stats={};normalizations={}; examples=[]
    for split,expected in [('train',(132,11592)),('val',(44,3824))]:
        ns=names(split); total=0; labels_total=0; clipped=0
        for i,name in enumerate(ns):
            xy=points(split,name);total+=len(xy)
            image,meta=normalize_mrc(DATA/'micrographs'/name);normalizations[name]=meta
            covered=np.zeros(len(xy),dtype=int)
            for iy,y0 in enumerate(STARTS):
                for ix,x0 in enumerate(STARTS):
                    tid=f'{Path(name).stem}_x{x0}_y{y0}'
                    im=EXP/'images'/split/(tid+'.tif');lb=EXP/'labels'/split/(tid+'.txt')
                    im.parent.mkdir(parents=True,exist_ok=True);lb.parent.mkdir(parents=True,exist_ok=True)
                    tile=image[y0:y0+SIZE,x0:x0+SIZE]
                    if not im.exists(): Image.fromarray(tile).save(im,compression='tiff_deflate')
                    # Every visible center is labeled, including clipped boundary boxes.
                    # Inference retains only the unique central ownership region (>=128 px context).
                    lines=[]
                    for j,(x,y) in enumerate(xy):
                        xx,yy=x-x0,y-y0
                        if not (0<=xx<SIZE and 0<=yy<SIZE): continue
                        a,b,c,d=max(0,xx-BOX/2),max(0,yy-BOX/2),min(SIZE,xx+BOX/2),min(SIZE,yy+BOX/2)
                        full=xx>=BOX/2 and yy>=BOX/2 and xx<=SIZE-BOX/2 and yy<=SIZE-BOX/2
                        if full:covered[j]+=1
                        else:clipped+=1
                        lines.append(f'0 {(a+c)/2/SIZE:.10f} {(b+d)/2/SIZE:.10f} {(c-a)/SIZE:.10f} {(d-b)/SIZE:.10f}')
                        if full:
                            assert abs(((a+c)/2+x0)-x)<1e-8 and abs(((b+d)/2+y0)-y)<1e-8
                    lb.write_text('\n'.join(lines)+'\n' if lines else '')
                    labels_total+=len(lines)
                    rows.append(dict(split=split,micrograph_name=name,tile_id=tid,image_path=str(im),x0=x0,y0=y0,
                        core_x1=BOUNDS[ix],core_x2=BOUNDS[ix+1],core_y1=BOUNDS[iy],core_y2=BOUNDS[iy+1],labels=len(lines)))
                    if i==0 and ix==1 and iy==1:
                        restored=cv2.imread(str(im),cv2.IMREAD_UNCHANGED)
                        assert restored.dtype==np.float32 and np.array_equal(restored,tile),'TIFF lost float data'
                        examples.append({'file':str(im),'float32_exact_roundtrip':True,'unique_values':len(np.unique(restored))})
            if not np.all(covered>0):raise ValueError(f'{name}: GT lacking full-context tile')
            if (i+1)%20==0 or i+1==len(ns):print(split,i+1,len(ns),flush=True)
        assert (len(ns),total)==expected
        stats[split]={'micrographs':len(ns),'unique_gt':total,'tiles':len(ns)*9,'tile_labels_with_overlap':labels_total,'clipped_boundary_boxes':clipped,'all_gt_have_full_context_tile':True}
    with (EXP/'tiles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (EXP/'data.yaml').write_text(f'path: {EXP}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [particle]\n')
    # Small training/inference connectivity check; formal model restarts from official weights.
    for split in ['train','val']:
        chosen=[r for r in rows if r['split']==split][:8 if split=='train' else 4]
        (EXP/f'smoke_{split}.txt').write_text('\n'.join(r['image_path'] for r in chosen)+'\n')
    (EXP/'smoke.yaml').write_text(f'path: {EXP}\ntrain: smoke_train.txt\nval: smoke_val.txt\nnc: 1\nnames: [particle]\n')
    write_json(EXP/'adaptation_inputs.json',{'status':'PASS','tile_size':SIZE,'tile_starts':STARTS,'overlap_px':256,'ownership_boundaries':BOUNDS,
        'box_original_px':BOX,'box_source':'existing confirmed CryoPPP 224px training proxy; not tuned on validation',
        'normalization':'median, sigma=(P84-P16)/2, clip +/-5 sigma, float32 TIFF lossless',
        'orientation':'image flipud to CryoPPP axes before tiling; canonical center=(tile_x+x0,tile_y+y0)',
        'splits':stats,'normalization_parameters':normalizations,'examples':examples,'test_accessed':False,
        'manifest_hashes':{s:sha(SPLIT/f'{s}.txt') for s in ['train','val']}})
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()

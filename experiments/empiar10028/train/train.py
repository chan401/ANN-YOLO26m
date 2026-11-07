import argparse,json,os
from common import *
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--weights',default=str(WEIGHT));p.add_argument('--name',default='standard_m_p3p5');p.add_argument('--epochs',type=int,default=80);p.add_argument('--lr',type=float,default=.002);p.add_argument('--batch',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('Formal ANN training requires available GPU')
    name='smoke' if a.smoke and a.name=='standard_m_p3p5' else a.name
    out=EXP/'runs'/name
    if out.exists():raise FileExistsError(out)
    # Ultralytics' AMP self-check resolves yolo26n.pt relative to cwd; reuse its existing local file.
    os.chdir(ULTRA)
    m=YOLO(a.weights)
    head=m.model.model[-1]
    assert head.stride.tolist()==[8.,16.,32.]
    assert getattr(m.model,'end2end',False)
    cfg=dict(data=str(EXP/('smoke.yaml' if a.smoke else 'data.yaml')),imgsz=SIZE,
        epochs=1 if a.smoke else a.epochs,batch=a.batch,device=0,workers=2 if a.smoke else 4,
        project=str(EXP/'runs'),name=name,exist_ok=False,pretrained=True,optimizer='AdamW',lr0=a.lr,lrf=.05,
        weight_decay=.0005,warmup_epochs=.1 if a.smoke else 3,warmup_bias_lr=.01,cos_lr=True,nbs=16,
        patience=20,amp=True,cache=False,rect=False,multi_scale=0.0,max_det=2000,conf=.0001,
        seed=10028,deterministic=True,save=True,save_period=10,plots=False,
        mosaic=0.,close_mosaic=0,mixup=0.,cutmix=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
        degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,fliplr=0.,flipud=0.,val=True)
    write_json(EXP/(name+'_config.json'),{'training':cfg,'initial_weights':a.weights,'initial_weights_sha256':sha(a.weights),
        'head_strides':head.stride.tolist(),'head_max_det':2000,'source_code_sha256':{q.name:sha(q) for q in Path(__file__).parent.glob('*.py')},
        'actual_augmentations':'custom quarter-turns, H/V flips, gain [.9,1.1], bias +/-.02, Gaussian noise std<=.01 with p=.5; same noise across channels',
        'baseline':'standard P3/P4/P5 with official weights, no architectural change'})
    result=m.train(trainer=CryoTrainer,**cfg)
    out=Path(result.save_dir)
    assert (out/'weights/best.pt').exists() and (out/'weights/last.pt').exists()
    rows=list(csv.DictReader((out/'results.csv').open()))
    for row in rows:
        for k,v in row.items():
            if 'loss' in k and not np.isfinite(float(v)):raise RuntimeError('Nonfinite loss')
    write_json(out/'completion.json',{'status':'PASS','epochs_completed':len(rows),'best':str(out/'weights/best.pt'),'last':str(out/'weights/last.pt'),
        'best_sha256':sha(out/'weights/best.pt'),'max_cuda_memory_gb':torch.cuda.max_memory_allocated()/1e9,'smoke':a.smoke})
    print('TRAINING_COMPLETED',str(out),flush=True)

if __name__=='__main__':main()

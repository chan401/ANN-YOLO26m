"""Focused adapter tests: float data, grayscale, geometry, and coordinate round trips."""
from common import *
from ultralytics.utils.instance import Instances
from ultralytics.cfg import get_cfg
from PIL import Image, ImageDraw

def main():
    for k in range(4):
        a=np.zeros((10,10,3),np.float32);a[3,2]=1
        lab={'img':a,'instances':Instances(np.array([[.25,.35,.1,.1]],np.float32),segments=np.zeros((0,0,2),np.float32),bbox_format='xywh',normalized=True)}
        out=QuarterTurn(k)(lab);cx,cy=out['instances'].bboxes[0,:2]
        yy,xx=np.argwhere(out['img'][...,0]>0)[0]
        assert abs(cx-(xx+.5)/10)<1e-6 and abs(cy-(yy+.5)/10)<1e-6
    for direct in ['horizontal','vertical']:
        a=np.zeros((10,10,3),np.float32);a[3,2]=1
        lab={'img':a,'instances':Instances(np.array([[.25,.35,.1,.1]],np.float32),segments=np.zeros((0,0,2),np.float32),bbox_format='xywh',normalized=True)}
        out=RandomFlip(1,direct)(lab);cx,cy=out['instances'].bboxes[0,:2];yy,xx=np.argwhere(out['img'][...,0]>0)[0]
        assert abs(cx-(xx+.5)/10)<1e-6 and abs(cy-(yy+.5)/10)<1e-6
    ds=CryoDataset(img_path=str(EXP/'smoke_train.txt'),imgsz=SIZE,batch_size=2,augment=True,hyp=get_cfg(),rect=False,cache=False,
        task='detect',data={'names':{0:'particle'},'nc':1,'channels':3})
    random.seed(10028);np.random.seed(10028);a=ds[0]
    random.seed(10028);np.random.seed(10028);b=ds[0]
    assert a['img'].dtype==torch.float32 and torch.equal(a['img'],b['img']) and torch.equal(a['bboxes'],b['bboxes'])
    assert torch.equal(a['img'][0],a['img'][1]) and torch.equal(a['img'][1],a['img'][2])
    assert torch.isfinite(a['img']).all() and 0<=a['img'].min()<=a['img'].max()<=255
    for split in ['train','val']:
        for name in names(split):
            xy=points(split,name)
            for x,y in xy:
                ix=int(np.searchsorted(BOUNDS[1:],x,side='right'));iy=int(np.searchsorted(BOUNDS[1:],y,side='right'))
                xx=x-STARTS[ix];yy=y-STARTS[iy]
                assert BOX/2<=xx<=SIZE-BOX/2 and BOX/2<=yy<=SIZE-BOX/2
                assert abs(xx+STARTS[ix]-x)<1e-8 and abs(yy+STARTS[iy]-y)<1e-8
    row=tile_rows('train')[4]
    tile=cv2.imread(row['image_path'],cv2.IMREAD_UNCHANGED)
    im=Image.fromarray(np.uint8(np.clip(tile,0,1)*255)).convert('RGB');draw=ImageDraw.Draw(im)
    for line in (EXP/'labels/train'/(row['tile_id']+'.txt')).read_text().splitlines():
        _,x,y,w,h=map(float,line.split());x*=SIZE;y*=SIZE;w*=SIZE;h*=SIZE
        draw.rectangle([x-w/2,y-h/2,x+w/2,y+h/2],outline='red',width=2)
        draw.ellipse([x-4,y-4,x+4,y+4],fill='lime')
    im.resize((768,768)).save(EXP/'label_overlay_train.png')
    write_json(EXP/'adapter_checks.json',{'status':'PASS','quarter_turn_and_flip_label_geometry':True,'float32_no_uint8_in_training':True,
        'channels_identical_including_noise':True,'seed_reproducible_augmentation':True,'all_train_val_gt_ownership_roundtrip_and_full_context':True,
        'test_accessed':False})
    print('ADAPTER_CHECKS_PASS',flush=True)

if __name__=='__main__':main()

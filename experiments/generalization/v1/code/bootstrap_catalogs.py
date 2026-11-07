"""Download official annotation archives/metadata only; no micrograph or model selection."""
from pathlib import Path
import concurrent.futures,csv,hashlib,json,tarfile,time,urllib.request,os
G=Path('/root/autodl-tmp/cryoem_generalization/v1')
IDS=['10017','10077','10291','10075','10081','10059']
def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'cryo-generalization-research/1.0'}),timeout=45) as r:return r.read()
        except Exception:
            if i==3:raise
            time.sleep(2*(i+1))
def main():
    G.mkdir(parents=True,exist_ok=True);(G/'catalogs').mkdir(exist_ok=True)
    api=json.loads(get('https://zenodo.org/api/records/7934683'));(G/'catalogs/zenodo_7934683.json').write_text(json.dumps(api,indent=2))
    files={r['key']:r for r in api['files']}
    def one(ds):
        root=G/'datasets'/ds;root.mkdir(parents=True,exist_ok=True);ann=root/'annotations_original';ann.mkdir(exist_ok=True)
        entry=files[ds+'.tar.gz'];archive=G/'catalogs'/(ds+'.tar.gz')
        if not archive.exists():
            tmp=archive.with_suffix('.partial');tmp.write_bytes(get(entry['links']['self']));os.replace(tmp,archive)
        data=archive.read_bytes();assert len(data)==entry['size'];assert hashlib.md5(data).hexdigest()==entry['checksum'].split(':',1)[1]
        exported=[]
        with tarfile.open(archive,'r:gz') as t:
            for member in t:
                name=member.name
                if member.isfile() and '/ground_truth/particle_coordinates/' in '/'+name and name.lower().endswith('.csv'):
                    target=ann/Path(name).name;raw=t.extractfile(member).read()
                    if target.exists():assert target.read_bytes()==raw
                    else:target.write_bytes(raw)
                    exported.append({'csv_name':target.name,'archive_member':name,'sha256':hashlib.sha256(raw).hexdigest()})
        assert len(exported)>=30,(ds,len(exported))
        meta=json.loads(get(f'https://www.ebi.ac.uk/empiar/api/entry/{ds}/'));(root/'empiar_entry.json').write_text(json.dumps(meta,indent=2))
        provenance={'dataset':ds,'annotation_source':entry['links']['self'],'zenodo_md5':entry['checksum'],'archive_sha256':hashlib.sha256(data).hexdigest(),'annotation_files':exported}
        (root/'annotation_catalog.json').write_text(json.dumps(provenance,indent=2))
        example=ann/exported[0]['csv_name'];entrymeta=meta['EMPIAR-'+ds]
        print('CATALOG_READY',ds,'annotations',len(exported),'title',entrymeta['title'],'example',example.name,'HEADER',example.read_text().splitlines()[:2],flush=True)
        return {'dataset':ds,'annotation_count':len(exported),'title':entrymeta['title'],'imagesets':[{k:r.get(k) for k in ['directory','category','image_width','image_height','pixel_width']} for r in entrymeta['imagesets']]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(one,IDS))
    (G/'catalogs/datasets_summary.json').write_text(json.dumps(result,indent=2));print('ALL_CATALOGS_READY',flush=True)
if __name__=='__main__':main()

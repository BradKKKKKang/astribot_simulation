"""Validate and encode cloth-study videos and synchronized comparison panels."""
import argparse,json,subprocess
from pathlib import Path
import cv2
import numpy as np


def h264(path):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name','-of','json',str(path)]))
    if probe['streams'][0]['codec_name']=='h264': return
    temp=path.with_suffix('.encoded.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(temp)],check=True,timeout=120)
    temp.replace(path)


def validate(directory):
    path=directory/'interaction.mp4'
    validation=directory/'video.validation.json'
    if validation.exists() and validation.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return
    h264(path)
    capture=cv2.VideoCapture(str(path));count=0;selected={}
    while True:
        ok,frame=capture.read()
        if not ok: break
        if count in (170,226): selected[count]=frame
        count+=1
    fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
    if count!=247 or abs(fps-20)>1e-6: raise RuntimeError(f'invalid video timing {path}: {count}, {fps}')
    diffs=[float(np.abs(selected[170][64:,x:x+640].astype(float)-selected[226][64:,x:x+640]).mean()) for x in (0,640)]
    views=float(np.abs(selected[170][64:,:640].astype(float)-selected[170][64:,640:]).mean())
    if min(diffs)<.05 or views<5: raise RuntimeError(f'frozen or duplicate camera views {path}: {diffs}, {views}')
    records=[json.loads(l) for l in (directory/'video.frames.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==list(range(0,1231,5))
    (directory/'video.validation.json').write_text(json.dumps({'codec':'h264','decoded_frames':count,'fps':fps,'duration_s':count/fps,'two_panel_motion_difference':diffs,'inter_view_difference':views,'time_mapping':'video.frames.jsonl'},indent=2))


def comparison(out,name,cases):
    destination=out/(name+'.mp4')
    if destination.exists() and destination.stat().st_mtime_ns >= max((out/c/'interaction.mp4').stat().st_mtime_ns for c in cases):
        return
    captures=[cv2.VideoCapture(str(out/c/'interaction.mp4')) for c in cases]
    for cap in captures: cap.set(cv2.CAP_PROP_POS_FRAMES,150)
    path=out/(name+'.mp4')
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,816))
    if not writer.isOpened(): raise RuntimeError('encoder unavailable')
    try:
        for _ in range(97):
            canvas=np.zeros((816,1280,3),np.uint8)
            for i,cap in enumerate(captures):
                ok,frame=cap.read()
                if not ok: raise RuntimeError('source ended early')
                x,y=(i%2)*640,(i//2)*272
                canvas[y:y+272,x:x+640]=cv2.resize(frame,(640,272))
            writer.write(canvas)
    finally:
        writer.release()
        for cap in captures: cap.release()
    h264(path)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,width,height,codec_name','-of','json',str(path)]))['streams'][0]
    assert int(probe['nb_frames'])==97 and probe['width']==1280 and probe['height']==816 and probe['codec_name']=='h264'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--partial',action='store_true');args=p.parse_args()
    cases=json.loads((args.output/'manifest.json').read_text())['conditions']
    for case in cases:
        directory=args.output/case['name'];status=directory/'run_status.json'
        if args.partial and (not status.exists() or json.loads(status.read_text())['status']!='completed'):continue
        validate(directory);print('VIDEO_OK',case['name'],flush=True)
    if not args.partial:
        comparison(args.output,'closure_comparison',['closure_90','closure_93','baseline','closure_98','closure_100'])
        comparison(args.output,'position_comparison',['baseline','position_x_minus10mm','position_x_plus10mm','position_y_minus10mm','position_y_plus10mm'])

if __name__=='__main__':main()

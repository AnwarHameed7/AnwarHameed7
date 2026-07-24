import streamlit as st, cv2, numpy as np, tempfile, os, math, pandas as pd
import requests, re, json, time
from ultralytics import YOLO
from datetime import datetime
from collections import deque
from analysis.pose import PoseAnalyzer
from analysis.ball_detector import BallDetector
from analysis.sports_science import SportsScienceEngine
from analysis.football_iq import FootballIQAnalyzer
from analysis.heatmap import HeatmapAnalyzer
from analysis.running import RunningAnalyzer
from analysis.report import ReportGenerator
from tracking.player_memory import PlayerMemory
from utils.visualizer import Visualizer

st.set_page_config(layout='wide')
for k in ['v','p','fb','sk','rpt']:
  if k not in st.session_state: st.session_state[k]=[] if k=='fb' else (None if k!='rpt' else [])
if st.session_state.fb is None: st.session_state.fb=[]

CSS = '''
.c{background:#0a0a0a;padding:1rem;border-radius:8px;border-left:4px solid #0f0;margin:0.25rem 0}
.s{background:#111;padding:0.5rem;border-radius:6px;border:1px solid #333;text-align:center}
.w{background:#1a0a0a;padding:0.5rem;border-left:3px solid #f44;margin:0.25rem 0}
.st{background:#0a1a0a;padding:0.5rem;border-left:3px solid #4f4;margin:0.25rem 0}
.sub{background:#111;padding:0.5rem;border-radius:6px;border:1px solid #444;max-height:150px;overflow-y:auto}
.sc{border-radius:8px;margin:0.25rem 0}
.mt{background:#0a0a1a;padding:0.5rem;border-radius:4px;margin:0.15rem}
'''
st.markdown(f'<style>{CSS}</style>',unsafe_allow_html=True)
st.title('FootballAI Pro')
st.caption('Professional AI Football Analysis Platform - Elite Player Edition')

def build_pipeline():
  p=type('P',(),{'__init__':lambda s:None})()
  p.ra=RunningAnalyzer();p.fiq=FootballIQAnalyzer()
  p.hm=HeatmapAnalyzer();p.pa=PoseAnalyzer()
  p.sse=SportsScienceEngine();p.bd=BallDetector()
  p.pm=PlayerMemory();p.vz=Visualizer()
  p.sel=None;p.im={};p.nid=1
  p.tr=YOLO('weights/yolo11n.pt') if os.path.exists('weights/yolo11n.pt') else YOLO('yolo11n.pt')
  p.spd_hist={};p.speed_buf={}
  return p

def get_video(inp,url,uf):
  vp=None
  if inp.startswith('URL') and url and st.button('Download & Analyze'):
    with st.spinner('Downloading...'):
      t=tempfile.NamedTemporaryFile(delete=False)
      m=re.search(r'drive[.]google[.]com/file/d/([^/]+)',url)
      if m:
        fid=m.group(1);g='https://drive.google.com/uc?export=download&id='+fid
        s=requests.Session();r=s.get(g,stream=True)
        for kv in r.cookies.items():
          if kv[0].startswith('download_warning'):g+='&confirm='+kv[1];r=s.get(g,stream=True);break
      else:
        r=requests.get(url,stream=True)
      r.raise_for_status()
      for c in r.iter_content(8192):
        if c:t.write(c)
      t.close();vp=t.name
  elif uf is not None:
    t=tempfile.NamedTemporaryFile(delete=False);t.write(uf.read());vp=t.name
  return vp

POSE_CONNECTIONS=[(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def process_frame(pr,fr,sel,sk_frames,frame_idx,detected_set):
  if frame_idx%(sk_frames+1)!=0: return fr,None,None
  r=pr.tr.track(fr,persist=True,tracker='botsort.yaml',conf=0.3,iou=0.45,classes=[0],imgsz=1280,verbose=False)[0]
  ball_pos=pr.bd.detect(fr)
  findings=[]
  if r.boxes is not None:
    for b in r.boxes:
      if b.id is None: continue
      iid=int(b.id[0]);tid=pr.im.get(iid)
      if tid is None: pr.im[iid]=pr.nid;tid=pr.nid;pr.nid+=1
      bb=[int(x) for x in b.xyxy[0]];cx,cy=(bb[0]+bb[2])//2,(bb[1]+bb[3])//2
      pr.pm.create_or_update(tid,bb,fr)
      if tid==sel:
        pr.ra.update(sel,(cx,cy));pr.fiq.update(sel,(cx,cy));pr.hm.update(cx,cy)
        touch=pr.bd.check_touch(bb,ball_pos)
        if touch and 'ball_touch' not in detected_set: detected_set.add('ball_touch')
        _,pdata=pr.pa.analyze(fr,bb)
        if pdata and 'keypoints' in pdata:
          pr.vz.draw_pose(fr,pdata['keypoints'],POSE_CONNECTIONS)
        if pdata:
          f=pr.sse.analyze(pdata,pr.ra.get_speed(sel),pr.fiq.total_distance(sel))
          for x in f:
            dt=x.get('detected','')
            if dt and dt not in detected_set:
              detected_set.add(dt);findings.append(x)
          if tid not in pr.spd_hist: pr.spd_hist[tid]=[]
          pr.spd_hist[tid].append(pr.ra.get_speed(sel))
  return fr,findings,ball_pos

inp=st.radio('Input',['URL (up to 4GB)','Upload (up to 200MB)'],horizontal=True)
vp=None;url=None;uf=None
if inp.startswith('URL'):
  url=st.text_input('Paste video URL (Google Drive, Dropbox, etc.)')
else:
  uf=st.file_uploader('Upload video',['mp4','avi','mov','mkv'])
vp=get_video(inp,url,uf)

if vp is not None:
  pr=st.session_state.p
  if pr is None: pr=build_pipeline();st.session_state.p=pr
  cap=cv2.VideoCapture(vp);ret,fr=cap.read();cap.release()
  if not ret: st.error('Could not read video')
  else:
    pr.hm.initialize(fr)
    r=pr.tr.track(fr,persist=True,tracker='botsort.yaml',conf=0.3,iou=0.45,classes=[0],imgsz=1280,verbose=False)[0]
    if r.boxes is not None:
      for b in r.boxes:
        if b.id is None: continue
        iid=int(b.id[0])
        if iid not in pr.im: pr.im[iid]=pr.nid;pr.nid+=1
    pl=list(pr.im.values())
    if not pl: st.error('No players detected. Try a different video.')
    else:
      sel=st.selectbox('Select Player to Analyze',pl)
      pr.sel=sel
      if st.button('Start Professional Analysis'):
        cap=cv2.VideoCapture(vp)
        fps=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ot=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name
        out=cv2.VideoWriter(ot,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        frames=0;last_fb='';all_findings=[];detected_set=set()
        sk=st.slider('Frame skip (faster processing)',0,5,2)
        bar=st.progress(0)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with st.spinner('Analyzing frames...'):
          while True:
            ret,fr=cap.read()
            if not ret: break
            ofr=fr.copy()
            _,findings,ball_pos=process_frame(pr,ofr,sel,sk,frames,detected_set)
            if findings: all_findings.extend(findings);last_fb=findings[0].get('problem','')+' - '+findings[0].get('corrective_drills','')
            # Draw overlays
            ofr=pr.hm.overlay(ofr)
            path=pr.ra.get_path(sel)
            if len(path)>1:
              pts=[(int(p[0]),int(p[1])) for p in path]
              pr.vz.draw_trail(ofr,pts)
            # Draw ball if detected
            if ball_pos:
              cv2.circle(ofr,ball_pos,8,(0,0,255),-1)
              cv2.circle(ofr,ball_pos,12,(0,0,255),2)
            # Overlay subtitle
            if last_fb:
              cv2.rectangle(ofr,(0,h-60),(w,h-5),(0,0,0),-1)
              cv2.putText(ofr,last_fb[:80],(20,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            out.write(ofr)
            frames+=1;bar.progress(min(frames/total,1.0))
        cap.release();out.release();bar.empty()
        st.success(f'Done! {frames} frames processed.')

        fiq_st=pr.fiq.stats(sel);ra_st=pr.ra.get_stats(sel)
        assess=pr.sse.get_overall_assessment(fiq_st,all_findings)
        rpt=ReportGenerator()
        rpt.generate_player_report(sel,ra_st,fiq_st)
        st.session_state.rpt=rpt
        col1,col2=st.columns([2,1])
        with col1:
          with open(ot,'rb') as f:st.video(f.read())
        with col2:
          st.markdown('<div class="c">',unsafe_allow_html=True)
          st.subheader('Player Dashboard')
          sc=assess
          st.markdown(f"**Overall Score:** {sc['overall_score']}/100")
          st.progress(sc['overall_score']/100)
          c1,c2=st.columns(2)
          with c1:
            st.markdown(f"<div class='s'>Tech: {sc['technical_score']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Physical: {sc['physical_score']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Tactical: {sc['tactical_score']}</div>",unsafe_allow_html=True)
          with c2:
            st.markdown(f"<div class='s'>Biomech: {sc['biomechanics_score']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Fatigue: {sc['fatigue_estimate']}%</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Risk: {sc['risk_level']}</div>",unsafe_allow_html=True)
          st.markdown('</div>',unsafe_allow_html=True)
          st.markdown('<div class="sub">',unsafe_allow_html=True)
          st.subheader('Live Coach Commentary')
          st.info(last_fb or 'No issues detected - good mechanics')
          st.markdown('</div>',unsafe_allow_html=True)
          c1,c2,c3=st.columns(3)
          c1.metric('Speed',f"{ra_st['speed']:.1f}")
          c2.metric('Distance',f"{ra_st['distance']:.0f}")
          c3.metric('Sprints',fiq_st['sprints'])
          spd_hist=pr.spd_hist.get(sel,[])
          if len(spd_hist)>1:
            sg=pr.vz.create_speed_graph(spd_hist)
            st.image(sg,caption='Speed Graph',use_container_width=True)
            db=pr.vz.create_distance_bar(ra_st['distance'])
            st.image(db,caption='Distance Progress',use_container_width=True)
          if sc['weaknesses']:
            st.markdown('**Weaknesses**')
            for w in sc['weaknesses']:
              st.markdown(f"<div class='w'>{w}</div>",unsafe_allow_html=True)
          if sc['strengths']:
            st.markdown('**Strengths**')
            for s in sc['strengths']:
              st.markdown(f"<div class='st'>{s}</div>",unsafe_allow_html=True)

        # Export section
        st.markdown('---')
        st.subheader('Export Report')
        ec1,ec2,ec3,ec4=st.columns(4)
        rpt=st.session_state.rpt
        if ec1.button('CSV Report'):
          fp=os.path.join(tempfile.gettempdir(),f'player_{sel}_report.csv')
          rpt.export_to_csv(fp)
          with open(fp,'rb') as f:st.download_button('Download CSV',f,'report.csv','text/csv')
        if ec2.button('JSON Report'):
          jdata=json.dumps({str(sel):rpt.report_data.get(sel,{})},indent=2)
          st.download_button('Download JSON',jdata,'report.json','application/json')
        if ec3.button('PDF Report'):
          fp=os.path.join(tempfile.gettempdir(),f'player_{sel}_report.pdf')
          rpt.export_to_pdf(fp,sc)
          with open(fp,'rb') as f:st.download_button('Download PDF',f,'report.pdf','application/pdf')
        if ec4.button('Training Plan'):
          tp=pr.sse.generate_training_plan(all_findings)
          st.download_button('Download Plan',tp['plan'],'training_plan.txt','text/plain')
elif url is None:
  st.write('Provide a video to begin analysis.')

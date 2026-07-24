import streamlit as st, cv2, numpy as np, tempfile, os, math, pandas as pd
import sys as _sys, requests, re, json, time
from ultralytics import YOLO
from datetime import datetime
from collections import deque

# ========== INLINED MODULES ==========

class RunningAnalyzer:
  def __init__(s):s.pos={};s.td={};s.cs={};s.ms={}
  def update(s,t,c):
    if t not in s.pos:s.pos[t]=[];s.td[t]=0;s.cs[t]=0;s.ms[t]=0
    l=s.pos[t][-1] if s.pos[t] else c;d=math.hypot(c[0]-l[0],c[1]-l[1])
    s.td[t]+=d;s.cs[t]=d;s.pos[t].append(c)
    if d>s.ms[t]:s.ms[t]=d
    if len(s.pos[t])>300:s.pos[t].pop(0)
  def get_speed(s,t):return s.cs.get(t,0)
  def get_distance(s,t):return s.td.get(t,0)
  def get_max_speed(s,t):return s.ms.get(t,0)
  def get_path(s,t):return s.pos.get(t,[])
  def get_stats(s,t):return {'speed':s.get_speed(t),'distance':s.get_distance(t),'max_speed':s.get_max_speed(t)}

class FootballIQAnalyzer:
  def __init__(s):s.ph={}
  def update(s,t,c):
    if t not in s.ph:s.ph[t]=[]
    s.ph[t].append(c)
    if len(s.ph[t])>300:s.ph[t].pop(0)
  def td(s,t):
    p=s.ph.get(t,[]);return sum(math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]) for i in range(1,len(p))) if len(p)>1 else 0
  def asp(s,t):p=s.ph.get(t,[]);return s.td(t)/len(p) if len(p)>1 else 0
  def sc(s,t,th=15):
    h=s.ph.get(t,[]);return sum(1 for i in range(1,len(h)) if math.hypot(h[i][0]-h[i-1][0],h[i][1]-h[i-1][1])>th)
  def dc(s,t):
    p=s.ph.get(t,[]);c=0
    for i in range(2,len(p)):
      d1=p[i-1][0]-p[i-2][0],p[i-1][1]-p[i-2][1];d2=p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]
      m1,m2=math.hypot(*d1),math.hypot(*d2)
      if m1==0 or m2==0:continue
      a=math.degrees(math.acos(max(-1,min(1,(d1[0]*d2[0]+d1[1]*d2[1])/(m1*m2)))))
      if a>45:c+=1
    return c
  def wr(s,t):d=s.td(t);return 'High' if d>1500 else ('Medium' if d>500 else 'Low')
  def ms(s,t):d=s.td(t);t=s.dc(t);sp=s.sc(t);return min(100,round(d*0.02+t*2+sp*5))
  def stats(s,t):return {'distance':s.td(t),'avg_speed':s.asp(t),'sprints':s.sc(t),'direction_changes':s.dc(t),'work_rate':s.wr(t),'movement_score':s.ms(t)}

class HeatmapAnalyzer:
  def __init__(s):s.c=None
  def initialize(s,f):
    if s.c is not None:return
    h,w=f.shape[:2];s.c=np.zeros((h,w),np.float32)
  def update(s,x,y):
    if s.c is not None:cv2.circle(s.c,(int(x),int(y)),20,1,-1)
  def overlay(s,f,a=0.45):
    if s.c is None:return f
    n=cv2.normalize(s.c,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    h=cv2.applyColorMap(n,cv2.COLORMAP_JET)
    return cv2.addWeighted(f,1-a,h,a,0)

class PoseAnalyzer:
  def __init__(s,mp='weights/yolo11n-pose.pt'):s.m=YOLO(mp)
  def analyze(s,f,b):
    x1,y1,x2,y2=b;h,w=f.shape[:2]
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
    r=f[y1:y2,x1:x2]
    if r.size==0:return f,None
    rs=s.m.predict(r,verbose=False)
    if not rs or not rs[0].keypoints or len(rs[0].keypoints.xy[0])<17:return f,None
    kp=rs[0].keypoints.xy[0].cpu().numpy()
    def ang(a,b,c):
      a=math.degrees(math.atan2(c[1]-b[1],c[0]-b[0])-math.atan2(a[1]-b[1],a[0]-b[0]))
      return abs(a) if abs(a)<=180 else 360-abs(a)
    lk=ang(kp[11],kp[13],kp[15]);rk=ang(kp[12],kp[14],kp[16])
    lh=ang(kp[5],kp[11],kp[13]);rh=ang(kp[6],kp[12],kp[14])
    kp_abs=kp.copy();kp_abs[:,0]+=x1;kp_abs[:,1]+=y1
    return f,{'lk':lk,'rk':rk,'lh':lh,'rh':rh,'po':'upright' if (lh+rh)/2>120 else 'leaning','kp':kp_abs}

POSE_CONNS=[(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

class BallDetector:
  def __init__(s):s.lp=None
  def detect(s,f):
    h=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(h,np.array([0,0,200]),np.array([180,30,255]))
    m=cv2.erode(m,None,iterations=2);m=cv2.dilate(m,None,iterations=2)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cs:
      c=max(cs,key=cv2.contourArea)
      if cv2.contourArea(c)>50:
        (x,y),r=cv2.minEnclosingCircle(c)
        if 5<r<30:s.lp=(int(x),int(y));return s.lp
    return None
  def check_touch(s,bx,bp):
    if bp is None:return False
    x1,y1,x2,y2=bx;cx,cy=bp
    return x1<=cx<=x2 and y1<=cy<=y2

class SSE:
  def __init__(s):
    s.db={
      'stiff':{'p':'Stiff running gait','rc':'Weak hip flexors','b':'Reduced knee flexion increases ground contact 12%','d':'A-Skips 3x20, High Knee 3x15, Wall Drills 3x12','st':'Hip flexor raises 3x15, RDL 3x8','f':'3x/week 45min','rec':'Foam roll 10min','sev':'moderate'},
      'lean':{'p':'Excessive forward lean','rc':'Weak core','b':'Anterior pelvic tilt shifts COM forward','d':'Dead Bug 3x10, Pallof Press 3x8, SL RDL 3x8','st':'Planks 3x60s, Bird Dogs 3x10','f':'4x/week 30min','rec':'Massage lower back 5min','sev':'moderate'},
      'low':{'p':'Drop in intensity','rc':'Low aerobic capacity','b':'Glycogen depletion in fast-twitch fibers','d':'HIIT 30:30x8, Tempo 4minx4, Fartlek 15min','st':'Box Jumps 3x8, KB Swings 3x15','f':'3x/week 40min','rec':'48h recovery between HIIT','sev':'high'},
    }
  def analyze(s,pd,sp,di):
    f=[]
    if pd is None:return f
    ak=(pd.get('lk',0)+pd.get('rk',0))/2
    if ak>160 and sp>5:f.append({**s.db['stiff'],'det':'gait'})
    if pd.get('po')=='leaning' and sp<3:f.append({**s.db['lean'],'det':'posture'})
    if sp<2 and di>100:f.append({**s.db['low'],'det':'intensity'})
    return f
  def gen_plan(s,fs):
    if not fs:return 'No weaknesses detected. Maintain current training.'
    p='Weekly Plan:\n'
    for x in fs:p+=f"\n{x['p']}\n  Corrective: {x['d']}\n  Strength: {x['st']}\n  Frequency: {x['f']}\n  Recovery: {x['rec']}\n"
    return p
  def assess(s,st,fs):
    sc=min(100,st.get('movement_score',50)+st.get('sprints',0)*2)
    if st.get('work_rate')=='High':sc+=10
    if st.get('work_rate')=='Low':sc-=10
    sc=max(0,min(100,sc));r='Low'
    for x in fs:
      if x.get('sev')=='high':r='High'
      elif x.get('sev')=='moderate' and r!='High':r='Moderate'
    return {'os':sc,'ts':min(100,sc+5),'ps':max(0,sc-5),'tcs':sc,'bs':max(0,sc-10) if fs else min(100,sc+10),'risk':r,'fat':min(100,max(0,100-sc)),'weak':[x['p'] for x in fs],'strong':['Good sprint capacity' if st.get('sprints',0)>10 else 'Steady movement']}

class PlayerMemory:
  def __init__(s):s.pl={};s.ch={}
  def create_or_update(s,t,bx,f):
    x1,y1,x2,y2=bx
    if t not in s.pl:
      s.pl[t]={'first_seen':0,'last_seen':0,'total_frames':0,'name':f'Player {t}'}
      roi=f[y1:y2,x1:x2]
      if roi.size>0:
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[50,60],[0,180,0,256]);cv2.normalize(hist,hist)
        s.ch[t]=hist

class Visualizer:
  def draw_trail(s,f,pts,color=(0,255,255),tl=30):
    if len(pts)<2:return f
    for i in range(1,len(pts[-tl:])):
      t=max(1,int(8*i/len(pts[-tl:])));cv2.line(f,pts[-tl:][i-1],pts[-tl:][i],color,t)
    return f
  def draw_pose(s,f,kp,conns=None):
    if kp is None:return f
    for pt in kp:
      x,y=int(pt[0]),int(pt[1])
      if x>0 and y>0:cv2.circle(f,(x,y),3,(0,255,0),-1)
    if conns:
      for a,b in conns:
        if a<len(kp) and b<len(kp):
          x1,y1=int(kp[a][0]),int(kp[a][1]);x2,y2=int(kp[b][0]),int(kp[b][1])
          if x1>0 and y1>0 and x2>0 and y2>0:cv2.line(f,(x1,y1),(x2,y2),(0,255,0),2)
    return f
  def create_speed_graph(s,sh,w=300,h=150):
    g=np.zeros((h,w,3),np.uint8)
    if len(sh)<2:return g
    mx=max(sh) if max(sh)>0 else 1
    pts=[(i*w//min(len(sh),w),h-int(v*h/mx)) for i,v in enumerate(sh[-w:])]
    for i in range(1,len(pts)):cv2.line(g,pts[i-1],pts[i],(0,255,0),2)
    return g
  def create_distance_bar(s,d,mxd=1000):
    b=np.zeros((30,200,3),np.uint8);fl=min(200,int(d*200/mxd))
    b[:,:fl]=(0,255,0);b[:,fl:]=(50,50,50);return b

class ReportGen:
  def __init__(s):s.rd={}
  def gen(s,pid,rs,iqs):
    s.rd[pid]={'pid':pid,'ts':datetime.now().isoformat(),'m':{'dist':rs.get('distance',0),'ms':rs.get('max_speed',0),'as':iqs.get('avg_speed',0),'sprints':iqs.get('sprints',0),'wr':iqs.get('work_rate',''),'mov':iqs.get('movement_score',0),'dc':iqs.get('direction_changes',0)}}
  def to_csv(s,fp):
    if not s.rd:return False
    import pandas
    df=pandas.DataFrame([{'pid':k,**v['m']} for k,v in s.rd.items()]);df.to_csv(fp,index=False);return True
  def to_json(s,pid):return json.dumps({str(pid):s.rd.get(pid,{})},indent=2)
  def to_pdf(s,fp,assess=None):
    try:
      from fpdf import FPDF
      p=FPDF();p.add_page();p.set_font('Arial','B',16);p.cell(0,10,'FootballAI Pro Report',ln=True,align='C');p.ln(10)
      for pid,d in s.rd.items():
        p.set_font('Arial','B',12);p.cell(0,8,f'Player: {pid}',ln=True);p.set_font('Arial','',10);p.cell(0,6,f'Time: {d["ts"]}',ln=True);p.ln(4)
        for k,v in d['m'].items():p.cell(0,5,f'  {k}: {v}',ln=True)
      if assess:
        p.ln(8);p.set_font('Arial','B',12);p.cell(0,8,'Assessment',ln=True);p.set_font('Arial','',10)
        for k,v in assess.items():
          if isinstance(v,list):p.cell(0,5,f'  {k}: {", ".join(str(x) for x in v)}',ln=True)
          else:p.cell(0,5,f'  {k}: {v}',ln=True)
      p.output(fp);return True
    except:return False

# ========== APP ==========

st.set_page_config(layout='wide')
for kv in ['v','p','fb','sk','rpt']:
  if kv not in st.session_state:st.session_state[kv]=[] if kv=='fb' else(None if kv!='rpt' else[])
if st.session_state.fb is None:st.session_state.fb=[]
CSS='''.c{background:#0a0a0a;padding:1rem;border-radius:8px;border-left:4px solid #0f0;margin:0.25rem 0}
.s{background:#111;padding:0.5rem;border-radius:6px;border:1px solid #333;text-align:center}
.w{background:#1a0a0a;padding:0.5rem;border-left:3px solid #f44;margin:0.25rem 0}
.st{background:#0a1a0a;padding:0.5rem;border-left:3px solid #4f4;margin:0.25rem 0}
.sub{background:#111;padding:0.5rem;border-radius:6px;border:1px solid #444;max-height:150px;overflow-y:auto}'''
st.markdown(f'<style>{CSS}</style>',unsafe_allow_html=True)
st.title('FootballAI Pro')
st.caption('Professional AI Football Analysis Platform - Elite Player Edition')

def build_pipeline():
  p=type('P',(),{'__init__':lambda s:None})()
  p.ra=RunningAnalyzer();p.fiq=FootballIQAnalyzer();p.hm=HeatmapAnalyzer()
  p.pa=PoseAnalyzer();p.sse=SSE();p.bd=BallDetector();p.pm=PlayerMemory();p.vz=Visualizer()
  p.sel=None;p.im={};p.nid=1;p.tr=YOLO('yolo11n.pt');p.spd_hist={}
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
      else:r=requests.get(url,stream=True)
      r.raise_for_status()
      for c in r.iter_content(8192):
        if c:t.write(c)
      t.close();vp=t.name
  elif uf is not None:
    t=tempfile.NamedTemporaryFile(delete=False);t.write(uf.read());vp=t.name
  return vp

def process_frame(pr,fr,sel,sk_fr,fidx,dset):
  if fidx%(sk_fr+1)!=0:return fr,None,None,None
  r=pr.tr.track(fr,persist=True,tracker='bytetrack.yaml',conf=0.3,iou=0.45,classes=[0],imgsz=1280,verbose=False)[0]
  bp=pr.bd.detect(fr);findings=[]
  if r.boxes is not None:
    for b in r.boxes:
      if b.id is None:continue
      iid=int(b.id[0]);tid=pr.im.get(iid)
      if tid is None:pr.im[iid]=pr.nid;tid=pr.nid;pr.nid+=1
      bb=[int(x) for x in b.xyxy[0]];cx,cy=(bb[0]+bb[2])//2,(bb[1]+bb[3])//2
      pr.pm.create_or_update(tid,bb,fr)
      if tid==sel:
        pr.ra.update(sel,(cx,cy));pr.fiq.update(sel,(cx,cy));pr.hm.update(cx,cy)
        touch=pr.bd.check_touch(bb,bp)
        if touch and 'ball' not in dset:dset.add('ball')
        _,pd=pr.pa.analyze(fr,bb)
        if pd and 'kp' in pd:pr.vz.draw_pose(fr,pd['kp'],POSE_CONNS)
        if pd:
          f=pr.sse.analyze(pd,pr.ra.get_speed(sel),pr.fiq.td(sel))
          for x in f:
            dt=x.get('det','')
            if dt and dt not in dset:dset.add(dt);findings.append(x)
          if tid not in pr.spd_hist:pr.spd_hist[tid]=[]
          pr.spd_hist[tid].append(pr.ra.get_speed(sel))
  return fr,findings,bp,None

inp=st.radio('Input',['URL (up to 4GB)','Upload (up to 200MB)'],horizontal=True)
vp=None;url=None;uf=None
if inp.startswith('URL'):url=st.text_input('Paste video URL')
else:uf=st.file_uploader('Upload',['mp4','avi','mov','mkv'])
vp=get_video(inp,url,uf)

if vp is not None:
  pr=st.session_state.p
  if pr is None:pr=build_pipeline();st.session_state.p=pr
  cap=cv2.VideoCapture(vp);ret,fr=cap.read();cap.release()
  if not ret:st.error('Could not read video')
  else:
    pr.hm.initialize(fr)
    r=pr.tr.track(fr,persist=True,tracker='bytetrack.yaml',conf=0.3,iou=0.45,classes=[0],imgsz=1280,verbose=False)[0]
    if r.boxes is not None:
      for b in r.boxes:
        if b.id is None:continue
        iid=int(b.id[0])
        if iid not in pr.im:pr.im[iid]=pr.nid;pr.nid+=1
    pl=list(pr.im.values())
    if not pl:st.error('No players detected.')
    else:
      sel=st.selectbox('Select Player',pl);pr.sel=sel
      if st.button('Start Analysis'):
        cap=cv2.VideoCapture(vp)
        fps=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ot=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name
        out=cv2.VideoWriter(ot,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        frames=0;last_fb='';all_findings=[];dset=set()
        sk=st.slider('Skip',0,5,2);bar=st.progress(0)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with st.spinner('Analyzing...'):
          while True:
            ret,fr=cap.read()
            if not ret:break
            ofr=fr.copy()
            _,findings,bp,_=process_frame(pr,ofr,sel,sk,frames,dset)
            if findings:all_findings.extend(findings);last_fb=findings[0].get('p','')+' - '+findings[0].get('d','')
            ofr=pr.hm.overlay(ofr)
            path=pr.ra.get_path(sel)
            if len(path)>1:pr.vz.draw_trail(ofr,[(int(p[0]),int(p[1])) for p in path])
            if bp:
              cv2.circle(ofr,bp,8,(0,0,255),-1);cv2.circle(ofr,bp,12,(0,0,255),2)
            if last_fb:
              cv2.rectangle(ofr,(0,h-60),(w,h-5),(0,0,0),-1)
              cv2.putText(ofr,last_fb[:80],(20,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            out.write(ofr);frames+=1;bar.progress(min(frames/total,1.0))
        cap.release();out.release();bar.empty()
        st.success(f'Done! {frames} frames.')

        fiq_st=pr.fiq.stats(sel);ra_st=pr.ra.get_stats(sel)
        assess=pr.sse.assess(fiq_st,all_findings)
        rpt=ReportGen();rpt.gen(sel,ra_st,fiq_st);st.session_state.rpt=rpt
        col1,col2=st.columns([2,1])
        with col1:
          with open(ot,'rb') as f:st.video(f.read())
        with col2:
          st.markdown('<div class="c">',unsafe_allow_html=True)
          st.subheader('Dashboard')
          sc=assess
          st.markdown(f'**Overall:** {sc["os"]}/100');st.progress(sc["os"]/100)
          c1,c2=st.columns(2)
          with c1:
            st.markdown(f"<div class='s'>Tech: {sc['ts']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Physical: {sc['ps']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Tactical: {sc['tcs']}</div>",unsafe_allow_html=True)
          with c2:
            st.markdown(f"<div class='s'>Biomech: {sc['bs']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Fatigue: {sc['fat']}%</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='s'>Risk: {sc['risk']}</div>",unsafe_allow_html=True)
          st.markdown('</div>',unsafe_allow_html=True)
          st.markdown('<div class="sub">',unsafe_allow_html=True)
          st.subheader('Coach')
          st.info(last_fb or 'Good mechanics')
          st.markdown('</div>',unsafe_allow_html=True)
          c1,c2,c3=st.columns(3)
          c1.metric('Speed',f"{ra_st['speed']:.1f}")
          c2.metric('Distance',f"{ra_st['distance']:.0f}")
          c3.metric('Sprints',fiq_st['sprints'])
          sh=pr.spd_hist.get(sel,[])
          if len(sh)>1:
            st.image(pr.vz.create_speed_graph(sh),use_container_width=True)
            st.image(pr.vz.create_distance_bar(ra_st['distance']),use_container_width=True)
          if sc['weak']:
            st.markdown('**Weaknesses**')
            for w in sc['weak']:st.markdown(f"<div class='w'>{w}</div>",unsafe_allow_html=True)
          if sc['strong']:
            st.markdown('**Strengths**')
            for s in sc['strong']:st.markdown(f"<div class='st'>{s}</div>",unsafe_allow_html=True)

        st.markdown('---');st.subheader('Export')
        ec1,ec2,ec3,ec4=st.columns(4);rpt=st.session_state.rpt
        if ec1.button('CSV'):
          fp=os.path.join(tempfile.gettempdir(),f'p{sel}.csv')
          rpt.to_csv(fp)
          with open(fp,'rb') as f:st.download_button('DL CSV',f,'report.csv','text/csv')
        if ec2.button('JSON'):
          jd=rpt.to_json(sel)
          st.download_button('DL JSON',jd,'report.json','application/json')
        if ec3.button('PDF'):
          fp=os.path.join(tempfile.gettempdir(),f'p{sel}.pdf')
          rpt.to_pdf(fp,sc)
          with open(fp,'rb') as f:st.download_button('DL PDF',f,'report.pdf','application/pdf')
        if ec4.button('Plan'):
          st.download_button('DL Plan',pr.sse.gen_plan(all_findings),'plan.txt','text/plain')
elif url is None:st.write('Provide a video.')

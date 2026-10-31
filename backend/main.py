from __future__ import annotations

import asyncio
# Load .env automatically so GROQ_API_KEY is available without shell exports
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

BASE = Path(__file__).resolve().parent.parent

# Ensure the venv's Scripts/bin dir is on PATH so shutil.which() and subprocess
# can find yt-dlp and other venv-installed tools even when uvicorn is invoked
# via an absolute path.
_venv_scripts = BASE / ".venv" / ("Scripts" if os.name == "nt" else "bin")
if _venv_scripts.exists():
    os.environ["PATH"] = str(_venv_scripts) + os.pathsep + os.environ.get("PATH", "")
STORAGE = BASE / "storage"
TEMP = BASE / "temp"
FRONTEND = BASE / "frontend" / "dist"
STORAGE.mkdir(exist_ok=True)
TEMP.mkdir(exist_ok=True)

DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE / 'clipforge.db'}")
engine = create_engine(DB_URL, pool_pre_ping=True, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase): pass

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(500))
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    requested_count: Mapped[int] = mapped_column(Integer)
    clip_length: Mapped[str] = mapped_column(String(20))
    caption_style: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(120), default="Queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Clip(Base):
    __tablename__ = "clips"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(String(300))
    score: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    file_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_words: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

app = FastAPI(title="ClipForge", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if FRONTEND.exists(): app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

executor = ThreadPoolExecutor(max_workers=4)
locks: dict[str, threading.Lock] = {}

class CreateJob(BaseModel):
    url: str
    requested_count: int = Field(10, ge=1, le=100)
    clip_length: str = "30-60s"
    caption_style: str = "bold-pop"
    @field_validator("url")
    @classmethod
    def valid_url(cls, v: str):
        if not re.match(r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+", v):
            raise ValueError("Paste a valid public YouTube URL.")
        return v.strip()
    @field_validator("clip_length")
    @classmethod
    def valid_length(cls, v: str):
        if v not in {"15-30s", "30-60s", "60-90s"}: raise ValueError("Clip length must be 15-30s, 30-60s, or 60-90s.")
        return v

class EditClip(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    @field_validator("end")
    @classmethod
    def ordered(cls, v, info):
        start = info.data.get("start")
        if start is not None and v <= start: raise ValueError("End must be after start.")
        if start is not None and v - start < 5: raise ValueError("Clip must be at least 5 seconds.")
        return v


def update_job(job_id: str, **values):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job:
            for k, v in values.items(): setattr(job, k, v)
            job.updated_at = datetime.utcnow(); db.commit()

def run(cmd: list[str], cwd: Path | None = None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0: raise RuntimeError((p.stderr or p.stdout)[-1800:])
    return p.stdout

def seconds_for(value: str) -> tuple[int, int]: return {"15-30s": (15,30), "30-60s": (30,60), "60-90s": (60,90)}[value]

def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\[.*\]", text, re.S)
    if not match: raise ValueError("Groq returned invalid JSON.")
    return json.loads(match.group(0))

def groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY is not configured. Add it to the server environment before processing videos.")
    from groq import Groq
    return Groq(api_key=key)

def transcribe(audio: Path, job_id: str) -> list[dict]:
    client = groq_client(); duration = float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(audio)]).strip())
    words=[]; chunk=480; offset=0
    while offset < duration:
        part = TEMP / f"{job_id}_{int(offset)}.mp3"
        run(["ffmpeg","-y","-ss",str(offset),"-t",str(chunk),"-i",str(audio),"-ac","1","-ar","16000","-b:a","64k",str(part)])
        with open(part,"rb") as fh:
            result = client.audio.transcriptions.create(file=(part.name,fh.read()), model="whisper-large-v3", response_format="verbose_json", timestamp_granularities=["word"])
        for w in (getattr(result,"words",None) or []):
            item = w if isinstance(w,dict) else w.model_dump()
            words.append({"word":item.get("word","").strip(),"start":float(item.get("start",0))+offset,"end":float(item.get("end",0))+offset})
        part.unlink(missing_ok=True); offset += chunk
        update_job(job_id, progress=min(40, 10+int(offset/max(duration,1)*30)), stage=f"Transcribing audio ({min(offset,duration):.0f}s / {duration:.0f}s)")
    return words

def analyze(words: list[dict], requested: int, clip_length: str, job_id: str) -> list[dict]:
    client=groq_client(); lo,hi=seconds_for(clip_length)
    transcript=" ".join(f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}" for w in words)
    prompt=f"""You are a viral short-form editor. Analyze this timestamped transcript and select up to {requested} non-overlapping moments of {lo}-{hi} seconds. Return ONLY a JSON array. Each item must have start (number), end (number), hook_title (string under 80 chars), virality_score (integer 1-100), reason (string). Select fewer if there are not enough genuinely strong moments. Start/end must align to words. Rank strongest first. Transcript:\n{transcript}"""
    result=client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0.2, response_format={"type":"json_object"})
    data=parse_json(result.choices[0].message.content)
    if isinstance(data,dict): data=data.get("clips",[])
    accepted=[]
    for item in sorted(data,key=lambda x:int(x.get("virality_score",0)),reverse=True):
        s=float(item["start"]); e=float(item["end"])
        if e-s < lo: e=s+lo
        if e-s > hi: e=s+hi
        if s < 0 or any(not (e <= c["start"] or s >= c["end"]) for c in accepted): continue
        accepted.append({"start":s,"end":e,"title":str(item.get("hook_title","Untitled hook")),"score":max(1,min(100,int(item.get("virality_score",50)))),"reason":str(item.get("reason","Strong conversational moment."))})
        if len(accepted)>=requested: break
    update_job(job_id, progress=52, stage=f"Analyzing transcript ({len(accepted)} strong moments)")
    return accepted

def make_ass(words: list[dict], start: float, end: float, style: str, path: Path):
    colors={"bold-pop":("&H00FFFFFF","&H0000D7FF"),"minimal":("&H00FFFFFF","&H0000FFFF"),"neon":("&H00FFFFFF","&H00FF70FF")}; primary,highlight=colors.get(style,colors["bold-pop"])
    header=f"""[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,74,{primary},{primary},&H00000000,&H99000000,1,0,0,0,100,100,0,0,1,5,2,5,80,80,360,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
    lines=[]
    selected=[w for w in words if w["end"]>=start and w["start"]<=end]
    for i,w in enumerate(selected):
        ss=max(start,w["start"]); ee=min(end,w["end"]); nxt=selected[i+1]["start"] if i+1<len(selected) else w["end"]
        window=selected[max(0,i-3):i+4]
        show=" ".join(("{\\c"+highlight+"}"+x["word"]+"{\\c"+primary+"}" if x is w else x["word"]) for x in window)
        def ts(x):
            x=max(0,x); return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:04.1f}"
        lines.append(f"Dialogue: 0,{ts(ss)},{ts(min(end,max(ee,nxt)))},Default,,0,0,0,,{show}")
    path.write_text(header+"\n".join(lines),encoding="utf-8")

def render_clip(job_id: str, clip: Clip, source: Path, words: list[dict], style: str):
    outdir=STORAGE/job_id; outdir.mkdir(exist_ok=True)
    ass=TEMP/f"{clip.id}.ass"; make_ass(words,clip.start,clip.end,style,ass)
    out=outdir/f"clip-{clip.rank:02d}.mp4"
    # Detect a stable speaker center; if no face exists, use a blurred background with fit-to-width foreground.
    cap=cv2.VideoCapture(str(source)); centers=[]; cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    for _ in range(12):
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); faces=cascade.detectMultiScale(gray,1.1,5)
        if len(faces):
            x,y,w,h=max(faces,key=lambda f:f[2]*f[3]); centers.append((x+w/2)/frame.shape[1])
    cap.release(); center=sum(centers)/len(centers) if centers else .5
    # crop around detected center; scale to portrait. Blur fallback remains visually safe for landscape footage.
    ass_path=ass.as_posix().replace(":", "\\:")
    vf=f"scale=1080:-2,crop=1080:1920:0:0,subtitles='{ass_path}'"
    # FFmpeg's crop after scaling uses a vertical center crop; for landscape videos, pad with a blurred base.
    vf=f"split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=30[bg];[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,subtitles='{ass_path}'"
    run(["ffmpeg","-y","-ss",str(clip.start),"-t",str(clip.end-clip.start),"-i",str(source),"-filter_complex",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

def process_job(job_id: str):
    lock=locks.setdefault(job_id,threading.Lock())
    with lock:
        jobdir=TEMP/job_id; jobdir.mkdir(exist_ok=True)
        try:
            with SessionLocal() as db:
                job=db.get(Job,job_id)
                if not job: raise RuntimeError(f"Job {job_id} not found in database.")
            update_job(job_id,status="processing",progress=3,stage="Downloading source video",error=None)
            source=jobdir/"source.mp4"
            run(["yt-dlp","--no-playlist","--merge-output-format","mp4","-o",str(source),"--print","after_move:title",job.url])
            update_job(job_id,progress=12,stage="Extracting audio")
            audio=jobdir/"audio.mp3"; run(["ffmpeg","-y","-i",str(source),"-vn","-ac","1","-ar","16000","-b:a","64k",str(audio)])
            words=transcribe(audio,job_id); moments=analyze(words,job.requested_count,job.clip_length,job_id)
            with SessionLocal() as db:
                db.query(Clip).filter(Clip.job_id==job_id).delete(); db.commit()
                clips=[]
                for i,m in enumerate(moments,1):
                    c=Clip(id=str(uuid.uuid4()),job_id=job_id,rank=i,start=m["start"],end=m["end"],title=m["title"],score=m["score"],reason=m["reason"],transcript_words=words,status="queued"); db.add(c); clips.append(c)
                db.commit()
                for i,c in enumerate(clips,1):
                    try:
                        update_job(job_id,progress=52+int(i/max(len(clips),1)*43),stage=f"Rendering clip {i}/{len(clips)}")
                        out=render_clip(job_id,c,source,words,job.caption_style); c.file_name=out.name; c.status="ready"
                    except Exception as e: c.status="failed"; c.error=str(e)
                    db.commit()
            persisted=STORAGE/job_id; persisted.mkdir(exist_ok=True)
            shutil.copy2(source, persisted/"source.mp4")
            update_job(job_id,status="completed",progress=100,stage=f"Complete — {len(moments)} clips ready")
        except Exception as e:
            msg=str(e)
            if "Sign in" in msg or "age" in msg.lower(): msg="YouTube blocked this video (private, age-restricted, or sign-in required). Try a public video."
            update_job(job_id,status="failed",progress=0,stage="Processing failed",error=msg)
        finally: shutil.rmtree(jobdir,ignore_errors=True)

def job_url(job_id):
    with SessionLocal() as db: return db.get(Job,job_id).url

@app.get("/api/health")
def health(): return {"ok":True,"groq_configured":bool(os.getenv("GROQ_API_KEY")),"ffmpeg":shutil.which("ffmpeg") is not None,"yt_dlp":shutil.which("yt-dlp") is not None}

@app.post("/api/jobs")
def create_job(req: CreateJob):
    jid=str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Job(id=jid,url=req.url,requested_count=req.requested_count,clip_length=req.clip_length,caption_style=req.caption_style,status="queued",progress=0,stage="Queued")); db.commit()
    executor.submit(process_job,jid); return {"id":jid}

@app.get("/api/jobs")
def list_jobs():
    with SessionLocal() as db:
        jobs=db.scalars(select(Job).order_by(Job.created_at.desc())).all()
        return [{"id":j.id,"url":j.url,"title":j.title,"requested_count":j.requested_count,"clip_length":j.clip_length,"caption_style":j.caption_style,"status":j.status,"progress":j.progress,"stage":j.stage,"error":j.error,"created_at":j.created_at.isoformat(),"clips":len(db.scalars(select(Clip).where(Clip.job_id==j.id)).all())} for j in jobs]

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    with SessionLocal() as db:
        j=db.get(Job,job_id)
        if not j: raise HTTPException(404,"Job not found")
        clips=db.scalars(select(Clip).where(Clip.job_id==job_id).order_by(Clip.rank)).all()
        return {"id":j.id,"url":j.url,"title":j.title,"requested_count":j.requested_count,"clip_length":j.clip_length,"caption_style":j.caption_style,"status":j.status,"progress":j.progress,"stage":j.stage,"error":j.error,"created_at":j.created_at.isoformat(),"clips":[{"id":c.id,"rank":c.rank,"start":c.start,"end":c.end,"title":c.title,"score":c.score,"reason":c.reason,"status":c.status,"error":c.error,"download":f"/api/clips/{c.id}/download"} for c in clips]}

@app.get("/api/jobs/{job_id}/events")
async def events(job_id:str):
    async def stream():
        last=None
        for _ in range(720):
            with SessionLocal() as db:
                j=db.get(Job,job_id)
                if not j: yield f"data: {json.dumps({'error':'Job not found'})}\n\n"; return
                payload={"status":j.status,"progress":j.progress,"stage":j.stage,"error":j.error}
            marker=json.dumps(payload,sort_keys=True)
            if marker!=last: yield f"data: {marker}\n\n"; last=marker
            if payload["status"] in {"completed","failed"}: return
            await asyncio.sleep(1)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/api/clips/{clip_id}/download")
def download_clip(clip_id:str):
    with SessionLocal() as db:
        c=db.get(Clip,clip_id)
        if not c or not c.file_name: raise HTTPException(404,"Clip is not ready")
        path=STORAGE/c.job_id/c.file_name
        if not path.exists(): raise HTTPException(404,"Clip file has expired")
        return FileResponse(path,media_type="video/mp4",filename=c.file_name)

@app.get("/api/jobs/{job_id}/zip")
def download_zip(job_id:str):
    with SessionLocal() as db: clips=db.scalars(select(Clip).where(Clip.job_id==job_id,Clip.status=="ready")).all()
    if not clips: raise HTTPException(404,"No rendered clips")
    zip_path=TEMP/f"{job_id}.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for c in clips:
            path=STORAGE/job_id/c.file_name
            if path.exists(): z.write(path,c.file_name)
    return FileResponse(zip_path,media_type="application/zip",filename=f"clipforge-{job_id[:8]}.zip")

@app.post("/api/clips/{clip_id}/rerender")
def rerender_clip(clip_id:str, req:EditClip):
    with SessionLocal() as db:
        c=db.get(Clip,clip_id); j=db.get(Job,c.job_id) if c else None
        if not c or not j: raise HTTPException(404,"Clip not found")
        c.start=req.start; c.end=req.end; c.status="queued"; c.error=None; c.file_name=None; db.commit()
        source=STORAGE/j.id/"source.mp4"
        if not source.exists(): raise HTTPException(409,"Source video expired; rerun the job to edit this clip.")
        executor.submit(lambda: rerender_worker(c.id,j.id))
    return {"ok":True}

def rerender_worker(clip_id,job_id):
    try:
        with SessionLocal() as db: c=db.get(Clip,clip_id); j=db.get(Job,job_id); words=c.transcript_words or []
        source=STORAGE/job_id/"source.mp4"; out=render_clip(job_id,c,source,words,j.caption_style)
        with SessionLocal() as db: c=db.get(Clip,clip_id); c.file_name=out.name; c.status="ready"; db.commit()
    except Exception as e:
        with SessionLocal() as db: c=db.get(Clip,clip_id); c.status="failed"; c.error=str(e); db.commit()

@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id:str):
    with SessionLocal() as db:
        if not db.get(Job,job_id): raise HTTPException(404,"Job not found")
    executor.submit(process_job,job_id); return {"ok":True}

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id:str):
    with SessionLocal() as db:
        j=db.get(Job,job_id)
        if not j: raise HTTPException(404,"Job not found")
        db.query(Clip).filter(Clip.job_id==job_id).delete(); db.delete(j); db.commit()
    shutil.rmtree(STORAGE/job_id,ignore_errors=True); shutil.rmtree(TEMP/job_id,ignore_errors=True); return {"ok":True}

@app.on_event("startup")
def cleanup():
    cutoff=time.time()-86400
    for folder in [STORAGE,TEMP]:
        for p in folder.iterdir():
            try:
                if p.stat().st_mtime<cutoff: shutil.rmtree(p) if p.is_dir() else p.unlink()
            except FileNotFoundError: pass

@app.get("/{path:path}")
def spa(path: str):
    if FRONTEND.exists() and (FRONTEND/"index.html").exists(): return FileResponse(FRONTEND/"index.html")
    return {"service":"ClipForge","message":"Frontend bundle not built. Run npm install && npm run build in frontend/."}

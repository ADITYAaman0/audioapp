# file: app.py
"""
Advanced TTS Studio  –  Streamlit Dark-Theme Demo
Convert TXT, DOCX, PDF → speech with intelligent character filtering,
voice-rate / pitch / volume controls, live waveform + 3-D Lottie avatar.
Ready for Python 3.13 (no audioop / pydub).
"""

import re, io, base64, tempfile, os, pathlib, subprocess, wave, json, requests
from typing import Dict, List

import numpy as np
import streamlit as st
import pyttsx3
from docx import Document
import fitz                           # PyMuPDF
from streamlit_lottie import st_lottie

# ──────────────────────────────── Page config ────────────────────────────────
st.set_page_config(
    page_title="Advanced TTS Studio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────── Theme override ─────────────────────────────
st.markdown(
    """
    <style>
    /* Global dark background */
    html, body, [class*="st-"] {
        background-color:#0B1120;
        color:#E5E7EB;
        font-family:'Inter',sans-serif;
    }
    /* Title */
    .title       {text-align:center; font-weight:800; font-size:3rem; color:#C084FC;}
    /* Section cards */
    .card        {background:#111827;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1.2rem;border:1px solid #1F2937;}
    /* Drag’n’drop dashed box */
    .uploader > div:first-child {border:2px dashed #475569;border-radius:10px;}
    /* Gradient CTA button */
    .stButton>button {
        border:0;
        padding:0.6rem 1.2rem;
        border-radius:8px;
        background:linear-gradient(90deg,#7C3AED 0%,#EC4899 100%);
        font-weight:600;color:#fff;
    }
    /* Sliders  — thumb/track colors */
    input[type=range] {
        accent-color:#C084FC;
    }
    /* Small footer */
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────── Constants ────────────────────────────────
DEFAULT_RATE   = 175
DEFAULT_VOLUME = 1.0
DEFAULT_PITCH  = 1.0            # 0.5–1.5
ALLOWED_RE     = re.compile(r"[A-Za-z0-9\s()]+")

# ──────────────────────────────── Helpers ────────────────────────────────
def load_lottie(url:str)->dict:
    try:
        r=requests.get(url,timeout=6)
        if r.status_code==200:
            return r.json()
    except Exception:
        pass
    return {}

def extract_text(uploaded, suffix)->str:
    if suffix==".pdf":
        with fitz.open(stream=uploaded.read(), filetype="pdf") as doc:
            return " ".join(p.get_text() for p in doc)
    if suffix==".docx":
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".docx")
        tmp.write(uploaded.read()); tmp.close()
        txt=" ".join(p.text for p in Document(tmp.name).paragraphs)
        os.unlink(tmp.name); return txt
    if suffix==".txt":
        return uploaded.read().decode(errors="ignore")
    return ""

def clean_text(raw:str)->str:
    filtered="".join(ALLOWED_RE.findall(raw))
    return re.sub(r"\((.*?)\)",lambda m:m.group(1),filtered)

def list_voices()->Dict[str,str]:
    engine=pyttsx3.init()
    vdict={}
    for v in engine.getProperty("voices"):
        lang=v.languages[0]
        if isinstance(lang,bytes):
            lang=lang.decode(errors="ignore")
        vdict[v.id]=f"{v.name} – {lang}"
    return vdict

def synthesize_wav(text:str, voice_id:str, rate:int, volume:float, pitch:float)->bytes:
    engine=pyttsx3.init()
    engine.setProperty("voice",voice_id)
    engine.setProperty("rate",rate)
    engine.setProperty("volume",volume)
    # Pitch is not supported on all engines; safe try:
    try:
        engine.setProperty("pitch", int(pitch*100))
    except Exception:
        pass
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); tmp.close()
    engine.save_to_file(text,tmp.name); engine.runAndWait()
    data=open(tmp.name,"rb").read(); os.remove(tmp.name)
    return data

def convert_wav_to_mp3(wav_bytes:bytes, bitrate:str="192k")->bytes:
    """Requires ffmpeg installed and on PATH."""
    wav_file=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); wav_file.write(wav_bytes); wav_file.close()
    mp3_file=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); mp3_file.close()
    cmd=["ffmpeg","-y","-i",wav_file.name,"-b:a",bitrate,mp3_file.name]
    try:
        subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        mp3_data=open(mp3_file.name,"rb").read()
    finally:
        os.remove(wav_file.name); os.remove(mp3_file.name)
    return mp3_data

def wav_waveform(wav_bytes:bytes, buckets:int=80)->List[float]:
    buffer=io.BytesIO(wav_bytes)
    with wave.open(buffer,"rb") as wf:
        frames=wf.readframes(wf.getnframes())
        samples=np.frombuffer(frames,dtype=np.int16)
        # Down-sample into RMS buckets
        samples=np.abs(samples)
        bucket_size=int(len(samples)/buckets)+1
        rms=[float(np.mean(samples[i:i+bucket_size])) for i in range(0,len(samples),bucket_size)]
        norm=np.array(rms)/max(rms)
    return norm.tolist()

def download_link(data:bytes, filename:str, mime:str)->str:
    b64=base64.b64encode(data).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">Download</a>'

# ──────────────────────────────── Layout ────────────────────────────────
st.markdown('<div class="title">Advanced TTS Studio</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94A3B8'>Convert text, PDF, and Word files to high-quality audio with intelligent character processing and immersive 3-D animations.</p>", unsafe_allow_html=True)
st.markdown("")

left, right = st.columns((2,1),gap="large")

# 3-D Lottie avatar
with right:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("### AI Character")
    st_lottie(
        load_lottie("https://assets8.lottiefiles.com/packages/lf20_1pxqjqps.json"),
        height=380,
        key="avatar",
    )
    st.markdown('<small style="color:#64748B">Click and drag to rotate<br>Scroll to zoom</small>', unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

with left:
    # ── Upload Section ────────────────────────────────────────────────────
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("### Upload Files")
    uploaded=st.file_uploader(
        "Drag & drop or browse to choose (TXT, DOCX, PDF, max 10 MB)",
        type=["txt","docx","pdf"],
        label_visibility="collapsed",
        help="Supported formats: .txt . docx . pdf",
    )
    st.markdown("</div>",unsafe_allow_html=True)

    # ── Text Content Section ──────────────────────────────────────────────
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("### Text Content")
    sample_text=("India as global waste management innovator Community Empowerment. "
                 "Grassroots environmental stewardship development …")
    # Session storage for text
    if "text_content" not in st.session_state: st.session_state.text_content=""
    colA,colB=st.columns([1,8])
    if colA.button("Load Sample"):
        st.session_state.text_content=sample_text
    colB.button("Clear", on_click=lambda:st.session_state.pop("text_content",None))
    if uploaded and st.session_state.get("uploaded_parsed")!=uploaded.name:
        st.session_state.text_content=extract_text(uploaded, pathlib.Path(uploaded.name).suffix.lower())
        st.session_state.uploaded_parsed=uploaded.name
    text=st.text_area(
        "",
        value=st.session_state.get("text_content",""),
        height=180,
        key="text_input",
    )
    st.markdown(f"<small style='color:#64748B'>Characters: {len(text):,} &nbsp; Words: {len(text.split()):,}</small>", unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # ── Audio Settings Section ────────────────────────────────────────────
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("### Audio Settings")

    voices=list_voices()
    voice_id=st.selectbox("Voice Selection", list(voices.keys()), format_func=lambda k:voices[k])
    rate=st.slider("Speech Rate",0.5,2.0,1.0,0.05, format="%.2fx")
    pitch=st.slider("Voice Pitch",0.5,1.5,DEFAULT_PITCH,0.05, format="%.2f")
    volume=st.slider("Volume",0.0,1.0,DEFAULT_VOLUME,0.05, format="%.0f%%",label_visibility="visible")

    output_format=st.selectbox("Output Format", ["WAV (High Quality)","MP3 (Compressed)"])
    st.markdown("#### Settings Preview")
    st.markdown(f"""
    • **Voice:** {voices[voice_id]}  
    • **Speed:** {rate:.2f}×  
    • **Pitch:** {pitch:.2f}  
    • **Volume:** {int(volume*100)} %  
    • **Format:** {output_format.split()[0]}
    """)
    st.markdown("</div>",unsafe_allow_html=True)

    # ── Generate Button ───────────────────────────────────────────────────
    if st.button("Generate Audio"):
        if not text.strip():
            st.warning("Please enter or upload some text.")
        else:
            cleaned=clean_text(text)
            with st.spinner("Synthesizing audio…"):
                wav_bytes=synthesize_wav(cleaned,voice_id,int(rate*100),volume,pitch)
                if output_format.startswith("MP3"):
                    try:
                        audio_bytes=convert_wav_to_mp3(wav_bytes)
                        mime="audio/mpeg"; ext="mp3"
                    except Exception as e:
                        st.error("FFmpeg not found. Falling back to WAV output.")
                        audio_bytes=wav_bytes; mime="audio/wav"; ext="wav"
                else:
                    audio_bytes=wav_bytes; mime="audio/wav"; ext="wav"
            st.success("Audio generated!")

            # ── Generated Audio Section ───────────────────────────────────
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown("### Generated Audio")

            # Waveform preview
            amplitude=wav_waveform(wav_bytes, buckets=100)
            st.bar_chart(amplitude,use_container_width=True)

            # Player + slider
            st.audio(audio_bytes, format=mime)

            size_kb=len(audio_bytes)/1024
            st.markdown(f"""
            **Duration**&nbsp; ~{len(wav_bytes)/32000:.0f}s &nbsp;&nbsp;
            **Size**&nbsp; ~{size_kb:.0f} KB &nbsp;&nbsp;
            **Format**&nbsp; {ext.upper()}
            """)
            st.markdown(download_link(audio_bytes,f"speech.{ext}",mime),unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)

    # ── Tips Section ───────────────────────────────────────────────────────
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("#### 💡 Pro Tips")
    st.write("""
    • Use slower rates for better comprehension  
    • Adjust pitch for character-specific voices  
    • WAV format offers maximum fidelity  
    • Higher volumes work better with headphones  
    """)
    st.markdown("</div>",unsafe_allow_html=True)

# ───────────── Footer ─────────────
st.markdown(
    """
    <div style='text-align:center;padding:1.5rem 0;color:#94A3B8'>
        Experience the future of text-to-speech with AI-powered character animations
        <br>
        <span style='color:#7C3AED'>⚙️ Smart Character Filtering</span> &nbsp;|&nbsp;
        <span style='color:#C084FC'>🎛 Advanced Voice Controls</span> &nbsp;|&nbsp;
        <span style='color:#4ADE80'>🛠 Multi-Format Export</span>
    </div>
    """,
    unsafe_allow_html=True,
)


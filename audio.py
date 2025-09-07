# ────────────────────────────────────────────────────────────────
#  Advanced TTS Studio – Cloud-friendly (edge-tts backend)
#  • Multiple accents, genders & speaking styles
#  • Tone presets (Deep, Bright, Robot, Narrator …)
#  • Dark neo-UI + 3-D Lottie avatar
#  • WAV or MP3 download   (no audioop / pydub / pyttsx3)
# ────────────────────────────────────────────────────────────────
import re, io, base64, tempfile, os, pathlib, asyncio, json, subprocess, wave
from typing import List, Dict

import streamlit as st
import numpy as np
import fitz                     # PyMuPDF  (PDF ➜ text)
from docx import Document
from streamlit_lottie import st_lottie
import edge_tts                 # Microsoft online TTS
import requests

# ─────────────── Global look & feel ───────────────
st.set_page_config(page_title="Advanced TTS Studio", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    html,body,[class*="st-"]{background:#0B1120;color:#E5E7EB;font-family:Inter,sans-serif}
    .title{font-weight:800;font-size:3rem;color:#C084FC;text-align:center}
    .card{background:#111827;border-radius:12px;padding:1.25rem 1.6rem;margin-bottom:1.3rem;
          border:1px solid #1F2937}
    .stButton>button{background:linear-gradient(90deg,#7C3AED 0%,#EC4899 100%);
                     color:#fff;border:0;padding:.55rem 1.3rem;border-radius:8px;font-weight:600}
    input[type=range]{accent-color:#C084FC}
    footer{visibility:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────── Utils ───────────────
ALLOWED_RE = re.compile(r"[A-Za-z0-9\s()]+")

TONE_PRESETS = {
    "Neutral (Default)": dict(rate="+0%", pitch="+0Hz"),
    "Deep Male":         dict(rate="-5%", pitch="-4Hz"),
    "Bright Female":     dict(rate="+5%", pitch="+6Hz"),
    "Robot":             dict(rate="+20%", pitch="-10Hz"),
    "Story Narrator":    dict(rate="-10%", pitch="-2Hz"),
}

@st.cache_data(show_spinner="▶️  Fetching voice catalogue…")
def load_voice_catalog() -> List[Dict]:
    """Query Microsoft Edge TTS for full voice list (once per session)."""
    return asyncio.run(edge_tts.list_voices())

def lottie(url:str)->Dict:
    try:
        r=requests.get(url,timeout=6)
        if r.status_code==200: return r.json()
    except Exception: pass
    return {}

def extract_text(buf:io.BytesIO, suffix:str)->str:
    if suffix==".pdf":
        with fitz.open(stream=buf.read(), filetype="pdf") as doc:
            return " ".join(p.get_text() for p in doc)
    if suffix==".docx":
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".docx")
        tmp.write(buf.read()); tmp.close()
        txt=" ".join(p.text for p in Document(tmp.name).paragraphs)
        os.unlink(tmp.name); return txt
    if suffix==".txt":
        return buf.read().decode(errors="ignore")
    return ""

def clean_text(raw:str)->str:
    filtered="".join(ALLOWED_RE.findall(raw))
    return re.sub(r"\((.*?)\)", lambda m: m.group(1), filtered)

async def edge_tts_async(text:str, voice:str, rate:str, pitch:str) -> bytes:
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); tmp.close()
    await edge_tts.Communicate(
        text, voice, rate=rate, pitch=pitch
    ).save(tmp.name)
    data=pathlib.Path(tmp.name).read_bytes(); os.remove(tmp.name)
    return data

def wav_from_mp3(mp3:bytes) -> bytes:
    """FFmpeg is available on Streamlit Cloud ─ convert to WAV for waveform display."""
    mp3f=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); mp3f.write(mp3); mp3f.close()
    wavf=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); wavf.close()
    subprocess.run(["ffmpeg","-loglevel","quiet","-y","-i",mp3f.name,wavf.name])
    wav=pathlib.Path(wavf.name).read_bytes()
    os.remove(mp3f.name); os.remove(wavf.name)
    return wav

def waveform_data(wav:bytes,buckets:int=90)->List[float]:
    with wave.open(io.BytesIO(wav)) as wf:
        samples=np.frombuffer(wf.readframes(wf.getnframes()),dtype=np.int16)
    samples=np.abs(samples); size=len(samples)//buckets+1
    levels=[float(np.mean(samples[i:i+size])) for i in range(0,len(samples),size)]
    mx=max(levels) or 1
    return list(np.array(levels)/mx)

def download_tag(data:bytes, fname:str, mime:str) -> str:
    b64=base64.b64encode(data).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{fname}">Download</a>'

# ─────────────── UI ───────────────
st.markdown('<div class="title">Advanced TTS Studio</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94A3B8'>Multi-accent speech with tone presets & 3-D character</p>", unsafe_allow_html=True)

left,right = st.columns((2,1), gap="large")

# 3-D avatar pane
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### AI Character")
    st_lottie(lottie("https://assets8.lottiefiles.com/packages/lf20_1pxqjqps.json"),
              height=380, key="avatar")
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    # Upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload File")
    upload = st.file_uploader("TXT · DOCX · PDF (≤ 10 MB)", type=["txt","docx","pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Text content
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Text Content")
    if "text" not in st.session_state: st.session_state.text=""
    if upload and st.session_state.get("loaded")!=upload.name:
        st.session_state.text = extract_text(upload, pathlib.Path(upload.name).suffix.lower())
        st.session_state.loaded = upload.name
    txt = st.text_area("", value=st.session_state.text, height=160)
    st.session_state.text = txt
    st.markdown(f"<small style='color:#64748B'>Chars: {len(txt):,}&nbsp;Words: {len(txt.split()):,}</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Audio settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Audio Settings")
    voices = load_voice_catalog()
    langs = sorted({v['Locale'] for v in voices})
    genders = sorted({v['Gender'] for v in voices})
    col1,col2 = st.columns(2)
    lang_filter   = col1.selectbox("Language / Accent", ["All"]+langs)
    gender_filter = col2.selectbox("Gender", ["All"]+genders)
    filtered = [v for v in voices if (lang_filter=="All" or v['Locale']==lang_filter)
                                   and (gender_filter=="All" or v['Gender']==gender_filter)]
    voice_name = st.selectbox("Voice", [v['ShortName'] for v in filtered],
                              format_func=lambda s: next(v['FriendlyName'] for v in voices if v['ShortName']==s))
    tone = st.selectbox("Tone Preset", list(TONE_PRESETS.keys()))
    add_rate = st.slider("Extra Rate (%)", -50, 50, 0, step=5)
    add_pitch= st.slider("Extra Pitch (Hz)", -20, 20, 0, step=2)
    out_fmt = st.selectbox("Output Format", ["MP3 (default)", "WAV (lossless)"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Generate
    if st.button("Generate Audio"):
        if not txt.strip():
            st.warning("Enter or upload some text.")
        else:
            base = TONE_PRESETS[tone]
            rate = f"{int(base['rate'][:-1])+add_rate:+d}%"
            pitch= f"{int(base['pitch'][:-2])+add_pitch:+d}Hz"
            cleaned = clean_text(txt)
            with st.spinner("Synthesising via edge-tts…"):
                mp3_bytes = asyncio.run(edge_tts_async(cleaned, voice_name, rate, pitch))
            if out_fmt.startswith("WAV"):
                wav_bytes = wav_from_mp3(mp3_bytes)
                audio_data, mime, ext = wav_bytes, "audio/wav", "wav"
            else:
                audio_data, mime, ext = mp3_bytes, "audio/mpeg", "mp3"

            st.success("Audio ready!")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Generated Audio")
            # Waveform
            try:
                wav_raw = wav_from_mp3(mp3_bytes) if mime=="audio/mpeg" else audio_data
                st.bar_chart(waveform_data(wav_raw), use_container_width=True)
            except Exception:
                st.info("Waveform preview unavailable.")
            st.audio(audio_data, format=mime)
            st.markdown(download_tag(audio_data, f"speech.{ext}", mime), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div style='text-align:center;padding:1rem 0;color:#64748B'>© 2025 Advanced TTS Studio · edge-tts backend</div>",
    unsafe_allow_html=True,
)
    

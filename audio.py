# ────────────────────────────────────────────────────────
#  Advanced TTS Studio  –  edge-tts backend (Cloud-ready)
#  Dark UI · multi-accent voices · tone presets
# ────────────────────────────────────────────────────────
import re, io, base64, tempfile, os, pathlib, subprocess, wave, json, asyncio
from typing import List, Dict

import streamlit as st
import numpy as np
import fitz                   # PyMuPDF
from docx import Document
from streamlit_lottie import st_lottie
import edge_tts
import requests

# ───────────────  Look & Feel  ───────────────
st.set_page_config(page_title="Advanced TTS Studio", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
        html,body,[class*="st-"]{background:#0B1120;color:#E5E7EB;font-family:Inter,sans-serif}
        .title{font-size:3rem;font-weight:800;color:#C084FC;text-align:center}
        .card{background:#111827;border:1px solid #1F2937;border-radius:12px;padding:1.25rem 1.6rem;margin-bottom:1.3rem}
        .stButton>button{background:linear-gradient(90deg,#7C3AED 0%,#EC4899 100%);color:#fff;border:0;border-radius:8px;padding:.55rem 1.3rem;font-weight:600}
        input[type=range]{accent-color:#C084FC}
        footer{visibility:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────  Globals ───────────────
ALLOWED_RE = re.compile(r"[A-Za-z0-9\s()]+")

TONE_PRESETS = {
    "Neutral (Default)": {"rate":"+0%",  "pitch":"+0Hz"},
    "Deep Male"        : {"rate":"-5%",  "pitch":"-4Hz"},
    "Bright Female"    : {"rate":"+5%",  "pitch":"+6Hz"},
    "Robot"            : {"rate":"+20%", "pitch":"-10Hz"},
    "Story Narrator"   : {"rate":"-10%", "pitch":"-2Hz"},
}

# ───────────────  Helpers ───────────────
@st.cache_data(show_spinner="Fetching voices…")
def load_voice_catalog() -> List[Dict]:
    """Return voice catalogue, each item with guaranteed keys: ShortName, Locale, Gender, friendly."""
    raw = asyncio.run(edge_tts.list_voices())
    fixed = []
    for v in raw:
        friendly = v.get("FriendlyName") or v.get("DisplayName") or v["ShortName"]
        fixed.append({
            "ShortName": v["ShortName"],
            "Locale"   : v["Locale"],
            "Gender"   : v["Gender"],
            "friendly" : friendly.replace("Microsoft ","").replace("Online ",""),
        })
    # de-duplicate on ShortName (rare duplicates happen)
    uniq = {v["ShortName"]: v for v in fixed}
    return list(uniq.values())

def lottie(url:str)->Dict:
    try:
        r=requests.get(url,timeout=6)
        if r.status_code==200:
            return r.json()
    except Exception:
        pass
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
    return re.sub(r"\((.*?)\)", lambda m:m.group(1), "".join(ALLOWED_RE.findall(raw)))

async def tts_async(text:str, voice:str, rate:str, pitch:str) -> bytes:
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); tmp.close()
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(tmp.name)
    data=pathlib.Path(tmp.name).read_bytes(); os.remove(tmp.name); return data

def mp3_to_wav(mp3:bytes)->bytes:
    mp3f=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); mp3f.write(mp3); mp3f.close()
    wavf=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); wavf.close()
    subprocess.run(["ffmpeg","-loglevel","quiet","-y","-i",mp3f.name,wavf.name])
    data=pathlib.Path(wavf.name).read_bytes(); os.remove(mp3f.name); os.remove(wavf.name); return data

def waveform(wav:bytes, buckets=90)->List[float]:
    with wave.open(io.BytesIO(wav)) as wf:
        samples=np.frombuffer(wf.readframes(wf.getnframes()),dtype=np.int16)
    samples=np.abs(samples); step=len(samples)//buckets+1
    levels=[samples[i:i+step].mean() for i in range(0,len(samples),step)]
    mx=max(levels) or 1
    return list((np.array(levels)/mx).astype(float))

def dl_tag(data:bytes, fname:str, mime:str)->str:
    b64=base64.b64encode(data).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{fname}">Download</a>'

# ───────────────  Title ───────────────
st.markdown('<div class="title">Advanced TTS Studio</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94A3B8'>Multi-accent speech · gender filter · tone presets · 3-D character</p>", unsafe_allow_html=True)

# ───────────────  Layout ───────────────
left, right = st.columns((2,1), gap="large")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### AI Character")
    st_lottie(lottie("https://assets8.lottiefiles.com/packages/lf20_1pxqjqps.json"), height=380)
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    # Upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload File")
    uploaded = st.file_uploader("TXT · DOCX · PDF", type=["txt","docx","pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Text
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Text Content")
    if "txt" not in st.session_state: st.session_state.txt = ""
    if uploaded and st.session_state.get("last_file") != uploaded.name:
        st.session_state.txt = extract_text(uploaded, pathlib.Path(uploaded.name).suffix.lower())
        st.session_state.last_file = uploaded.name
    txt = st.text_area("", value=st.session_state.txt, height=160)
    st.session_state.txt = txt
    st.markdown(f"<small style='color:#64748B'>Chars {len(txt):,} • Words {len(txt.split()):,}</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Audio settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Audio Settings")
    voices = load_voice_catalog()
    langs   = sorted({v["Locale"] for v in voices})
    genders = sorted({v["Gender"] for v in voices})
    c1, c2 = st.columns(2)
    f_lang   = c1.selectbox("Language", ["All"] + langs)
    f_gender = c2.selectbox("Gender",   ["All"] + genders)

    filtered = [v for v in voices if (f_lang=="All" or v["Locale"]==f_lang)
                                   and (f_gender=="All" or v["Gender"]==f_gender)]
    if not filtered: filtered = voices   # never let list be empty

    voice_options = [v["ShortName"] for v in filtered]
    def _pretty(short):   # safe pretty-printer
        v = next((x for x in voices if x["ShortName"] == short), None)
        return v["friendly"] if v else short

    voice_sel = st.selectbox("Voice", voice_options, index=0, format_func=_pretty)

    tone = st.selectbox("Tone Preset", list(TONE_PRESETS.keys()))
    add_rate  = st.slider("Extra Rate (%)",  -50, 50, 0, 5)
    add_pitch = st.slider("Extra Pitch (Hz)", -20, 20, 0, 2)
    out_fmt   = st.selectbox("Output Format", ["MP3 (default)", "WAV (lossless)"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Generate
    if st.button("Generate Audio"):
        if not txt.strip():
            st.warning("Enter or upload text first.")
        else:
            base = TONE_PRESETS[tone]
            rate  = f"{int(base['rate'][:-1])+add_rate:+d}%"
            pitch = f"{int(base['pitch'][:-2])+add_pitch:+d}Hz"
            cleaned = clean_text(txt)
            with st.spinner("Synthesising via edge-tts…"):
                mp3 = asyncio.run(tts_async(cleaned, voice_sel, rate, pitch))
            if out_fmt.startswith("WAV"):
                wav = mp3_to_wav(mp3)
                audio, mime, ext = wav, "audio/wav", "wav"
            else:
                audio, mime, ext = mp3, "audio/mpeg", "mp3"

            st.success("Audio ready!")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Generated Audio")
            try:
                wav_raw = mp3_to_wav(mp3) if mime=="audio/mpeg" else audio
                st.bar_chart(waveform(wav_raw), use_container_width=True)
            except Exception:
                st.info("Waveform unavailable.")
            st.audio(audio, format=mime)
            st.markdown(dl_tag(audio, f"speech.{ext}", mime), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div style='text-align:center;padding:1rem 0;color:#64748B'>© 2025 Advanced TTS Studio · edge-tts backend</div>",
    unsafe_allow_html=True,
)

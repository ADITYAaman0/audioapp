# ────────────────────────────────────────────────────────
#  Advanced TTS Studio with Self-Correcting RAG Pipeline
#  Dark UI · multi-accent voices · tone presets · Interactive AI Character · Document Analysis
# ────────────────────────────────────────────────────────
import re, io, base64, tempfile, os, pathlib, subprocess, wave, json, asyncio
from typing import List, Dict
import random
import time

import streamlit as st
import numpy as np
import fitz                   # PyMuPDF
from docx import Document
from streamlit_lottie import st_lottie
import edge_tts
import requests

# LangChain and RAG imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ───────────────  Look & Feel  ───────────────
st.set_page_config(page_title="Advanced TTS Studio with RAG", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
        html,body,[class*="st-"]{background:#0B1120;color:#E5E7EB;font-family:Inter,sans-serif}
        .title{font-size:3rem;font-weight:800;color:#C084FC;text-align:center}
        .card{background:#111827;border:1px solid #1F2937;border-radius:12px;padding:1.25rem 1.6rem;margin-bottom:1.3rem}
        .stButton>button{background:linear-gradient(90deg,#7C3AED 0%,#EC4899 100%);color:#fff;border:0;border-radius:8px;padding:.55rem 1.3rem;font-weight:600}
        input[type=range]{accent-color:#C084FC}
        footer{visibility:hidden}
        
        /* AI Character Speech Bubble */
        .speech-bubble {
            position: relative;
            background: #1F2937;
            border-radius: 20px;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid #374151;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.5s ease-in;
        }
        .speech-bubble:after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50px;
            border-width: 10px 10px 0;
            border-style: solid;
            border-color: #1F2937 transparent;
        }
        .character-name {
            color: #C084FC;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .character-message {
            color: #E5E7EB;
            font-size: 1rem;
            line-height: 1.4;
        }
        
        /* RAG Pipeline Status */
        .rag-status {
            background: #1F2937;
            border-left: 4px solid #10B981;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
        }
        .rag-status.warning {
            border-left-color: #F59E0B;
        }
        .rag-status.error {
            border-left-color: #EF4444;
        }
        
        /* Score Badge */
        .score-badge {
            display: inline-block;
            background: linear-gradient(90deg, #10B981 0%, #059669 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
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

CHARACTER_GREETINGS = [
    "Hello there! I'm your AI assistant, ready to bring your words to life with beautiful speech synthesis! 🎵",
    "Hey! Welcome to the TTS Studio with RAG! I can help you analyze documents and create amazing audio! ✨",
    "Greetings! I can transform text into speech AND intelligently analyze your documents! 🎧",
    "Hi there! Your personal voice studio with document intelligence is now active! 🎶",
    "Welcome! I'm thrilled to assist with speech synthesis and intelligent document analysis! 🌟"
]

CHARACTER_TIPS = [
    "💡 **Pro Tip:** Try the new RAG analysis to extract key insights from your documents!",
    "🎭 **Character Tip:** The 'Story Narrator' preset works great for audiobooks and long-form content.",
    "⚡ **Quick Tip:** You can analyze documents on any topic before converting to speech!",
    "🌍 **Global Tip:** RAG pipeline ensures factually accurate summaries from your documents!",
    "📊 **Format Tip:** Upload any document and query specific topics with intelligent filtering!"
]

# OpenAI API Key - get from environment or Streamlit secrets
OPENAI_API_KEY ="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"
# ───────────────  RAG Pipeline Classes  ───────────────

class RelevanceAgent:
    """Filters document content based on topic relevance"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance filtering expert. Evaluate if document chunks are relevant to the given topic.
            
Respond with JSON:
{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "reasoning": "brief explanation"
}"""),
            ("user", "Topic: {topic}\n\nDocument Chunk:\n{chunk}")
        ])
    
    def filter_chunks(self, chunks: List[str], topic: str, threshold: float = 0.6) -> List[Dict]:
        """Filter chunks based on relevance"""
        relevant_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                response = self.llm.invoke(
                    self.prompt.format_messages(topic=topic, chunk=chunk)
                )
                result = json.loads(response.content)
                
                if result["is_relevant"] and result["relevance_score"] >= threshold:
                    relevant_chunks.append({
                        "chunk": chunk,
                        "index": i,
                        "relevance_score": result["relevance_score"],
                        "reasoning": result["reasoning"]
                    })
            except Exception as e:
                continue
        
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_chunks


class GeneratorAgent:
    """Generates detailed summary answers from relevant content"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content synthesizer. Create comprehensive summaries from relevant chunks.

Format response as JSON:
{
    "summary": "detailed summary with citations",
    "key_points": ["point 1", "point 2"],
    "confidence": 0.0-1.0
}"""),
            ("user", "Topic: {topic}\n\nRelevant Chunks:\n{chunks}\n\nGenerate a detailed summary.")
        ])
    
    def generate(self, relevant_chunks: List[Dict], topic: str) -> Dict:
        """Generate summary from relevant chunks"""
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}] (Relevance: {chunk['relevance_score']:.2f})\n{chunk['chunk']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(topic=topic, chunks=chunks_text)
            )
            result = json.loads(response.content)
            result["source_chunks"] = relevant_chunks
            return result
        except Exception as e:
            return {
                "summary": f"Error generating summary: {e}",
                "key_points": [],
                "confidence": 0.0,
                "source_chunks": relevant_chunks
            }


class FactCheckAgent:
    """Verifies factual consistency against source documents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking expert. Verify summary accuracy against sources.

Respond with JSON:
{
    "factual_consistency_score": 0.0-1.0,
    "verified_claims": ["claim 1"],
    "unsupported_claims": ["claim 1"],
    "contradictions": ["contradiction 1"],
    "overall_assessment": "detailed assessment",
    "recommendation": "ACCEPT/REVISE/REJECT"
}"""),
            ("user", "Source Chunks:\n{source_chunks}\n\nSummary:\n{summary}\n\nVerify factual consistency.")
        ])
    
    def fact_check(self, summary: str, source_chunks: List[Dict]) -> Dict:
        """Fact-check summary against sources"""
        chunks_text = "\n\n".join([
            f"[Source {i+1}]\n{chunk['chunk']}"
            for i, chunk in enumerate(source_chunks)
        ])
        
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    source_chunks=chunks_text,
                    summary=summary
                )
            )
            return json.loads(response.content)
        except Exception as e:
            return {
                "factual_consistency_score": 0.0,
                "verified_claims": [],
                "unsupported_claims": [],
                "contradictions": [],
                "overall_assessment": f"Error: {e}",
                "recommendation": "REJECT"
            }


class SelfCorrectingRAGPipeline:
    """Complete RAG pipeline with three agents"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", max_iterations: int = 3):
        self.llm = ChatOpenAI(temperature=0, model=model, api_key=api_key)
        self.relevance_agent = RelevanceAgent(self.llm)
        self.generator_agent = GeneratorAgent(self.llm)
        self.fact_check_agent = FactCheckAgent(self.llm)
        self.max_iterations = max_iterations
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_document(self, document_text: str, topic: str, 
                        relevance_threshold: float = 0.6,
                        fact_check_threshold: float = 0.8) -> Dict:
        """Process document through RAG pipeline"""
        chunks = self.text_splitter.split_text(document_text)
        
        pipeline_results = {
            "topic": topic,
            "total_chunks": len(chunks),
            "iterations": []
        }
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # Step 1: Relevance Filtering
            relevant_chunks = self.relevance_agent.filter_chunks(
                chunks, topic, relevance_threshold
            )
            
            if not relevant_chunks:
                pipeline_results["iterations"].append({
                    "iteration": iteration,
                    "status": "FAILED",
                    "reason": "No relevant chunks found"
                })
                break
            
            # Step 2: Generate Summary
            generation_result = self.generator_agent.generate(relevant_chunks, topic)
            summary = generation_result["summary"]
            
            # Step 3: Fact-Check
            fact_check_result = self.fact_check_agent.fact_check(
                summary, relevant_chunks
            )
            consistency_score = fact_check_result["factual_consistency_score"]
            
            iteration_data = {
                "iteration": iteration,
                "relevant_chunks_count": len(relevant_chunks),
                "generation": generation_result,
                "fact_check": fact_check_result,
                "consistency_score": consistency_score
            }
            
            # Check acceptance
            if fact_check_result["recommendation"] == "ACCEPT" and \
               consistency_score >= fact_check_threshold:
                iteration_data["status"] = "SUCCESS"
                pipeline_results["iterations"].append(iteration_data)
                pipeline_results["final_summary"] = summary
                pipeline_results["final_score"] = consistency_score
                pipeline_results["status"] = "SUCCESS"
                break
            
            elif fact_check_result["recommendation"] == "REJECT":
                iteration_data["status"] = "REJECTED"
                pipeline_results["iterations"].append(iteration_data)
                relevance_threshold = max(0.5, relevance_threshold - 0.1)
            
            else:
                iteration_data["status"] = "REVISE"
                pipeline_results["iterations"].append(iteration_data)
                relevance_threshold = max(0.5, relevance_threshold - 0.05)
        
        if "status" not in pipeline_results:
            pipeline_results["status"] = "MAX_ITERATIONS_REACHED"
            pipeline_results["final_summary"] = summary if 'summary' in locals() else None
            pipeline_results["final_score"] = consistency_score if 'consistency_score' in locals() else 0.0
        
        return pipeline_results


# ───────────────  Helpers ───────────────
@st.cache_data(show_spinner="Fetching voices…")
def load_voice_catalog() -> List[Dict]:
    """Return voice catalogue"""
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

def get_character_message():
    """Get a random character message with occasional tips"""
    if random.random() < 0.3:
        return random.choice(CHARACTER_TIPS)
    return random.choice(CHARACTER_GREETINGS)

# Initialize RAG Pipeline
@st.cache_resource
def get_rag_pipeline():
    """Initialize and cache the RAG pipeline"""
    if not OPENAI_API_KEY:
        return None
    try:
        return SelfCorrectingRAGPipeline(api_key=OPENAI_API_KEY, model="gpt-4")
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

# ───────────────  Title ───────────────
st.markdown('<div class="title">Advanced TTS Studio with RAG</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94A3B8'>Multi-accent speech · gender filter · tone presets · Intelligent Document Analysis · Self-Correcting RAG Pipeline</p>", unsafe_allow_html=True)

# ───────────────  Tabs ───────────────
tab1, tab2 = st.tabs(["🎙️ Text-to-Speech", "🧠 Document Analysis (RAG)"])

# ═══════════════════════════════════════════════════════════
#  TAB 1: TEXT-TO-SPEECH
# ═══════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns((2,1), gap="large")

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎭 AI Assistant")
        
        if "character_message" not in st.session_state:
            st.session_state.character_message = get_character_message()
        
        st.markdown(
            f"""
            <div class="speech-bubble">
                <div class="character-name">TTS Assistant</div>
                <div class="character-message">
                    {st.session_state.character_message}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st_lottie(lottie("https://assets8.lottiefiles.com/packages/lf20_1pxqjqps.json"), height=280)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Tip", use_container_width=True, key="tip1"):
                st.session_state.character_message = get_character_message()
                st.rerun()
        
        with col2:
            if st.button("👋 Say Hello", use_container_width=True, key="hello1"):
                st.session_state.character_message = "Hello! Thanks for stopping by! I'm excited to help you create amazing audio content today! 🎉"
                st.rerun()
        
        st.markdown("---")
        st.markdown("#### 🎯 Assistant Status")
        
        if "txt" in st.session_state and st.session_state.txt.strip():
            text_length = len(st.session_state.txt)
            if text_length < 100:
                status = "🟢 Ready for short text"
            elif text_length < 500:
                status = "🔵 Perfect for paragraphs"
            else:
                status = "🟡 Great for long content!"
        else:
            status = "⚪ Waiting for your text..."
        
        st.markdown(f"**Status:** {status}")
        
        voices = load_voice_catalog()
        st.markdown(f"**Available Voices:** {len(voices):,}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        # Upload
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload File")
        uploaded = st.file_uploader("TXT · DOCX · PDF", type=["txt","docx","pdf"], label_visibility="collapsed", key="tts_upload")
        st.markdown("</div>", unsafe_allow_html=True)

        # Text
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Text Content")
        if "txt" not in st.session_state: st.session_state.txt = ""
        if uploaded and st.session_state.get("last_file") != uploaded.name:
            st.session_state.txt = extract_text(uploaded, pathlib.Path(uploaded.name).suffix.lower())
            st.session_state.last_file = uploaded.name
            st.session_state.character_message = "Great! I see you've uploaded a file. Ready to convert it to speech? 🎙️"
            st.rerun()
        
        txt = st.text_area("", value=st.session_state.txt, height=160, placeholder="Type or paste your text here... Or upload a file above!", key="tts_text")
        st.session_state.txt = txt
        
        char_count = len(txt)
        word_count = len(txt.split())
        
        if word_count > 0:
            reading_time = max(1, word_count // 200)
            stats_text = f"📊 **Stats:** {char_count:,} chars • {word_count:,} words • ⏱️ ~{reading_time} min read"
        else:
            stats_text = f"📊 **Stats:** {char_count:,} chars • {word_count:,} words"
        
        st.markdown(f"<small style='color:#64748B'>{stats_text}</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Audio settings
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎚️ Audio Settings")
        voices = load_voice_catalog()
        langs   = sorted({v["Locale"] for v in voices})
        genders = sorted({v["Gender"] for v in voices})
        c1, c2 = st.columns(2)
        f_lang   = c1.selectbox("Language", ["All"] + langs, key="lang1")
        f_gender = c2.selectbox("Gender",   ["All"] + genders, key="gender1")

        filtered = [v for v in voices if (f_lang=="All" or v["Locale"]==f_lang)
                                       and (f_gender=="All" or v["Gender"]==f_gender)]
        if not filtered: filtered = voices

        voice_options = [v["ShortName"] for v in filtered]
        def _pretty(short):
            v = next((x for x in voices if x["ShortName"] == short), None)
            return v["friendly"] if v else short

        voice_sel = st.selectbox("Voice", voice_options, index=0, format_func=_pretty, key="voice1")

        tone = st.selectbox("Tone Preset", list(TONE_PRESETS.keys()), key="tone1")
        add_rate  = st.slider("Extra Rate (%)",  -50, 50, 0, 5, key="rate1")
        add_pitch = st.slider("Extra Pitch (Hz)", -20, 20, 0, 2, key="pitch1")
        out_fmt   = st.selectbox("Output Format", ["MP3 (default)", "WAV (lossless)"], key="format1")
        st.markdown("</div>", unsafe_allow_html=True)

        # Generate
        if st.button("🎵 Generate Audio", use_container_width=True, key="gen1"):
            if not txt.strip():
                st.warning("Please enter or upload some text first.")
                st.session_state.character_message = "I notice you haven't entered any text yet. Could you please add some text to convert? 📝"
                st.rerun()
            else:
                st.session_state.character_message = "I'm working on your audio now! This will just take a moment... ⚡"
                st.rerun()
                
                base = TONE_PRESETS[tone]
                rate  = f"{int(base['rate'][:-1])+add_rate:+d}%"
                pitch = f"{int(base['pitch'][:-2])+add_pitch:+d}Hz"
                cleaned = clean_text(txt)
                with st.spinner("🎙️ Synthesising audio with edge-tts..."):
                    mp3 = asyncio.run(tts_async(cleaned, voice_sel, rate, pitch))
                if out_fmt.startswith("WAV"):
                    wav = mp3_to_wav(mp3)
                    audio, mime, ext = wav, "audio/wav", "wav"
                else:
                    audio, mime, ext = mp3, "audio/mpeg", "mp3"

                st.success("✅ Audio ready! Listen to your creation below.")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🎧 Generated Audio")
                try:
                    wav_raw = mp3_to_wav(mp3) if mime=="audio/mpeg" else audio
                    st.bar_chart(waveform(wav_raw), use_container_width=True)
                except Exception:
                    st.info("Waveform visualization unavailable.")
                st.audio(audio, format=mime)
                st.markdown(dl_tag(audio, f"speech.{ext}", mime), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.session_state.character_message = "Perfect! Your audio is ready. How does it sound? Feel free to adjust settings and try again! 🎶"
                st.rerun()

# ═══════════════════════════════════════════════════════════
#  TAB 2: DOCUMENT ANALYSIS (RAG)
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🧠 Intelligent Document Analysis")
    st.markdown("Upload a document and query specific topics using our self-correcting RAG pipeline with three AI agents.")
    
    # Check if RAG is available
    rag_pipeline = get_rag_pipeline()
    
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API Key not found. Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
        st.info("💡 You can still use the Text-to-Speech features without an API key!")
    elif not rag_pipeline:
        st.error("Failed to initialize RAG pipeline. Please check your API key and internet connection.")
    else:
        left_rag, right_rag = st.columns((3,2), gap="large")
        
        with left_rag:
            # Document Upload
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📄 Upload Document")
            doc_upload = st.file_uploader(
                "Upload a document to analyze", 
                type=["txt","docx","pdf"], 
                key="rag_upload",
                help="Supported formats: TXT, DOCX, PDF"
            )
            
            if doc_upload:
                doc_text = extract_text(doc_upload, pathlib.Path(doc_upload.name).suffix.lower())
                st.session_state.doc_text = doc_text
                st.session_state.doc_name = doc_upload.name
                
                # Show document stats
                doc_chars = len(doc_text)
                doc_words = len(doc_text.split())
                st.markdown(f"""
                    <div class="rag-status">
                        ✅ <strong>Document loaded:</strong> {doc_upload.name}<br>
                        📊 {doc_chars:,} characters · {doc_words:,} words
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Query Input
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Query Topic")
            query_topic = st.text_input(
                "What would you like to know about this document?",
                placeholder="e.g., 'key findings and methodology' or 'action items and decisions'",
                key="rag_query",
                help="Be specific about what you want to extract from the document"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Advanced Settings
            with st.expander("⚙️ Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    relevance_threshold = st.slider(
                        "Relevance Threshold",
                        0.5, 1.0, 0.6, 0.05,
                        help="Higher = stricter filtering (only very relevant content)"
                    )
                with col2:
                    fact_check_threshold = st.slider(
                        "Fact-Check Threshold",
                        0.6, 1.0, 0.8, 0.05,
                        help="Minimum score to accept summary"
                    )
                
                st.markdown("**Pipeline Info:**")
                st.markdown("- 🔍 **Relevance Agent**: Filters chunks by topic relevance")
                st.markdown("- ✍️ **Generator Agent**: Creates detailed summaries")
                st.markdown("- ✅ **Fact-Check Agent**: Verifies accuracy against source")
            
            # Analyze Button
            analyze_btn = st.button("🚀 Analyze Document", use_container_width=True, type="primary", key="analyze_btn")
            
            # Process Analysis
            if analyze_btn:
                if "doc_text" not in st.session_state or not st.session_state.doc_text:
                    st.warning("⚠️ Please upload a document first!")
                elif not query_topic.strip():
                    st.warning("⚠️ Please enter a query topic!")
                else:
                    with st.spinner("🔄 Processing through RAG pipeline..."):
                        try:
                            # Run RAG pipeline
                            results = rag_pipeline.process_document(
                                document_text=st.session_state.doc_text,
                                topic=query_topic,
                                relevance_threshold=relevance_threshold,
                                fact_check_threshold=fact_check_threshold
                            )
                            
                            st.session_state.rag_results = results
                            st.success("✅ Analysis complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.session_state.rag_results = None
            
            # Display Results
            if "rag_results" in st.session_state and st.session_state.rag_results:
                results = st.session_state.rag_results
                
                st.markdown("---")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📊 Analysis Results")
                
                # Status and Score
                status = results.get("status", "UNKNOWN")
                final_score = results.get("final_score", 0.0)
                
                if status == "SUCCESS":
                    status_class = ""
                    status_icon = "✅"
                elif status == "MAX_ITERATIONS_REACHED":
                    status_class = "warning"
                    status_icon = "⚠️"
                else:
                    status_class = "error"
                    status_icon = "❌"
                
                st.markdown(f"""
                    <div class="rag-status {status_class}">
                        {status_icon} <strong>Status:</strong> {status}<br>
                        <span class="score-badge">Score: {final_score:.2%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Final Summary
                if results.get("final_summary"):
                    st.markdown("#### 📝 Summary")
                    st.markdown(results["final_summary"])
                    
                    # Option to convert summary to speech
                    if st.button("🎙️ Convert Summary to Speech", key="summary_to_speech"):
                        st.session_state.txt = results["final_summary"]
                        st.session_state.character_message = "I've loaded the RAG summary into the Text-to-Speech tab! Switch over to generate audio! 🎵"
                        st.success("✅ Summary loaded! Switch to Text-to-Speech tab to generate audio.")
                
                # Key Points
                if results.get("iterations") and results["iterations"]:
                    last_iteration = results["iterations"][-1]
                    if "generation" in last_iteration and "key_points" in last_iteration["generation"]:
                        key_points = last_iteration["generation"]["key_points"]
                        if key_points:
                            st.markdown("#### 🎯 Key Points")
                            for i, point in enumerate(key_points, 1):
                                st.markdown(f"{i}. {point}")
                
                # Iteration Details
                with st.expander("🔬 Pipeline Iterations Details"):
                    st.markdown(f"**Total Iterations:** {len(results['iterations'])}")
                    st.markdown(f"**Total Chunks:** {results['total_chunks']}")
                    
                    for iter_data in results["iterations"]:
                        st.markdown(f"---")
                        st.markdown(f"**Iteration {iter_data['iteration']}**")
                        st.markdown(f"- Status: {iter_data['status']}")
                        st.markdown(f"- Relevant Chunks: {iter_data.get('relevant_chunks_count', 'N/A')}")
                        
                        if "consistency_score" in iter_data:
                            st.markdown(f"- Consistency Score: {iter_data['consistency_score']:.2%}")
                        
                        if "fact_check" in iter_data:
                            fc = iter_data["fact_check"]
                            st.markdown(f"- Recommendation: {fc['recommendation']}")
                            st.markdown(f"- Verified Claims: {len(fc.get('verified_claims', []))}")
                            st.markdown(f"- Unsupported Claims: {len(fc.get('unsupported_claims', []))}")
                            
                            if fc.get('unsupported_claims'):
                                st.warning(f"⚠️ Unsupported: {', '.join(fc['unsupported_claims'][:3])}")
                
                # Fact-Check Details
                if results.get("iterations") and results["iterations"]:
                    last_iter = results["iterations"][-1]
                    if "fact_check" in last_iter:
                        with st.expander("✅ Fact-Check Report"):
                            fc = last_iter["fact_check"]
                            
                            st.markdown(f"**Overall Assessment:**")
                            st.markdown(fc.get("overall_assessment", "No assessment available"))
                            
                            if fc.get("verified_claims"):
                                st.markdown("**✅ Verified Claims:**")
                                for claim in fc["verified_claims"]:
                                    st.markdown(f"- {claim}")
                            
                            if fc.get("unsupported_claims"):
                                st.markdown("**⚠️ Unsupported Claims:**")
                                for claim in fc["unsupported_claims"]:
                                    st.markdown(f"- {claim}")
                            
                            if fc.get("contradictions"):
                                st.markdown("**❌ Contradictions:**")
                                for contradiction in fc["contradictions"]:
                                    st.markdown(f"- {contradiction}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with right_rag:
            # RAG Info Card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🤖 RAG Pipeline Info")
            
            st.markdown("""
                <div class="speech-bubble">
                    <div class="character-name">RAG Assistant</div>
                    <div class="character-message">
                        I use a three-agent system to ensure accurate, factually-consistent summaries from your documents! 🎯
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### How it works:")
            st.markdown("""
                1. **🔍 Relevance Agent**
                   - Filters document chunks by topic
                   - Scores each chunk (0.0-1.0)
                   - Keeps only relevant content
                
                2. **✍️ Generator Agent**
                   - Creates detailed summaries
                   - Synthesizes information
                   - Adds citations and confidence
                
                3. **✅ Fact-Check Agent**
                   - Verifies against sources
                   - Identifies unsupported claims
                   - Ensures accuracy
            """)
            
            st.markdown("#### 🎯 Use Cases:")
            st.markdown("""
                - 📚 **Research Papers**: Extract methodology & findings
                - 📝 **Meeting Notes**: Identify action items & decisions
                - 📄 **Legal Docs**: Find relevant clauses
                - 📊 **Reports**: Summarize key insights
                - 🎓 **Study Materials**: Generate focused summaries
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Example Queries
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 💡 Example Queries")
            
            example_queries = [
                "key findings and methodology",
                "action items and decisions",
                "main arguments and conclusions",
                "risks and mitigation strategies",
                "recommendations and next steps"
            ]
            
            st.markdown("**Try these queries:**")
            for query in example_queries:
                if st.button(f"📌 {query}", key=f"example_{query}", use_container_width=True):
                    st.session_state.example_query = query
                    st.rerun()
            
            # Pre-fill example query if selected
            if "example_query" in st.session_state:
                st.info(f"💡 Example query loaded: '{st.session_state.example_query}'. Paste it in the query field above!")
                del st.session_state.example_query
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Statistics
            if "rag_results" in st.session_state and st.session_state.rag_results:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📈 Analysis Statistics")
                
                results = st.session_state.rag_results
                
                # Create metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Iterations", len(results.get("iterations", [])))
                    st.metric("Total Chunks", results.get("total_chunks", 0))
                
                with col2:
                    if results.get("iterations"):
                        last_iter = results["iterations"][-1]
                        st.metric("Relevant Chunks", last_iter.get("relevant_chunks_count", 0))
                        st.metric("Accuracy Score", f"{results.get('final_score', 0):.1%}")
                
                st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;padding:1rem 0;color:#64748B'>© 2025 Advanced TTS Studio with RAG · edge-tts · LangChain · Interactive AI Assistant</div>",
    unsafe_allow_html=True,
)


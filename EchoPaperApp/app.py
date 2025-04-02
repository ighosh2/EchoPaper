import streamlit as st
import time
import random
import arxiv
import re
import os
import openai
import PyPDF2
import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
from openai import OpenAI
from io import BytesIO
import tempfile

# Initialize the OpenAI client with your API key
openAI_client = OpenAI(api_key='')


#################### SIMPLE TTS FUNCTIONS ####################

@st.cache_resource
def load_silero_model(language="en", speaker="v3_en", device="cpu"):
    """
    Load the Silero-TTS model.
    """
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        source="github",
        language=language,
        speaker=speaker,
        trust_repo=True
    )
    model.to(device)
    return model

model = load_silero_model()

def modify_audio_speed(audio, sample_rate, speed_factor=0.5):
    """
    Slow down the audio by resampling.
    """
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    
    new_sample_rate = int(sample_rate * speed_factor)
    resampler = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    audio = resampler(audio)
    return audio

def text_to_speech_simple(text, model, sample_rate=24000, speed_factor=1.2, silence_duration=1.0):
    """
    Convert text to speech using a simple chunking method.
    
    Splits the text into chunks of up to 250 characters, applies TTS on each chunk,
    slows down the audio as requested, adds a short silence at the end, and finally 
    concatenates the audio chunks.
    
    Returns:
        full_audio (np.ndarray): The concatenated audio waveform.
        new_sample_rate (int): The updated sample rate after speed adjustment.
    """
    text= "Hello welcome to the EchoPaper App , developed by Indraajit.  " + text
    print("The updated text"+text)
    max_length = 250  # maximum characters per chunk
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    full_audio = []  # List to collect processed audio chunks

    for chunk in chunks:
        with torch.no_grad():
            # Generate audio for the chunk
            audio = model.apply_tts(text=chunk, sample_rate=sample_rate, speaker='en_30')
            # Slow down the audio
            audio = modify_audio_speed(audio, sample_rate, speed_factor)
        
        # Convert to NumPy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        full_audio.append(audio)

    # Append a short silence to ensure complete playback
    effective_silence_samples = int(sample_rate * speed_factor * silence_duration)
    silence = np.zeros(effective_silence_samples, dtype=full_audio[0].dtype)
    full_audio.append(silence)

    # Concatenate all audio chunks
    full_audio = np.concatenate(full_audio)
    new_sample_rate = int(sample_rate * speed_factor)
    return full_audio, new_sample_rate

#################### END SIMPLE TTS FUNCTIONS ####################


#################### OTHER HELPER FUNCTIONS ####################

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDF2."""
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_paragraphs(text, max_tokens=1000, overlap=200):
    
    paragraphs = text.split("\n\n")
    all_chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        words = paragraph.split()
        if len(words) <= max_tokens:
            all_chunks.append(paragraph)
        else:
            start = 0
            while start < len(words):
                end = start + max_tokens
                chunk_words = words[start:end]
                all_chunks.append(" ".join(chunk_words))
                start += (max_tokens - overlap)
                if start < 0:
                    start = 0
    
    return all_chunks

def generate_podcast(pdf_path: str, paper_title: str) -> str:
    """
    Generate a podcast narration by:
      1. Extracting text from the PDF.
      2. Chunking it into manageable pieces.
      3. Summarizing each chunk via OpenAI.
      4. Combining the summaries into a final narration.
    """
    full_text = extract_pdf_text(pdf_path)
    if not full_text.strip():
        return f"❌ Could not extract text from {paper_title}. (Is it a scanned PDF?)"

    chunks = chunk_paragraphs(full_text, max_tokens=1000, overlap=200)
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = (
            f"Summarize the following text (chunk {i+1}/{len(chunks)}) in a concise, "
            f"conversational style:\n\n{chunk}"
        )
        summary = call_openai_api(prompt)
        partial_summaries.append(summary)

    combined_text = "\n".join(partial_summaries)

    prompt_final = (
        "Combine the following chunk-wise summaries into one cohesive, detailed summary. "
        "Make it sound like an engaging podcast narration,do not use any music:\n\n"
        "Completed the whole narration within 1000 words:\n\n" + combined_text
    )
    final_summary = call_openai_api(prompt_final)

    return final_summary

def call_openai_api(prompt, model="gpt-4o-mini") -> str:
    """
    Call OpenAI’s chat API to generate a summary.
    """
    try:
        response = openAI_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI API error: {e}"

def fetch_arxiv_results(query, max_results=5):
    """
    Fetch search results from arXiv.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": ", ".join([author.name for author in result.authors]),
            "link": result.entry_id,
            "arxiv_obj": result  # store the entire result object
        })
    return results

def sanitize_filename(filename: str) -> str:
    """
    Replace invalid filename characters with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# Create a folder to store downloaded PDFs if it doesn't exist
os.makedirs("downloaded_pdfs", exist_ok=True)

#################### END HELPER FUNCTIONS ####################

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #D93118;
        }
         /* Top-Right Corner Text */
        .top-right {
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        /* Center the text input field */
        .stTextInput > div {
            display: flex;
            justify-content: center;
        }
        /* Set width & left-align text inside the input */
        .stTextInput > div > div > input {
            width: 500px !important;
            text-align: left !important;
        }
        /* Style the search results */
        .result-box {
            border: 2px solid white;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Top-right developer credit and header
st.markdown("<div class='top-right'>-Developed by Indrajit Ghosh</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>EchoPaper</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your Personal AI-Powered Research Narrator</h4>", unsafe_allow_html=True)

# Session state for search query
if "query" not in st.session_state:
    st.session_state["query"] = ""

# Centered search input
query = st.text_input("Search your research paper:", value=st.session_state["query"], key="query")
refined_query = f'"{query}" AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)'

# Buttons for Search and Clear
col1, col2 = st.columns([7.5, 2])
if col1.button("🔍 Search"):
    if refined_query.strip():
        with st.spinner("Fetching results..."):
            st.session_state["results"] = fetch_arxiv_results(refined_query)
            st.session_state["refined_query"] = refined_query
    else:
        st.warning("Please enter a search query.")

def clear_search():
    st.session_state["query"] = ""

if col2.button("🗑️ Clear Results", on_click=clear_search):
    st.session_state.pop("results", None)
    st.session_state.pop("refined_query", None)
    st.success("Results cleared.")

# Display search results
if "results" in st.session_state and isinstance(st.session_state["results"], list):
    st.subheader("Search Results:")
    for paper in st.session_state["results"]:
        if isinstance(paper, dict):
            with st.container():
                col1, col2 = st.columns([0.9, 0.3])
                col1.markdown(
                    f"<div class='result-box'>📄 <b>{paper['title']}</b><br>"
                    f"<small>🖊️ {paper['authors']}</small><br>"
                    f"<a href='{paper['link']}' target='_blank'>🔗 Read Paper</a></div>",
                    unsafe_allow_html=True
                )
                if col2.button("🎙️ Generate Podcast", key=paper['title']):
                    # with st.spinner(f"Generating podcast for '{paper['title']}'..."):
                    with st.status(f"Generating podcast for '{paper['title']}'...", expanded=True) as status:
                        clean_title = sanitize_filename(paper["title"]) + ".pdf"
                        pdf_path = os.path.join("downloaded_pdfs", clean_title)
                        try:
                            
                                # Download the PDF from Arxiv Step 1
                                st.write("🟡 EchoPaper is downloading your research paper..")
                                paper["arxiv_obj"].download_pdf(
                                    dirpath="downloaded_pdfs", 
                                    filename=clean_title
                                )
                                # st.info(f"PDF saved to: {pdf_path}")
                                st.write("✅ Completed Download")
                                
                                st.write("🟡 EchoPaper is breaking the research paper and performing context analysis and augmentation...")
                                # Generate podcast narration text
                                podcast_text = generate_podcast(pdf_path, paper["title"])
                                st.write("✅ Completed Context Analysis & Text Augmentation!")

                                #print(podcast_text)
                                st.write("🟡 EchoPaper is creating custom speech from the refined context..")
                                # Convert narration text to speech using the simple TTS function
                                audio_data, new_sample_rate = text_to_speech_simple(podcast_text, model)
                                st.write("✅ Podcast Generation Complete!")
                                 #Status close bar
                                status.update(label="✅ All tasks completed!", state="complete", expanded=True)
                                # Save the audio to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                                    torchaudio.save(temp_audio_file.name, torch.tensor(audio_data).unsqueeze(0), new_sample_rate)
                                    audio_path = temp_audio_file.name

                                st.success("🎙️ Podcast Generated!")
                                st.audio(audio_path, format="audio/wav")

                                # Provide a download button for the podcast audio
                                with open(audio_path, "rb") as audio_file:
                                    st.download_button(
                                        label="⬇️ Download Podcast",
                                        data=audio_file,
                                        file_name="podcast_summary.wav",
                                        mime="audio/wav"
                                    )
                               
                                # Clean up temporary files
                                if os.path.exists(pdf_path):
                                    os.remove(pdf_path)
                                if os.path.exists(audio_path):
                                    os.remove(audio_path)

                        except Exception as e:
                            st.error(f"Failed to download PDF: {e}")
        else:
            st.error("Unexpected data format in search results.")

import tempfile
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from openai import OpenAI
import os
import io
from gtts import gTTS
from gtts.lang import tts_langs
import librosa
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import difflib

api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

# Function to check file size
def check_file_size(file):
    MAX_SIZE = 20 * 1024 * 1024  # 20 MB in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size <= MAX_SIZE

# Function to transcribe audio using OpenAI
def transcribe_audio(audio_file):    
    try:
        if not check_file_size(audio_file):
            return "Error: Audio file size exceeds 20 MB limit."
        
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file, 
            response_format="text"
        )
        return transcription
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

# Function to analyze audio tone and pitch
def analyze_audio(audio_file):
    try:
        # Save the uploaded file to a temporary location
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.getvalue())
        
        # Load the audio file using librosa
        y, sr = librosa.load("temp_audio.mp3")
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.median(pitches[magnitudes > 0])  # Consider all non-zero magnitudes
        # pitch = pitch if np.isfinite(pitch) else 0  # Handle potential NaN or inf values
        
        # Tone analysis (using spectral centroid as a proxy for "brightness")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        tone_brightness = np.mean(spectral_centroids)
        
        # Rhythm analysis (using tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Energy analysis
        energy = np.mean(librosa.feature.rms(y=y)[0]) ** 2
        
        # Normalize scores to a 0-100 scale
        pitch_score = min(100, max(0, (pitch - 50) / 1000 * 100)) 
        tone_score = min(100, max(0, tone_brightness / 5000 * 100))
        rhythm_score = min(100, max(0, tempo / 2))
        energy_score = min(100, max(0, energy * 1000))
        
        overall_score = (pitch_score + tone_score + rhythm_score + energy_score) / 4
        
        # Remove the temporary file
        os.remove("temp_audio.mp3")
        
        return overall_score, pitch_score, tone_score, rhythm_score, energy_score
    except Exception as e:
        return f"An error occurred during audio analysis: {str(e)}", None, None, None, None

from operator import xor
from typing import List

# These imports should be in your Python module path
# after installing the `pyacoustid` package from PyPI.
import acoustid
import chromaprint


def get_fingerprint(filename: str) -> List[int]:
    """
    Reads an audio file from the filesystem and returns a
    fingerprint.

    Args:
        filename: The filename of an audio file on the local
            filesystem to read.

    Returns:
        Returns a list of 32-bit integers. Two fingerprints can
        be roughly compared by counting the number of
        corresponding bits that are different from each other.
    """
    _, encoded = acoustid.fingerprint_file(filename)
    fingerprint, _ = chromaprint.decode_fingerprint(
        encoded
    )
    return fingerprint


def fingerprint_distance(
    f1: List[int],
    f2: List[int],
    fingerprint_len: int,
) -> float:
    """
    Returns a normalized distance between two fingerprints.

    Args:
        f1: The first fingerprint.

        f2: The second fingerprint.

        fingerprint_len: Only compare the first `fingerprint_len`
            integers in each fingerprint. This is useful
            when comparing audio samples of a different length.

    Returns:
        Returns a number between 0.0 and 1.0 representing
        the distance between two fingerprints. This value
        represents distance as like a percentage.
    """
    max_hamming_weight = 32 * fingerprint_len
    hamming_weight = sum(
        sum(
            c == "1"
            for c in bin(xor(f1[i], f2[i]))
        )
        for i in range(fingerprint_len)
    )
    return hamming_weight / max_hamming_weight

# Function to compare audio fingerprints
def compare_audio_fingerprints(ideal_audio, comparison_audio):
    try:
        # Save the uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_ideal:
            temp_ideal.write(ideal_audio.getvalue())
            temp_ideal_path = temp_ideal.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_comparison:
            temp_comparison.write(comparison_audio.getvalue())
            temp_comparison_path = temp_comparison.name

        # Get fingerprints
        ideal_fingerprint = get_fingerprint(temp_ideal_path)
        comparison_fingerprint = get_fingerprint(temp_comparison_path)

        # Calculate fingerprint distance
        fingerprint_len = min(len(ideal_fingerprint), len(comparison_fingerprint))
        distance = fingerprint_distance(ideal_fingerprint, comparison_fingerprint, fingerprint_len)

        # Remove temporary files
        os.unlink(temp_ideal_path)
        os.unlink(temp_comparison_path)

        return distance
    except Exception as e:
        return f"An error occurred during audio fingerprint comparison: {str(e)}"



# Function to compare texts and generate HTML with colored differences using OpenAI
def compare_texts(ideal_text, comparison_text):
    
    try:
        # Split texts into chunks
        chunk_size = 3000  # Adjust this value based on your needs
        ideal_chunks = [ideal_text[i:i+chunk_size] for i in range(0, len(ideal_text), chunk_size)]
        comparison_chunks = [comparison_text[i:i+chunk_size] for i in range(0, len(comparison_text), chunk_size)]
        
        comparisons = []
        for ideal_chunk, comparison_chunk in zip(ideal_chunks, comparison_chunks):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a text comparison assistant. Compare the following two text chunks, where the first is the ideal text and the second is the text to be compared. Highlight the differences and provide feedback on how the second text can be improved to match the ideal text."},
                    {"role": "user", "content": f"Ideal text chunk: {ideal_chunk}\nComparison text chunk: {comparison_chunk}"}
                ]
            )
            comparisons.append(response.choices[0].message.content)
        
        comparison = "\n".join(comparisons)
        
        # Calculate similarity using OpenAI's embedding
        embedding1 = client.embeddings.create(input=ideal_text, model="text-embedding-3-large").data[0].embedding
        embedding2 = client.embeddings.create(input=comparison_text, model="text-embedding-3-large").data[0].embedding
        
        # Calculate cosine similarity
        similarity = sum(a*b for a, b in zip(embedding1, embedding2))
        similarity = similarity / (sum(a*a for a in embedding1)**0.5 * sum(b*b for b in embedding2)**0.5)
        
        return comparison, similarity * 100
    except Exception as e:
        return f"An error occurred during comparison: {str(e)}", 0

# Function to format Arabic text
def format_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    formatted_text = get_display(reshaped_text)
    return formatted_text

# Function to convert text to speech
def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text
        )
        audio_bytes = io.BytesIO()
        for chunk in response.iter_bytes():
            audio_bytes.write(chunk)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        return f"An error occurred during text-to-speech conversion: {str(e)}"

# Function to compare audio scores
def compare_audio_scores(ideal_scores, comparison_scores):
    overall_diff = comparison_scores[0] - ideal_scores[0]
    pitch_diff = comparison_scores[1] - ideal_scores[1]
    tone_diff = comparison_scores[2] - ideal_scores[2]
    rhythm_diff = comparison_scores[3] - ideal_scores[3]
    energy_diff = comparison_scores[4] - ideal_scores[4]
    
    return overall_diff, pitch_diff, tone_diff, rhythm_diff, energy_diff

# Function to create a comparison chart
def create_comparison_chart(ideal_scores, comparison_scores):
    ideal_scores = [float(score) for score in ideal_scores]
    comparison_scores = [float(score) for score in comparison_scores]
    categories = ['Overall', 'Pitch', 'Tone', 'Rhythm', 'Energy']
    
    differences = [comparison - ideal for ideal, comparison in zip(ideal_scores, comparison_scores)]
    
    fig = go.Figure(data=[
        go.Bar(name='Difference', x=categories, y=differences,
               text=[f"{diff:.2f}" for diff in differences],
               textposition='outside')
    ])
    
    fig.update_layout(
        title='Audio Score Difference',
        xaxis_title='Categories',
        yaxis_title='Score Difference',
        height=500,
        width=700
    )
    
    # Ensure y-axis is centered at 0 and covers the range of differences
    max_abs_diff = max(abs(min(differences)), abs(max(differences)))
    fig.update_yaxes(range=[-max_abs_diff * 1.1, max_abs_diff * 1.1], zeroline=True)
    
    # Add a horizontal line at y=0
    fig.add_shape(type="line", x0=-0.5, y0=0, x1=len(categories)-0.5, y1=0,
                  line=dict(color="black", width=1, dash="dash"))
    
    return fig

# Function to create a Waveform comparison with plotly
def create_waveform_comparison(ideal_audio, comparison_audio):
    # Read audio files
    # Save the uploaded files to temporary locations
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_ideal:
        temp_ideal.write(ideal_audio.getvalue())
        temp_ideal_path = temp_ideal.name

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_comparison:
        temp_comparison.write(comparison_audio.getvalue())
        temp_comparison_path = temp_comparison.name

    # Load audio files
    ideal_y, ideal_sr = librosa.load(temp_ideal_path, sr=None)
    comparison_y, comparison_sr = librosa.load(temp_comparison_path, sr=None)
    
    # Create time arrays
    ideal_time = np.arange(0, len(ideal_y)) / ideal_sr
    comparison_time = np.arange(0, len(comparison_y)) / comparison_sr
    
    # Downsample the audio data to reduce size
    max_points = 10000
    if len(ideal_y) > max_points:
        step = len(ideal_y) // max_points
        ideal_y = ideal_y[::step]
        ideal_time = ideal_time[::step]
    if len(comparison_y) > max_points:
        step = len(comparison_y) // max_points
        comparison_y = comparison_y[::step]
        comparison_time = comparison_time[::step]
    
    # Create figure
    fig = go.Figure()
    
    # Add ideal waveform
    fig.add_trace(go.Scatter(
        x=ideal_time,
        y=ideal_y,
        name='Ideal',
        line=dict(color='blue', width=1)
    ))
    
    # Add comparison waveform
    fig.add_trace(go.Scatter(
        x=comparison_time,
        y=comparison_y,
        name='Comparison',
        line=dict(color='red', width=1)
    ))
    
    # Remove the temporary files
    os.unlink(temp_ideal_path)
    os.unlink(temp_comparison_path)

    # Update layout
    fig.update_layout(
        title='Waveform Comparison',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=400,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Function to create a Spectrogram comparison with plotly
def compare_spectrograms(ideal_audio, comparison_audio):
    # Save the uploaded files to temporary locations
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_ideal:
        temp_ideal.write(ideal_audio.getvalue())
        temp_ideal_path = temp_ideal.name

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_comparison:
        temp_comparison.write(comparison_audio.getvalue())
        temp_comparison_path = temp_comparison.name

    # Load audio files
    ideal_y, ideal_sr = librosa.load(temp_ideal_path, sr=None, duration=480)  # Limit to 30 seconds
    comparison_y, comparison_sr = librosa.load(temp_comparison_path, sr=None, duration=480)  # Limit to 30 seconds
    
    # Compute spectrograms
    n_fft = 2048  # Reduce FFT window size
    hop_length = 512  # Increase hop length
    ideal_spec = librosa.stft(ideal_y, n_fft=n_fft, hop_length=hop_length)
    ideal_spec_db = librosa.amplitude_to_db(abs(ideal_spec))
    comparison_spec = librosa.stft(comparison_y, n_fft=n_fft, hop_length=hop_length)
    comparison_spec_db = librosa.amplitude_to_db(abs(comparison_spec))
    
    # Downsample spectrograms
    ideal_spec_db = ideal_spec_db[::4, ::4]
    comparison_spec_db = comparison_spec_db[::4, ::4]
    
    # Create figure
    fig = go.Figure()
    
    # Add ideal spectrogram
    fig.add_trace(go.Heatmap(
        z=ideal_spec_db,
        colorscale='Viridis',
        name='Ideal',
        colorbar=dict(title='dB')
    ))
    
    # Add comparison spectrogram
    fig.add_trace(go.Heatmap(
        z=comparison_spec_db,
        colorscale='Viridis',
        name='Comparison',
        colorbar=dict(title='dB')
    ))
    
    # Remove the temporary files
    os.unlink(temp_ideal_path)
    os.unlink(temp_comparison_path)

    # Update layout
    fig.update_layout(
        title='Spectrogram Comparison',
        xaxis_title='Time',
        yaxis_title='Frequency',
        height=600,
        width=1000,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=[
                dict(label="Ideal",
                     method="update",
                     args=[{"visible": [True, False]}]),
                dict(label="Comparison",
                     method="update",
                     args=[{"visible": [False, True]}]),
                dict(label="Both",
                     method="update",
                     args=[{"visible": [True, True]}]),
            ]
        )]
    )
    
    return fig

# Function to perform time domain analysis
def time_domain_analysis(audio_file):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_path = temp_file.name

        # Load the audio file using librosa
        y, sr = librosa.load(temp_path)

        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Calculate zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # Calculate amplitude envelope
        frame_length = 2048
        hop_length = 512
        amplitude_envelope = np.array([max(y[i:i+frame_length]) for i in range(0, len(y), hop_length)])
        
        # Calculate temporal centroid
        times = librosa.times_like(y)
        temporal_centroid = np.sum(times * y**2) / np.sum(y**2)
        
        # Remove the temporary file
        os.unlink(temp_path)
        
        return {
            'rms': rms,
            'zcr': zcr,
            'amplitude_envelope': amplitude_envelope,
            'temporal_centroid': temporal_centroid
        }
    except Exception as e:
        return f"An error occurred during time domain analysis: {str(e)}"

# Function to plot time domain features
def plot_time_domain_features(ideal_features, comparison_features):
    fig = go.Figure()

    # Time array for x-axis
    time = np.arange(len(ideal_features['rms'])) / len(ideal_features['rms'])

    # Plot RMS
    fig.add_trace(go.Scatter(x=time, y=ideal_features['rms'], name='Ideal RMS', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time, y=comparison_features['rms'], name='Comparison RMS', line=dict(color='red')))

    # Plot ZCR
    fig.add_trace(go.Scatter(x=time, y=ideal_features['zcr'], name='Ideal ZCR', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=time, y=comparison_features['zcr'], name='Comparison ZCR', line=dict(color='orange')))

    # Plot Amplitude Envelope
    time_ae = np.arange(len(ideal_features['amplitude_envelope'])) / len(ideal_features['amplitude_envelope'])
    fig.add_trace(go.Scatter(x=time_ae, y=ideal_features['amplitude_envelope'], name='Ideal Amplitude Envelope', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=time_ae, y=comparison_features['amplitude_envelope'], name='Comparison Amplitude Envelope', line=dict(color='brown')))

    # Update layout
    fig.update_layout(
        title='Time Domain Analysis',
        xaxis_title='Time (normalized)',
        yaxis_title='Amplitude',
        height=600,
        width=1000,
        legend_title='Features'
    )

    return fig


# Function to highlight differences between two texts
def highlight_diff(text1, text2):
    d = difflib.Differ()
    diff = list(d.compare(text1.split(), text2.split()))
    
    result = []
    for word in diff:
        if word.startswith('  '):
            result.append(word[2:])
        elif word.startswith('- '):
            result.append(f'<span style="background-color: #ffcccb;">{word[2:]}</span>')
        elif word.startswith('+ '):
            result.append(f'<span style="background-color: #90EE90;">{word[2:]}</span>')
    
    return ' '.join(result)

def calculate_overall_score(similarity, audio_scores_diff):
    # Normalize similarity to 0-100 scale if it's not already
    similarity_score = similarity if 0 <= similarity <= 100 else similarity * 100
    
    # Calculate audio score (inverse of the absolute difference)
    audio_score = 100 - min(100, abs(audio_scores_diff))
    
    # Calculate weighted score
    weighted_score = (similarity_score * 0.7) + (audio_score * 0.3)
    
    # Ensure the weighted_score is a float before rounding
    return round(float(weighted_score), 2)


# Streamlit app
def main():
    st.set_page_config(page_title="Imam Evaluator App", page_icon="🕌", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stTextInput>div>div>input {
            color: #4F8BF9;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Imam Evaluator App 🕌")

    # Create tabs with icons
    tab1, tab2 = st.tabs(["📊 Transcription and Comparison", "🛠️ Tools"])

    with tab1:
        st.write("Upload two audio files (max 20 MB each). The first audio will be considered the ideal, and the second will be compared to it.")

        col1, col2 = st.columns(2)
        with col1:
            ideal_audio = st.file_uploader("Choose the ideal audio file (max 20 MB)", type=["wav", "mp3", "ogg", "mpeg"])
        with col2:
            comparison_audio = st.file_uploader("Choose the audio file to compare (max 20 MB)", type=["wav", "mp3", "ogg", "mpeg"])

        if ideal_audio is not None and comparison_audio is not None:
            if not check_file_size(ideal_audio) or not check_file_size(comparison_audio):
                st.error("Error: One or both audio files exceed the 20 MB limit.")
            else:
                with st.spinner("Processing audio files..."):
                    # Transcribe and analyze ideal audio
                    ideal_text = transcribe_audio(ideal_audio)
                    ideal_overall, ideal_pitch, ideal_tone, ideal_rhythm, ideal_energy = analyze_audio(ideal_audio)
                    
                    # Transcribe and analyze comparison audio
                    comparison_text = transcribe_audio(comparison_audio)
                    comparison_overall, comparison_pitch, comparison_tone, comparison_rhythm, comparison_energy = analyze_audio(comparison_audio)
                
                # Format texts for Arabic
                formatted_ideal_text = format_arabic_text(ideal_text)
                formatted_comparison_text = format_arabic_text(comparison_text)
                
                # Display texts with highlighted differences
                st.subheader("Transcription Comparison")
                highlighted_diff = highlight_diff(ideal_text, comparison_text)
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;">
                        {highlighted_diff}
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        <span style="background-color: #ffcccb; padding: 2px 4px; border-radius: 3px;">Red</span>: Removed text
                        <span style="background-color: #90EE90; padding: 2px 4px; border-radius: 3px; margin-left: 10px;">Green</span>: Added text
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Compare audio scores
                overall_diff, pitch_diff, tone_diff, rhythm_diff, energy_diff = compare_audio_scores(
                    (ideal_overall, ideal_pitch, ideal_tone, ideal_rhythm, ideal_energy),
                    (comparison_overall, comparison_pitch, comparison_tone, comparison_rhythm, comparison_energy)
                )

                # Time domain analysis
                st.subheader("Time Domain Analysis")
                
                ideal_analysis = time_domain_analysis(ideal_audio)
                comparison_analysis = time_domain_analysis(comparison_audio)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig_rms = go.Figure()
                    fig_rms.add_trace(go.Scatter(y=ideal_analysis['rms'], name='Ideal', line=dict(color='blue')))
                    fig_rms.add_trace(go.Scatter(y=comparison_analysis['rms'], name='Comparison', line=dict(color='red')))
                    fig_rms.update_layout(title='RMS Energy', xaxis_title='Frame', yaxis_title='Energy', height=300)
                    st.plotly_chart(fig_rms, use_container_width=True)
                
                with col2:
                    fig_zcr = go.Figure()
                    fig_zcr.add_trace(go.Scatter(y=ideal_analysis['zcr'], name='Ideal', line=dict(color='blue')))
                    fig_zcr.add_trace(go.Scatter(y=comparison_analysis['zcr'], name='Comparison', line=dict(color='red')))
                    fig_zcr.update_layout(title='Zero Crossing Rate', xaxis_title='Frame', yaxis_title='Rate', height=300)
                    st.plotly_chart(fig_zcr, use_container_width=True)
                
                with col3:
                    fig_ae = go.Figure()
                    fig_ae.add_trace(go.Scatter(y=ideal_analysis['amplitude_envelope'], name='Ideal', line=dict(color='blue')))
                    fig_ae.add_trace(go.Scatter(y=comparison_analysis['amplitude_envelope'], name='Comparison', line=dict(color='red')))
                    fig_ae.update_layout(title='Amplitude Envelope', xaxis_title='Frame', yaxis_title='Amplitude', height=300)
                    st.plotly_chart(fig_ae, use_container_width=True)
                
                # Display Temporal Centroid
                st.metric("Temporal Centroid", 
                          f"Ideal: {ideal_analysis['temporal_centroid']:.4f}s | Comparison: {comparison_analysis['temporal_centroid']:.4f}s")
                
                # Create and display comparison chart
                st.subheader("Audio Score Comparison")
                chart = create_comparison_chart(
                    [ideal_overall, ideal_pitch, ideal_tone, ideal_rhythm, ideal_energy],
                    [comparison_overall, comparison_pitch, comparison_tone, comparison_rhythm, comparison_energy]
                )
                st.plotly_chart(chart, use_container_width=True)

                # Create and display waveform comparison
                st.subheader("Waveform Comparison")
                waveform_fig = create_waveform_comparison(ideal_audio, comparison_audio)
                st.plotly_chart(waveform_fig, use_container_width=True)
                
                # Create and display spectrogram comparison
                st.subheader("Spectrogram Comparison")
                spectrogram_fig = compare_spectrograms(ideal_audio, comparison_audio)
                st.plotly_chart(spectrogram_fig, use_container_width=True)

                # Audio Score Comparison details
                with st.expander("Detailed Audio Score Comparison"):
                    st.info("The following differences are calculated by subtracting the ideal score from the comparison score for each category. Positive values mean the comparison score is higher than the ideal score.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Difference", f"{float(overall_diff):.2f}", delta=f"{'Higher' if float(overall_diff) > 0 else 'Lower'}")
                        st.metric("Pitch Difference", f"{float(pitch_diff):.2f}", delta=f"{'Higher' if float(pitch_diff) > 0 else 'Lower'}")
                        st.caption("Calculated from the median pitch of non-zero magnitudes in the audio")
                    with col2:
                        st.metric("Tone Difference", f"{float(tone_diff):.2f}", delta=f"{'Higher' if float(tone_diff) > 0 else 'Lower'}")
                        st.caption("Based on spectral centroid, representing audio brightness")
                        st.metric("Rhythm Difference", f"{float(rhythm_diff):.2f}", delta=f"{'Higher' if float(rhythm_diff) > 0 else 'Lower'}")
                        st.caption("Derived from the tempo of the audio")
                        st.metric("Energy Difference", f"{float(energy_diff):.2f}", delta=f"{'Higher' if float(energy_diff) > 0 else 'Lower'}")
                        st.caption("Computed from the root mean square energy of the audio")
                
                # Compare texts and display similarity
                if not ideal_text.startswith("An error occurred") and not comparison_text.startswith("An error occurred"):
                    comparison, _ = compare_texts(ideal_text, comparison_text)
                    similarity = compare_audio_fingerprints(ideal_audio, comparison_audio)
                    st.subheader("Text Similarity Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Similarity to Ideal", similarity)
                    with col2:
                        overall_score = calculate_overall_score(similarity, overall_diff)
                        st.metric("Overall Score", f"{overall_score}%")
                    
                    with st.expander("Detailed Comparison and Feedback"):
                        st.markdown(comparison, unsafe_allow_html=True)
                else:
                    if ideal_text.startswith("An error occurred"):
                        st.error(ideal_text)
                    if comparison_text.startswith("An error occurred"):
                        st.error(comparison_text)

    with tab2:
        st.header("Text to Speech Converter")
        text_input = st.text_area("Enter text to convert to speech:", height=150)
        if st.button("Convert to Speech", key="convert_button"):
            if text_input:
                with st.spinner("Converting text to speech..."):
                    audio_result = text_to_speech(text_input)
                if isinstance(audio_result, io.BytesIO):
                    st.audio(audio_result, format="audio/mp3")
                    st.download_button(
                        label="Download Audio",
                        data=audio_result,
                        file_name="text_to_speech.mp3",
                        mime="audio/wav"
                    )
                else:
                    st.error(audio_result)
            else:
                st.warning("Please enter some text to convert.")

if __name__ == "__main__":
    main()

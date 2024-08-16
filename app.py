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
        pitch_score = min(100, max(0, (pitch - 50) / 400 * 100))
        tone_score = min(100, max(0, tone_brightness / 5000 * 100))
        rhythm_score = min(100, max(0, tempo / 2))
        energy_score = min(100, max(0, energy * 1000))
        
        overall_score = (pitch_score + tone_score + rhythm_score + energy_score) / 4
        
        # Remove the temporary file
        os.remove("temp_audio.mp3")
        
        return overall_score, pitch_score, tone_score, rhythm_score, energy_score
    except Exception as e:
        return f"An error occurred during audio analysis: {str(e)}", None, None, None, None

# Function to compare texts and generate HTML with colored differences using OpenAI
def compare_texts(ideal_text, comparison_text):
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from scipy.spatial.distance import cosine
        import numpy as np
        import tempfile

        # Save the texts to temporary audio files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_ideal:
            temp_ideal.write(text_to_speech(ideal_text).getvalue())
            ideal_path = temp_ideal.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_comparison:
            temp_comparison.write(text_to_speech(comparison_text).getvalue())
            comparison_path = temp_comparison.name

        # Load and preprocess the audio files
        wav1 = preprocess_wav(ideal_path)
        wav2 = preprocess_wav(comparison_path)

        # Initialize the VoiceEncoder
        encoder = VoiceEncoder()

        # Extract embeddings
        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)

        # Calculate Cosine Similarity
        similarity = 1 - cosine(embed1, embed2)

        # Calculate Euclidean Distance
        euclidean_distance = np.linalg.norm(embed1 - embed2)

        # Remove temporary files
        os.unlink(ideal_path)
        os.unlink(comparison_path)

        return f"Cosine Similarity: {similarity}", similarity * 100
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
    
    fig = go.Figure(data=[
        go.Bar(name='Ideal', x=categories, y=ideal_scores),
        go.Bar(name='Comparison', x=categories, y=comparison_scores)
    ])
    
    fig.update_layout(
        title='Audio Score Comparison',
        xaxis_title='Categories',
        yaxis_title='Scores',
        barmode='group',
        height=500,  # Increase the height of the chart
        width=700    # Increase the width of the chart
    )
    
    # Add data labels on top of each bar
    for i, (ideal, comparison) in enumerate(zip(ideal_scores, comparison_scores)):
        fig.add_annotation(x=categories[i], y=ideal,
                           text=f"{ideal:.2f}",
                           showarrow=False,
                           yshift=10)
        fig.add_annotation(x=categories[i], y=comparison,
                           text=f"{comparison:.2f}",
                           showarrow=False,
                           yshift=10)
    
    # Ensure y-axis starts from 0 and goes up to 100
    fig.update_yaxes(range=[0, 100])
    
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
    ideal_y, ideal_sr = librosa.load(temp_ideal_path, sr=None, duration=120)  # Limit to 30 seconds
    comparison_y, comparison_sr = librosa.load(temp_comparison_path, sr=None, duration=120)  # Limit to 30 seconds
    
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
    st.title("Imam Evaluator App")

    # Create tabs
    tab1, tab2 = st.tabs(["Transcription and Comparison", "Tools"])

    with tab1:
        st.write("Upload two audio files (max 20 MB each). The first audio will be considered the ideal, and the second will be compared to it.")

        # File uploaders
        ideal_audio = st.file_uploader("Choose the ideal audio file (max 20 MB)", type=["wav", "mp3", "ogg", "mpeg"])
        comparison_audio = st.file_uploader("Choose the audio file to compare (max 20 MB)", type=["wav", "mp3", "ogg", "mpeg"])

        if ideal_audio is not None and comparison_audio is not None:
            # Check file sizes
            if not check_file_size(ideal_audio):
                st.error("Error: Ideal audio file size exceeds 20 MB limit.")
            elif not check_file_size(comparison_audio):
                st.error("Error: Comparison audio file size exceeds 20 MB limit.")
            else:
                # Transcribe and analyze ideal audio
                st.write("Transcribing and analyzing ideal audio...")
                ideal_text = transcribe_audio(ideal_audio)
                ideal_overall, ideal_pitch, ideal_tone, ideal_rhythm, ideal_energy = analyze_audio(ideal_audio)
                
                # Transcribe and analyze comparison audio
                st.write("Transcribing and analyzing comparison audio...")
                comparison_text = transcribe_audio(comparison_audio)
                comparison_overall, comparison_pitch, comparison_tone, comparison_rhythm, comparison_energy = analyze_audio(comparison_audio)
                
                # Format texts for Arabic
                formatted_ideal_text = format_arabic_text(ideal_text)
                formatted_comparison_text = format_arabic_text(comparison_text)
                
                # Display texts with highlighted differences
                st.markdown("**Transcription Comparison:**")
                highlighted_diff = highlight_diff(ideal_text, comparison_text)
                st.markdown("### Transcription Comparison", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;">
                        {}
                    </div>
                    """.format(highlighted_diff),
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
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

                # Perform time domain analysis
                st.markdown("### Time Domain Analysis")
                
                ideal_analysis = time_domain_analysis(ideal_audio)
                comparison_analysis = time_domain_analysis(comparison_audio)
                
                # Plot RMS Energy
                fig_rms = go.Figure()
                fig_rms.add_trace(go.Scatter(y=ideal_analysis['rms'], name='Ideal', line=dict(color='blue')))
                fig_rms.add_trace(go.Scatter(y=comparison_analysis['rms'], name='Comparison', line=dict(color='red')))
                fig_rms.update_layout(title='RMS Energy', xaxis_title='Frame', yaxis_title='Energy')
                st.plotly_chart(fig_rms)
                
                # Plot Zero Crossing Rate
                fig_zcr = go.Figure()
                fig_zcr.add_trace(go.Scatter(y=ideal_analysis['zcr'], name='Ideal', line=dict(color='blue')))
                fig_zcr.add_trace(go.Scatter(y=comparison_analysis['zcr'], name='Comparison', line=dict(color='red')))
                fig_zcr.update_layout(title='Zero Crossing Rate', xaxis_title='Frame', yaxis_title='Rate')
                st.plotly_chart(fig_zcr)
                
                # Plot Amplitude Envelope
                fig_ae = go.Figure()
                fig_ae.add_trace(go.Scatter(y=ideal_analysis['amplitude_envelope'], name='Ideal', line=dict(color='blue')))
                fig_ae.add_trace(go.Scatter(y=comparison_analysis['amplitude_envelope'], name='Comparison', line=dict(color='red')))
                fig_ae.update_layout(title='Amplitude Envelope', xaxis_title='Frame', yaxis_title='Amplitude')
                st.plotly_chart(fig_ae)
                
                # Display Temporal Centroid
                st.markdown(f"**Temporal Centroid:**")
                st.markdown(f"Ideal: {ideal_analysis['temporal_centroid']:.4f} seconds")
                st.markdown(f"Comparison: {comparison_analysis['temporal_centroid']:.4f} seconds")
                
                # Create and display comparison chart
                chart = create_comparison_chart(
                    [ideal_overall, ideal_pitch, ideal_tone, ideal_rhythm, ideal_energy],
                    [comparison_overall, comparison_pitch, comparison_tone, comparison_rhythm, comparison_energy]
                )
                st.plotly_chart(chart)

                # Create and display waveform comparison
                st.markdown("### Waveform Comparison")
                waveform_fig = create_waveform_comparison(ideal_audio, comparison_audio)
                st.plotly_chart(waveform_fig)
                
                # Create and display spectrogram comparison
                st.markdown("### Spectrogram Comparison")
                spectrogram_fig = compare_spectrograms(ideal_audio, comparison_audio)
                st.plotly_chart(spectrogram_fig)


                
                st.markdown("**Audio Score Comparison:**")
                st.markdown("The following differences are calculated by subtracting the ideal score from the comparison score for each category:")
                st.markdown(f"Overall Difference: {float(overall_diff):.2f} ({'higher' if float(overall_diff) > 0 else 'lower'})")
                st.markdown("(Positive value means the comparison score is higher than the ideal score)")
                st.markdown(f"Pitch Difference: {float(pitch_diff):.2f} ({'higher' if float(pitch_diff) > 0 else 'lower'})")
                st.markdown("(Calculated from the median pitch of non-zero magnitudes in the audio)")
                st.markdown(f"Tone Difference: {float(tone_diff):.2f} ({'higher' if float(tone_diff) > 0 else 'lower'})")
                st.markdown("(Based on spectral centroid, representing audio brightness)")
                st.markdown(f"Rhythm Difference: {float(rhythm_diff):.2f} ({'higher' if float(rhythm_diff) > 0 else 'lower'})")
                st.markdown("(Derived from the tempo of the audio)")
                st.markdown(f"Energy Difference: {float(energy_diff):.2f} ({'higher' if float(energy_diff) > 0 else 'lower'})")
                st.markdown("(Computed from the root mean square energy of the audio)")
                
                # Compare texts and display similarity
                if not ideal_text.startswith("An error occurred") and not comparison_text.startswith("An error occurred"):
                    comparison, similarity = compare_texts(ideal_text, comparison_text)
                    st.markdown(f"**Similarity to Ideal: {similarity:.2f}%**")
                    st.markdown("**Comparison and Feedback:**")
                    # st.markdown(comparison, unsafe_allow_html=True)
                    overall_score = calculate_overall_score(similarity, overall_diff)
    
                    st.markdown(f"**Overall Score: {overall_score}%**")
                    # st.markdown(f"(Based on 80% text similarity and 20% audio comparison)")
                else:
                    if ideal_text.startswith("An error occurred"):
                        st.error(ideal_text)
                    if comparison_text.startswith("An error occurred"):
                        st.error(comparison_text)

    with tab2:
        st.header("Tools")
        st.subheader("Text to Speech Converter")
        text_input = st.text_area("Enter text to convert to speech:", height=150)
        if st.button("Convert to Speech"):
            if text_input:
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

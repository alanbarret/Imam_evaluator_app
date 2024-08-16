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
        pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7])
        
        # Tone analysis (using spectral centroid as a proxy for "brightness")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        tone_brightness = np.mean(spectral_centroids)
        
        # Normalize scores to a 0-100 scale
        pitch_score = min(100, max(0, (pitch - 50) / 400 * 100))
        tone_score = min(100, max(0, tone_brightness / 5000 * 100))
        
        overall_score = (pitch_score + tone_score) / 2
        
        # Remove the temporary file
        os.remove("temp_audio.mp3")
        
        return overall_score, pitch_score, tone_score
    except Exception as e:
        return f"An error occurred during audio analysis: {str(e)}", None, None

# Function to compare texts and generate HTML with colored differences using OpenAI
def compare_texts(ideal_text, comparison_text):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment variables.", 0
    
    client = OpenAI(api_key=api_key)
    
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
                    {"role": "system", "content": "You are a text comparison assistant. Compare the following two text chunks, where the first is the ideal text and the second is the text to be compared. Highlight the differences and provide feedback on how the second text can be improved to match the ideal text. If the audio doesn't match the ideal text then return a message that says 'The audio does not match the ideal text'"},
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
    
    return overall_diff, pitch_diff, tone_diff

# Function to create a comparison chart
def create_comparison_chart(ideal_scores, comparison_scores):
    categories = ['Overall', 'Pitch', 'Tone']
    
    fig = go.Figure(data=[
        go.Bar(name='Ideal', x=categories, y=ideal_scores),
        go.Bar(name='Comparison', x=categories, y=comparison_scores)
    ])
    
    fig.update_layout(
        title='Audio Score Comparison',
        xaxis_title='Categories',
        yaxis_title='Scores',
        barmode='group'
    )
    
    return fig

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
                ideal_overall, ideal_pitch, ideal_tone = analyze_audio(ideal_audio)
                
                # Transcribe and analyze comparison audio
                st.write("Transcribing and analyzing comparison audio...")
                comparison_text = transcribe_audio(comparison_audio)
                comparison_overall, comparison_pitch, comparison_tone = analyze_audio(comparison_audio)
                
                # Format texts for Arabic
                formatted_ideal_text = format_arabic_text(ideal_text)
                formatted_comparison_text = format_arabic_text(comparison_text)
                
                # # Display texts and scores side by side
                # col1, col2 = st.columns(2)
                
                # with col1:
                #     st.markdown("**Ideal Audio Transcription:**")
                #     st.text_area("", value=formatted_ideal_text, height=300, key="ideal_text")
                #     st.markdown(f"**Ideal Audio Quality Scores:**")
                #     st.markdown(f"Overall: {ideal_overall:.2f}")
                #     st.markdown(f"Pitch: {ideal_pitch:.2f}")
                #     st.markdown(f"Tone: {ideal_tone:.2f}")
                
                # with col2:
                #     st.markdown("**Comparison Audio Transcription:**")
                #     st.text_area("", value=formatted_comparison_text, height=300, key="comparison_text")
                #     st.markdown(f"**Comparison Audio Quality Scores:**")
                #     st.markdown(f"Overall: {comparison_overall:.2f}")
                #     st.markdown(f"Pitch: {comparison_pitch:.2f}")
                #     st.markdown(f"Tone: {comparison_tone:.2f}")
                
                # Compare audio scores
                overall_diff, pitch_diff, tone_diff = compare_audio_scores(
                    (ideal_overall, ideal_pitch, ideal_tone),
                    (comparison_overall, comparison_pitch, comparison_tone)
                )
                
                # Create and display comparison chart
                chart = create_comparison_chart(
                    [ideal_overall, ideal_pitch, ideal_tone],
                    [comparison_overall, comparison_pitch, comparison_tone]
                )
                st.plotly_chart(chart)
                
                st.markdown("**Audio Score Comparison:**")
                st.markdown(f"Overall Difference: {float(overall_diff):.2f} ({'higher' if float(overall_diff) > 0 else 'lower'})")
                st.markdown(f"Pitch Difference: {float(pitch_diff):.2f} ({'higher' if float(pitch_diff) > 0 else 'lower'})")
                st.markdown(f"Tone Difference: {float(tone_diff):.2f} ({'higher' if float(tone_diff) > 0 else 'lower'})")
                
                # Compare texts and display similarity
                if not ideal_text.startswith("An error occurred") and not comparison_text.startswith("An error occurred"):
                    comparison, similarity = compare_texts(ideal_text, comparison_text)
                    st.markdown(f"**Similarity to Ideal: {similarity:.2f}%**")
                    st.markdown("**Comparison and Feedback:**")
                    st.markdown(comparison, unsafe_allow_html=True)
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

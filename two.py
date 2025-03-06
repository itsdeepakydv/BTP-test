

# aai.settings.api_key = "85e63f64ed954279bff9a588c0bd2f2f"  # Replace with your actual API key


import streamlit as st
import assemblyai as aai
import os

aai.settings.api_key = "85e63f64ed954279bff9a588c0bd2f2f"  # Replace with your actual API key

def transcribe_audio(file_path):
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        speakers_expected=10
    )
    
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)
    return transcript

def calculate_interactivity_score(transcript, total_strength):
    unique_speakers = set()
    interactive_pair_count = 0
    non_interactive_count = 0
    previous_speaker = None
    previous_was_question = False
    speaker_durations = {}
    
    for utterance in transcript.utterances:
        unique_speakers.add(utterance.speaker)
        duration = utterance.end - utterance.start
        
        # Track speaking time per speaker
        if utterance.speaker not in speaker_durations:
            speaker_durations[utterance.speaker] = 0
        speaker_durations[utterance.speaker] += duration
        
        # Question-Response Analysis
        if "?" in utterance.text.lower() or any(word in utterance.text.lower() for word in ["what", "who", "where", "when", "why", "how"]):
            previous_was_question = True
        elif previous_was_question and previous_speaker != utterance.speaker:
            interactive_pair_count += 1
            previous_was_question = False
        else:
            non_interactive_count += 1
        
        previous_speaker = utterance.speaker
    
    unique_speakers_count = len(unique_speakers)
    
    if total_strength == 0:
        st.error("Total strength cannot be zero.")
        return None
    
    # Compute teacher speaking time
    teacher = max(speaker_durations, key=speaker_durations.get)
    total_duration = sum(speaker_durations.values())
    teacher_speech_ratio = (speaker_durations[teacher] / total_duration) * 100 if total_duration > 0 else 0
    
    # Compute Interactivity Score (Normalized 1-5)
    raw_score = (interactive_pair_count * 2) / (interactive_pair_count * 2 + non_interactive_count) + (unique_speakers_count / total_strength)
    score_normalized = 1 + (raw_score - 0) * (5 - 1) / (2 - 0)
    
    return {
        "interactive_pairs": interactive_pair_count,
        "non_interactive": non_interactive_count,
        "unique_speakers": unique_speakers_count,
        "teacher_speaking_time": teacher_speech_ratio,
        "final_score": round(score_normalized, 2)
    }

def analyze_teaching_style(teacher_speech_ratio, interactivity_score):
    if teacher_speech_ratio > 75:
        style = "Monotonous (Teacher-Dominated)"
    elif teacher_speech_ratio > 50:
        style = "Lecture-Based (Minimal Interaction)"
    elif interactivity_score > 3.5:
        style = "Highly Interactive"
    elif interactivity_score > 2.5:
        style = "Moderately Interactive"
    else:
        style = "Passive Discussion"
    return style

def main():
    st.title("Audio Interactivity & Teaching Style Analysis")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    total_strength = st.number_input("Enter total class strength", min_value=1, step=1)
    
    if uploaded_file is not None and total_strength:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Processing... This may take a while.")
        transcript = transcribe_audio(file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            st.error("Error in transcription: " + transcript.error)
        else:
            results = calculate_interactivity_score(transcript, total_strength)
            
            # Display results
            st.subheader("CLASS INTERACTIVITY STATS")
            st.write(f"**Interactive Pair Count:** {results['interactive_pairs']}")
            st.write(f"**Non-Interactive Count:** {results['non_interactive']}")
            st.write(f"**Unique Speakers Count:** {results['unique_speakers']}")
            st.write(f"**Teacher Speaking Time:** {results['teacher_speaking_time']:.2f}%")
            st.write(f"**Final Normalized Score (1-5):** {results['final_score']}")
            
            teaching_style = analyze_teaching_style(results['teacher_speaking_time'], results['final_score'])
            
            st.subheader("TEACHING STYLE")
            st.write(f"**Detected Style:** {teaching_style}")
            
        os.remove(file_path)  # Clean up the temporary file

if __name__ == "__main__":
    main()

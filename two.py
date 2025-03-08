import streamlit as st
import assemblyai as aai
import os
# import spacy
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os

# Load spaCy model for Named Entity Recognition (NER)
# nlp = spacy.load("en_core_web_sm")

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

# def extract_topics(transcript):
#     topic_timestamps = []
#     topic_count = {}
    
#     for utterance in transcript.utterances:
#         doc = nlp(utterance.text)
#         for ent in doc.ents:
#             if ent.label_ in ["ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "PERSON"]:
#                 topic = ent.text
#                 topic_timestamps.append((topic, utterance.start, utterance.end))
#                 topic_count[topic] = topic_count.get(topic, 0) + 1
                
#     return topic_timestamps, topic_count

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
        
        if utterance.speaker not in speaker_durations:
            speaker_durations[utterance.speaker] = 0
        speaker_durations[utterance.speaker] += duration
        
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
    
    teacher = max(speaker_durations, key=speaker_durations.get)
    total_duration = sum(speaker_durations.values())
    teacher_speech_ratio = (speaker_durations[teacher] / total_duration) * 100 if total_duration > 0 else 0
    
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


def generate_sankey_diagram(transcript,uploaded_file):
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        summarization=True,
        summary_model=aai.SummarizationModel.informative,
        iab_categories=True,
        auto_highlights=True,
        sentiment_analysis=True,
        entity_detection=True,
        speaker_labels=True,
        language_detection=True,
        speakers_expected=10,
        summary_type=aai.SummarizationType.bullets
    )   
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(uploaded_file)
    topic_timestamps = []
    topic_count = {}
    
    for result in transcript.iab_categories.results:
        start_time = result.timestamp.start // 1000
        end_time = result.timestamp.end // 1000
        for label in result.labels:
            topic = label.label.split(">")[-1]
            topic_timestamps.append((topic, start_time, end_time))
            topic_count[topic] = topic_count.get(topic, 0) + 1
    
    def format_time(seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"
    
    df = pd.DataFrame(topic_timestamps, columns=["Topic", "StartTime", "EndTime"])
    df = df.sort_values("StartTime")
    
    top_topics = sorted(topic_count, key=topic_count.get, reverse=True)[:10]
    df = df[df["Topic"].isin(top_topics)]
    
    labels = ["Start"] + top_topics + ["End"]
    source, target, values, timestamps = [], [], [], []
    prev_topic = "Start"
    
    for _, row in df.iterrows():
        current_topic = row["Topic"]
        start_time = row["StartTime"]

        if current_topic in labels:
            source.append(labels.index(prev_topic))
            target.append(labels.index(current_topic))
            values.append(1)
            timestamps.append(f"{format_time(start_time)}")
            prev_topic = current_topic
    
    source.append(labels.index(prev_topic))
    target.append(labels.index("End"))
    values.append(1)
    timestamps.append(f"{format_time(row['EndTime'])}")
    
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#F4A226",
              "#6A0DAD", "#2E8B57", "#1E90FF", "#8B0000", "#FFD700"]
    
    fig = go.Figure(go.Sankey(
        arrangement="perpendicular",
        node=dict(
            label=[""] * len(labels),
            pad=50,
            thickness=30,
            color=colors[:len(labels)],
            customdata=labels,
            hovertemplate="Topic: %{customdata}<extra></extra>",
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            customdata=timestamps,
            hovertemplate="From: %{source.labels} → To: %{target.labels}<br>Time: %{customdata}",
            color=['rgba(0, 0, 255, 0.3)'] * len(source)
        )
    ))
    
    fig.update_layout(
        title_text="🔗 Improved Classroom Topic Flow with Timestamps",
        font_size=12,
        height=400
    )
    
    return fig


def transcribe_and_visualize(file_url, speakers_expected=10):
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        summarization=True,
        summary_model=aai.SummarizationModel.informative,
        iab_categories=True,
        auto_highlights=True,
        sentiment_analysis=True,
        entity_detection=True,
        speaker_labels=True,
        language_detection=True,
        speakers_expected=10,
        summary_type=aai.SummarizationType.bullets
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_url)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Error: {transcript.error}")
        return

    speaker_activity = []
    for utterance in transcript.utterances:
        speaker = utterance.speaker
        start_time = utterance.start // 1000  # Convert ms to seconds
        end_time = utterance.end // 1000
        duration = end_time - start_time
        speaker_activity.append({
            "Speaker": speaker,
            "StartTime": start_time,
            "EndTime": end_time,
            "Duration": duration
        })
    
    df_speaker = pd.DataFrame(speaker_activity)

    # Convert time to datetime format
    df_speaker["StartTime"] = pd.to_datetime(df_speaker["StartTime"], unit="s")
    df_speaker["EndTime"] = pd.to_datetime(df_speaker["EndTime"], unit="s")

    # Create Speaker Timeline
    fig = px.timeline(
        df_speaker,
        x_start="StartTime",
        x_end="EndTime",
        y="Speaker",
        color="Speaker",
        title="🎤 Speaker Activity Timeline",
        labels={"Speaker": "Speakers"},
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Speakers",
        hovermode="x",
        plot_bgcolor="white"
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display Summary
    st.subheader("📌 Summary")
    st.write(transcript.summary)

    # Display Sentiment Analysis
    # st.subheader("😊 Sentiment Analysis")
    # for sentiment_result in transcript.sentiment_analysis:
    #     st.write(f"**{sentiment_result.text}**")
    #     st.write(f"Sentiment: `{sentiment_result.sentiment}`")
    #     st.write("---")




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
            topic_timestamps, topic_count = extract_topics(transcript)
            
            st.subheader("CLASS INTERACTIVITY STATS")
            st.write(f"**Interactive Pair Count:** {results['interactive_pairs']}")
            st.write(f"**Non-Interactive Count:** {results['non_interactive']}")
            st.write(f"**Unique Speakers Count:** {results['unique_speakers']}")
            st.write(f"**Teacher Speaking Time:** {results['teacher_speaking_time']:.2f}%")
            st.write(f"**Final Normalized Score (1-5):** {results['final_score']}")
            
            teaching_style = analyze_teaching_style(results['teacher_speaking_time'], results['final_score'])
            
            st.subheader("TEACHING STYLE")
            st.write(f"**Detected Style:** {teaching_style}")
            
            # Convert topic timestamps into DataFrame
            df = pd.DataFrame(topic_timestamps, columns=["Topic", "StartTime", "EndTime"])
            df = df.sort_values("StartTime")
            
            # Get the top 10 most discussed topics
            top_10_topics = sorted(topic_count, key=topic_count.get, reverse=True)[:10]
            df = df[df["Topic"].isin(top_10_topics)]
            
            # Format timestamps for readability
            df["StartTimeFormatted"] = df["StartTime"].apply(lambda x: f"{x//60:02}:{x%60:02}")
            df["EndTimeFormatted"] = df["EndTime"].apply(lambda x: f"{x//60:02}:{x%60:02}")
            
            # Create Interactive Bar Chart for Timeline
            fig = px.bar(
                df,
                x="StartTime",
                y="Topic",
                orientation="h",
                text="StartTimeFormatted",
                title="📊 Interactive Topic Discussion Timeline (Top 10 Topics)",
                color="Topic",
                labels={"StartTime": "Start Time (seconds)", "Topic": "Topic Discussed"},
                hover_data={"StartTimeFormatted": True, "EndTimeFormatted": True}
            )
            
            st.plotly_chart(fig)

            st.subheader("Improved Classroom Topic Flow with Timestamps")
            fig = generate_sankey_diagram(transcript,uploaded_file)
            st.plotly_chart(fig)

            # transcribe_and_visualize(uploaded_file)

        os.remove(file_path)

if __name__ == "__main__":
    main()

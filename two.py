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


aai.settings.api_key = "c5e969359a834a4b982f558b622d8edf"

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def demonstrate_clustering_with_audio(file_path, n_clusters=5, show_plot=True):
    
    # Load audio data from the file
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Extract a combination of acoustic features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)

    # Combine and transpose features for clustering (shape: [frames, features])
    feature_matrix = np.vstack([mfcc, chroma, contrast]).T

    # Apply Agglomerative Clustering to the feature matrix
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(feature_matrix)

    # Visualize using PCA (optional)
    if show_plot:
        reduced_features = PCA(n_components=2).fit_transform(feature_matrix)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
        plt.title("Agglomerative Clustering of Audio Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.tight_layout()
        plt.show()

    return {"labels": labels, "features": feature_matrix}







def transcribe_audio(file_path,speakers_expected):
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
   transcript = transcriber.transcribe(file_path)
   return transcript
    
    # transcriber = aai.Transcriber(config=config)
    # transcript = transcriber.transcribe(file_path)
    # return transcript

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


def process_transcription(transcript,file_url, api_key, speakers_expected):
    st.subheader("📢 Processing Audio File...")
    
    # Set API key
    # aai.settings.api_key = api_key

    # # Configure transcription settings
    # config = aai.TranscriptionConfig(
    #     speech_model=aai.SpeechModel.best,
    #     summarization=True,
    #     summary_model=aai.SummarizationModel.informative,
    #     iab_categories=True,
    #     auto_highlights=True,
    #     sentiment_analysis=True,
    #     entity_detection=True,
    #     speaker_labels=True,
    #     language_detection=True,
    #     speakers_expected=speakers_expected,
    #     summary_type=aai.SummarizationType.bullets
    # )

    # # Transcribe the file
    # transcriber = aai.Transcriber(config=config)
    # transcript = transcriber.transcribe(file_url)

    # if transcript.status == aai.TranscriptStatus.error:
    #     st.error(f"❌ Error: {transcript.error}")
    #     return

    # Process Speaker Activity
    speaker_activity = []
    for utterance in transcript.utterances:
        speaker_activity.append({
            "Speaker": utterance.speaker,
            "StartTime": utterance.start // 1000,
            "EndTime": utterance.end // 1000,
            "Duration": (utterance.end - utterance.start) // 1000
        })
    df_speaker = pd.DataFrame(speaker_activity)

    # Convert timestamps
    df_speaker["StartTime"] = pd.to_datetime(df_speaker["StartTime"], unit="s")
    df_speaker["EndTime"] = pd.to_datetime(df_speaker["EndTime"], unit="s")

    # Create Speaker Timeline
    # fig1 = px.timeline(
    #     df_speaker, x_start="StartTime", x_end="EndTime", y="Speaker",
    #     color="Speaker", title="🎤 Speaker Activity Timeline",
    #     color_discrete_sequence=px.colors.qualitative.Dark24
    # )
    # fig1.update_layout(xaxis_title="Time", yaxis_title="Speakers", hovermode="x", plot_bgcolor="white")


    #changes
    fig1 = px.timeline(
     df_speaker, x_start="StartTime", x_end="EndTime", y="Speaker",
     color="Speaker", title="🎤 Speaker Activity Timeline",
     color_discrete_sequence=px.colors.qualitative.Dark24
)
    fig1.update_layout(
     xaxis_title="Time",
     yaxis_title="Speakers",
     hovermode="x",
     plot_bgcolor="white",
     xaxis=dict(
        tickformat="%H:%M:%S"  # Ensures hover timestamp is formatted as HH:MM:SS instead of a full date
    )
)
   
    # Process Topic Analysis
    topic_timestamps, topic_mapping, topic_count = [], {}, {}
    for result in transcript.iab_categories.results:
        start_time = result.timestamp.start // 1000
        end_time = result.timestamp.end // 1000
        for label in result.labels:
            topic = label.label.split(">")[-1]
            topic_timestamps.append((topic, start_time, end_time))
            topic_mapping[topic] = label.label
            topic_count[topic] = topic_count.get(topic, 0) + 1

    df_topic = pd.DataFrame(topic_timestamps, columns=["Topic", "StartTime", "EndTime"])
    df_topic = df_topic.sort_values("StartTime")
    top_topics = sorted(topic_count, key=topic_count.get, reverse=True)[:10]
    df_topic = df_topic[df_topic["Topic"].isin(top_topics)]

    # Sankey Diagram for Topic Flow
    labels = ["Start"] + top_topics + ["End"]
    source, target, values, timestamps = [], [], [], []
    prev_topic = "Start"

    def format_time(seconds):
        return f"{seconds // 60:02}:{seconds % 60:02}"

    for _, row in df_topic.iterrows():
        current_topic = row["Topic"]
        if current_topic in labels:
            source.append(labels.index(prev_topic))
            target.append(labels.index(current_topic))
            values.append(1)
            timestamps.append(format_time(row["StartTime"]))
            prev_topic = current_topic

    source.append(labels.index(prev_topic))
    target.append(labels.index("End"))
    values.append(1)
    timestamps.append(format_time(row["EndTime"]))
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#F4A226",
          "#6A0DAD", "#2E8B57", "#1E90FF", "#8B0000", "#FFD700"]

    fig2 = go.Figure(go.Sankey(
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
        hovertemplate=(
            "From: <b>%{source.customdata}</b> → "
            "To: <b>%{target.customdata}</b><br>"
            "Time: %{customdata}"
        ),  # Displays time in MM:SS
        color=['rgba(0, 0, 255, 0.3)'] * len(source),
    )
))
    fig2.update_layout(
    plot_bgcolor="white",  # Set the plot background to white
    paper_bgcolor="white",  # Set the overall figure background to white
    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins for a clean look
)

    # Bar Chart for Topic Timeline
    df_topic["StartTimeFormatted"] = df_topic["StartTime"].apply(format_time)
    df_topic["EndTimeFormatted"] = df_topic["EndTime"].apply(format_time)

    # fig3 = px.bar(
    #     df_topic, x="StartTime", y="Topic", orientation="h", text="StartTimeFormatted",
    #     title="📊 Topic Discussion Timeline (Top 10 Topics)", color="Topic"
    # )
    # fig3.update_traces(hovertemplate="<b>Topic:</b> %{y}<br><b>Start:</b> %{customdata[0]}<br><b>End:</b> %{customdata[1]}")
    # # fig3.update_layout(xaxis_title="Time (seconds)", yaxis_title="Topics", plot_bgcolor="white", height=600)
    # fig3.update_layout(
    #  xaxis_title="Time (seconds)",
    #  yaxis_title="Topics",
    #  plot_bgcolor="white",
    #  yaxis=dict(
    #     title="Topics",
    #     categoryorder="total ascending",  # Order topics from least to most discussed
    #     tickmode="linear",
    #     dtick=1  # Ensure topics are spaced out evenly
    #  ),
    #  height=600,  # Increase height for better visibility
    #  margin=dict(l=150, r=50, t=50, b=50)  # Adjust margins for better readability
    # )


    #changes 3

    fig3 = px.bar(
    df_topic, x="StartTime", y="Topic", orientation="h", text="StartTimeFormatted",
    title="📊 Topic Discussion Timeline (Top 10 Topics)", color="Topic"
)

# Remove custom data for hover and use timestamp directly
    fig3.update_traces(hovertemplate="<b>Topic:</b> %{y}<br><b>Time:</b> %{text}")

    fig3.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Topics",
    plot_bgcolor="white",
    yaxis=dict(
        title="Topics",
        categoryorder="total ascending",  # Order topics from least to most discussed
        tickmode="linear",
        dtick=1  # Ensure topics are spaced out evenly
    ),
    height=600,  # Increase height for better visibility
    margin=dict(l=150, r=50, t=50, b=50)  # Adjust margins for better readability
)


    top_5_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:5]
    topics, counts = zip(*top_5_topics)  # Unpack topics and their respective counts

# Create pie chart using Plotly
    fig4 = px.pie(
    names=topics,
    values=counts,
    title="🎤 Top 5 Discussed Topics in Classroom Audio",
    color_discrete_sequence=px.colors.qualitative.Set2
    )

# Display in Streamlit
    # st.plotly_chart(fig4, use_container_width=True)

      

    # Display in Streamlit
    st.subheader("📜 Transcription Summary")
    st.write(transcript.summary)

    st.subheader("📌 Speaker Activity")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📌 Topic Flow")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Topic Discussion Timeline")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📌 Top 5 topics discussed in class")
    st.plotly_chart(fig4, use_container_width=True)
    # st.subheader("📈 Sentiment Analysis")
    # for result in transcript.sentiment_analysis:
    #     st.write(f"💬 **{result.text}** - Sentiment: **{result.sentiment}**")




def main():
    st.title("Audio Interactivity & Teaching Style Analysis")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    total_strength = st.number_input("Enter total class strength", min_value=1, step=1)
    
    if uploaded_file is not None and total_strength:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Processing... This may take a while.")
        transcript = transcribe_audio(file_path,total_strength)
        
        if transcript.status == aai.TranscriptStatus.error:
            st.error("Error in transcription: " + transcript.error)
        else:
            results = calculate_interactivity_score(transcript, total_strength)
            # topic_timestamps, topic_count = extract_topics(transcript)
            
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
            # df = pd.DataFrame(topic_timestamps, columns=["Topic", "StartTime", "EndTime"])
            # df = df.sort_values("StartTime")
            
            # # Get the top 10 most discussed topics
            # top_10_topics = sorted(topic_count, key=topic_count.get, reverse=True)[:10]
            # df = df[df["Topic"].isin(top_10_topics)]
            
            # # Format timestamps for readability
            # df["StartTimeFormatted"] = df["StartTime"].apply(lambda x: f"{x//60:02}:{x%60:02}")
            # df["EndTimeFormatted"] = df["EndTime"].apply(lambda x: f"{x//60:02}:{x%60:02}")
            
            # # Create Interactive Bar Chart for Timeline
            # fig = px.bar(
            #     df,
            #     x="StartTime",
            #     y="Topic",
            #     orientation="h",
            #     text="StartTimeFormatted",
            #     title="📊 Interactive Topic Discussion Timeline (Top 10 Topics)",
            #     color="Topic",
            #     labels={"StartTime": "Start Time (seconds)", "Topic": "Topic Discussed"},
            #     hover_data={"StartTimeFormatted": True, "EndTimeFormatted": True}
            # )
            
            # st.plotly_chart(fig)

            # st.subheader("Improved Classroom Topic Flow with Timestamps")
            # fig = generate_sankey_diagram(transcript,uploaded_file)
            # st.plotly_chart(fig)
             

            # transcribe_and_visualize(uploaded_file)
            api_key="85e63f64ed954279bff9a588c0bd2f2f"
            process_transcription(transcript,file_path, api_key,total_strength)

        os.remove(file_path)

if __name__ == "__main__":
    main()

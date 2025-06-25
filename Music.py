import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import altair as alt 
import joblib

from PIL import Image, ImageOps


from SongMatcher.songs import get_trending_playlist_data, get_access_token, hybrid_recommendations
from SongMatcher.config import CLIENT_ID,CLIENT_SECRET
from sklearn.preprocessing import MinMaxScaler


selected = option_menu(
    menu_title="MatchMaker : Music Suggestion ",
    options=["Spotify","Text"],
    icons=["music-note","card-text"],
    menu_icon="music-note-beamed",
    orientation="vertical"
)

if selected == "Spotify":
    CLIENT_ID = CLIENT_ID
    CLIENT_SECRET = CLIENT_SECRET
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    
    st.title("Spotify")
    status = False
    input_song_name = None
    suggestion = False

    playlist_id = st.text_input('Enter your playlist id:')
    if playlist_id:
        music_df, status = get_trending_playlist_data(playlist_id, access_token)
        if status:
            st.write("Tracks in Playlist:")
            for index, row in music_df.iterrows():
                st.write(f"{index + 1}. {row['Track Name']}")
        else:
            st.write("Failed to retrieve playlist data.")

        scaler = MinMaxScaler()
        music_features = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',  'Tempo']].values
        music_features_scaled = scaler.fit_transform(music_features)

        input_song_name = st.text_input('Submit a song to discover similar tracks : ')
        if input_song_name:
            recommendations, suggestion = hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5)
            if suggestion:
                st.write("Recommended Track names:")
                for index, row in recommendations.iterrows():
                    st.write(f'{row["Track Name"]}')

if selected == "Text":
    pipe_lr = joblib.load(open("C:\\Users\\prath\\OneDrive\\Desktop\\Project-SEM-4\\PYTHON\\MusicMatchmaker\\MusicMatchmaker\\Text\\model\\text_emotion.pkl", "rb"))

    def predict_emotions(docx):
        results = pipe_lr.predict([docx])
        return results[0]

    def get_prediction_proba(docx):
        results = pipe_lr.predict_proba([docx])
        return results
    
    # Emotion Detection

    Music_Player = pd.read_csv("C:\\Users\\prath\\OneDrive\\Desktop\\Project-SEM-4\\PYTHON\\MusicMatchmaker\\MusicMatchmaker\\Text\\data\\data_moods.csv")
    Music_Player = Music_Player[['name','artist','mood','popularity']]

    # Making Songs Recommendations Based on Predicted Class
    def Recommend_Songs(prediction):
        if( prediction=='disgust' or prediction=='shame'):
            Play = Music_Player[Music_Player['mood'] =='Sad' ]
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:5].reset_index(drop=True)
            return Play
   
        if( prediction=='joy' or prediction=='sadness' ):
            Play = Music_Player[Music_Player['mood'] =='Happy' ]
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:5].reset_index(drop=True)
            return Play
   
        if( prediction=='fear' or prediction=='anger' ):
            Play = Music_Player[Music_Player['mood'] =='Calm' ]
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:5].reset_index(drop=True)
            return Play
   
        if( prediction=='surprise' or prediction=='neutral' ):
            Play = Music_Player[Music_Player['mood'] =='Energetic' ]
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:5].reset_index(drop=True)
            return Play
    
    def song_list(songs):
        st.subheader('Songs : ')
        for i in songs["name"]:
            st.write(i)

    def main():
        st.title("Matching Emotion Via Text")
        st.subheader("Enter Text")

        with st.form(key="my_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            songs = Recommend_Songs(prediction)

            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            st.write("{}".format(prediction))


            st.write("Songs According To Your Predicted Mood Are: ")
            song_list(songs)


    if __name__ == "__main__":
        main()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# App Title
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")
st.title("🎵 Spotify Music Hit Prediction App")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Spotify_2023.csv", encoding='utf-8')
        st.write("Original columns found in CSV:")
        st.write(df.columns.tolist())

        df.columns = df.columns.str.replace('[^a-zA-Z0-9]', '_', regex=True)
        st.write("Cleaned column names:")
        st.write(df.columns.tolist())

        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        df = df.dropna(subset=['streams'])

        if len(df) < 100:
            st.error("Not enough valid data after cleaning. Please check your CSV file.")
            return None

        median_streams = df['streams'].median()
        df['hit'] = (df['streams'] > median_streams).astype(int)

        feature_mapping = {
            'danceability': None,
            'valence': None,
            'energy': None,
            'acousticness': None,
            'instrumentalness': None,
            'liveness': None,
            'speechiness': None,
            'bpm': None
        }

        for col in df.columns:
            col_lower = col.lower()
            if 'danceability' in col_lower:
                feature_mapping['danceability'] = col
            elif 'valence' in col_lower:
                feature_mapping['valence'] = col
            elif 'energy' in col_lower:
                feature_mapping['energy'] = col
            elif 'acousticness' in col_lower:
                feature_mapping['acousticness'] = col
            elif 'instrumentalness' in col_lower:
                feature_mapping['instrumentalness'] = col
            elif 'liveness' in col_lower:
                feature_mapping['liveness'] = col
            elif 'speechiness' in col_lower:
                feature_mapping['speechiness'] = col
            elif 'bpm' in col_lower or 'tempo' in col_lower:
                feature_mapping['bpm'] = col

        available_features = {k: v for k, v in feature_mapping.items() if v is not None}
        st.write("Found feature columns:")
        st.write(available_features)

        if len(available_features) == 0:
            st.error("No audio feature columns found. Please check your CSV file.")
            return None

        for feature_name, col_name in available_features.items():
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            df[col_name].fillna(df[col_name].median(), inplace=True)

        feature_cols = list(available_features.values())
        df = df.dropna(subset=feature_cols)

        track_col = None
        artist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'track' in col_lower and 'name' in col_lower:
                track_col = col
            elif 'artist' in col_lower and 'name' in col_lower:
                artist_col = col

        return_cols = feature_cols + ['hit', 'streams']
        if track_col:
            return_cols.append(track_col)
        if artist_col:
            return_cols.append(artist_col)

        result_df = df[return_cols].copy()
        result_df.attrs['feature_mapping'] = available_features
        result_df.attrs['track_col'] = track_col
        result_df.attrs['artist_col'] = artist_col

        return result_df

    except FileNotFoundError:
        st.error("Spotify_2023.csv file not found. Please make sure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None and len(df) > 0:
    feature_mapping = df.attrs.get('feature_mapping', {})
    track_col = df.attrs.get('track_col')
    artist_col = df.attrs.get('artist_col')

    st.subheader("🎧 Dataset Preview")
    st.write(f"Total songs: {len(df)}")
    st.write(f"Number of hits: {df['hit'].sum()} ({(df['hit'].mean()*100):.1f}%)")
    st.dataframe(df.head())

    st.subheader("📊 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📈 Feature Distributions")
    available_features = list(feature_mapping.values())

    if available_features:
        feature_to_plot = st.selectbox("Select feature to visualize", available_features)

        fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x=feature_to_plot, hue='hit', kde=True, ax=ax_dist)
        plt.tight_layout()
        st.pyplot(fig_dist)

        X = df[available_features]
        y = df['hit']

        if len(X) > 50:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)

            st.subheader("🎯 Predict If a Song is a Hit")
            col1, col2 = st.columns(2)

            user_inputs = {}

            with col1:
                if 'danceability' in feature_mapping:
                    col_name = feature_mapping['danceability']
                    user_inputs['danceability'] = st.slider("Danceability", 0.0, 100.0, float(df[col_name].mean()))
                if 'valence' in feature_mapping:
                    col_name = feature_mapping['valence']
                    user_inputs['valence'] = st.slider("Valence (Positivity)", 0.0, 100.0, float(df[col_name].mean()))
                if 'energy' in feature_mapping:
                    col_name = feature_mapping['energy']
                    user_inputs['energy'] = st.slider("Energy", 0.0, 100.0, float(df[col_name].mean()))
                if 'acousticness' in feature_mapping:
                    col_name = feature_mapping['acousticness']
                    user_inputs['acousticness'] = st.slider("Acousticness", 0.0, 100.0, float(df[col_name].mean()))

            with col2:
                if 'instrumentalness' in feature_mapping:
                    col_name = feature_mapping['instrumentalness']
                    user_inputs['instrumentalness'] = st.slider("Instrumentalness", 0.0, 100.0, float(df[col_name].mean()))
                if 'liveness' in feature_mapping:
                    col_name = feature_mapping['liveness']
                    user_inputs['liveness'] = st.slider("Liveness", 0.0, 100.0, float(df[col_name].mean()))
                if 'speechiness' in feature_mapping:
                    col_name = feature_mapping['speechiness']
                    user_inputs['speechiness'] = st.slider("Speechiness", 0.0, 100.0, float(df[col_name].mean()))
                if 'bpm' in feature_mapping:
                    col_name = feature_mapping['bpm']
                    user_inputs['bpm'] = st.slider("BPM (Beats Per Minute)", float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean()))

            if st.button("Predict"):
                input_values = [user_inputs[feature] for feature in feature_mapping if feature in user_inputs]
                if len(input_values) == 0:
                    st.error("No valid features available for prediction.")
                    st.stop()

                input_scaled = scaler.transform([input_values])
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]

                st.success(f"🎶 Prediction: {'Hit' if prediction == 1 else 'Not a Hit'}")
                st.info(f"🔢 Probability of Hit: {probability:.2%}")

                st.subheader("📉 Model Evaluation on Test Set")
                y_pred = model.predict(X_test)
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Hit', 'Hit'], yticklabels=['Not Hit', 'Hit'], ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm)

                st.subheader("🔍 Feature Importance")
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1]
                fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                feature_names = [list(feature_mapping.keys())[i] for i in indices]
                sns.barplot(x=importance[indices], y=feature_names, ax=ax_fi)
                ax_fi.set_title("Feature Importance")
                ax_fi.set_xlabel("Importance Score")
                plt.tight_layout()
                st.pyplot(fig_fi)

            st.subheader("🏆 Top Hits in the Dataset")
            top_hits = df[df['hit'] == 1].sort_values('streams', ascending=False).head(10)

            display_cols = ['streams']
            if track_col and track_col in df.columns:
                display_cols.insert(0, track_col)
            if artist_col and artist_col in df.columns:
                display_cols.insert(-1, artist_col)

            st.dataframe(top_hits[display_cols].reset_index(drop=True))
        else:
            st.error("Not enough data to train the model. Need at least 50 samples.")
    else:
        st.error("No valid feature columns found in the dataset.")
else:
    st.error("Unable to load or process the dataset. Please check your CSV file and column names.")

st.caption("Developed using Streamlit, scikit-learn, and Spotify 2023 dataset")

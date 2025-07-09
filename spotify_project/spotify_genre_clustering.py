# Spotify Genre Grouping and Clustering Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("spotify dataset.csv")

# 1. Data Preprocessing
features = ['danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

# Handle missing values
df.dropna(subset=features, inplace=True)

# 2. Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df['danceability'], kde=True)
plt.title("Danceability Distribution")
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[features])
plt.xticks(rotation=45)
plt.title("Audio Feature Distributions")
plt.show()

# 3. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 5. KMeans Clustering (Elbow Method optional)
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['energy'], y=df['valence'], hue=df['cluster'], palette='Set2')
plt.title("Clusters based on Energy and Valence")
plt.xlabel("Energy")
plt.ylabel("Valence")
plt.show()

# 7. Recommendation Function
def recommend_songs(input_song, df):
    cluster_id = df[df['track_name'].str.lower() == input_song.lower()]['cluster'].values
    if len(cluster_id) == 0:
        return "Song not found. Try another."
    cluster_id = cluster_id[0]
    recommendations = df[df['cluster'] == cluster_id].sample(5)
    return recommendations[['track_name', 'track_artist', 'playlist_genre']]

# Example usage:
print(recommend_songs("Memories - Dillon Francis Remix", df))

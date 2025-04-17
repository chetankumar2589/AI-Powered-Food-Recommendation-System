import gradio as gr
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load CSV
recipe_df = pd.read_csv("recipe_final.csv")
recipe_df.rename(columns={'Unnamed: 0': 'S.no'}, inplace=True)
recipe_df.set_index('S.no', inplace=True)

# Fit Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(recipe_df['ingredients_list'])
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X.toarray())

# Recommend function
def recommend(ingredients):
    input_vec = vectorizer.transform([ingredients])
    distances, indices = knn.kneighbors(input_vec.toarray())
    return recipe_df.iloc[indices[0]][['recipe_name', 'ingredients_list']]

# Gradio UI
demo = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(placeholder="Enter ingredients (e.g. eggs, onion, butter)", lines=2),
    outputs=gr.Dataframe(),
    title="🍲 AI-Powered Food Recommendation System",
    description="Get 5 recipe ideas based on what’s in your kitchen!"
)

demo.launch()

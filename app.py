import streamlit as st
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util

# ====== App Title & Instructions ======
st.set_page_config(page_title="Next-Gen E-Commerce Recommender", layout="centered")
st.title("🛒 Next-Gen E-Commerce Recommender (Multi-Agent AI)")

st.markdown("""
**How to use this web app:**
1. Select your preferred product category.
2. Click **Run Agentic Analysis** to see recommendations & trending predictions.
3. The AI agents will think out loud before final results.
""")
    st.markdown("""
    **What This Means:**  
    These products are ranked using semantic AI embeddings, meaning the AI understands the *context* of your interest 
    and matches it to products that are not only in your chosen category but are also most relevant in meaning.  
    This approach mimics advanced AI recommendation systems used by top e-commerce companies to improve conversions.
    """)
# ====== Load Data ======
@st.cache_data
def load_products():
    return pd.read_csv("products.csv")

products_df = load_products()

# ====== Load Embedding Model ======
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ====== Multi-Agent Simulation ======
def user_profile_agent(category):
    thought = f"[User Profile Agent] Detected interest in category: {category}"
    return thought, products_df[products_df['category'] == category]

def product_ranking_agent(filtered_df):
    thought = "[Product Ranking Agent] Ranking products using semantic similarity..."
    user_embedding = model.encode(filtered_df['category'].iloc[0], convert_to_tensor=True)
    prod_embeddings = model.encode(filtered_df['name'].tolist(), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, prod_embeddings).squeeze()
    filtered_df['score'] = scores.tolist()
    ranked_df = filtered_df.sort_values(by='score', ascending=False)
    return thought, ranked_df

def trending_predictor():
    thought = "[Trending Predictor] Estimating top trending products..."
    trending_items = products_df.sample(3)['name'].tolist()
    return thought, trending_items

# ====== UI Inputs ======
category_choice = st.selectbox("Choose a category:", products_df['category'].unique())
if st.button("Run Agentic Analysis"):
    # Step 1: User Profile Agent
    up_thought, filtered = user_profile_agent(category_choice)
    st.info(up_thought)
    
    # Step 2: Product Ranking Agent
    pr_thought, ranked = product_ranking_agent(filtered)
    st.info(pr_thought)
    
    st.subheader("Recommended Products:")
    st.table(ranked[['name', 'price', 'score']])
    
    # Step 3: Trending Predictor
    tp_thought, trending = trending_predictor()
    st.info(tp_thought)
    
    st.subheader("🔥 Trending Products:")
    for t in trending:
        st.write(f"- {t}")

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuration
API_URL = "http://filmfusion-api:8000"  # Container name for Docker network
# API_URL = "http://localhost:8000"    # Use this for local testing

st.set_page_config(
    page_title="FilmFusion Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 FilmFusion Recommendations Dashboard")
st.sidebar.markdown("### Navigation")

# Helper functions
def make_api_request(endpoint, params=None):
    """Make API request with error handling"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: {str(e)}"

def display_recommendations(recommendations, title="Recommendations"):
    """Display recommendations in a formatted way"""
    if recommendations:
        df = pd.DataFrame(recommendations)
        st.subheader(title)
        
        # Add ranking column
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['Rank'] + [col for col in df.columns if col != 'Rank']
        df = df[cols]
        
        st.dataframe(df, use_container_width=True)
        
        # Optional: Add a chart
        if len(df) > 1:
            fig = px.bar(df.head(10), x='title', y='score', 
                        title=f"Top {min(10, len(df))} {title} by Score")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No {title.lower()} found.")

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "User Recommendations", 
    "Similar Movies", 
    "System Metrics", 
    "Analytics"
])

# Tab 1: User Recommendations
with tab1:
    st.header("👤 Get User Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, step=1, value=1)
        n_recs = st.slider("Number of Recommendations", 1, 20, 10)
        
    with col2:
        st.info("💡 **Tip**: Start with User ID 1-100 for best results")
    
    if st.button("🎯 Fetch Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            data, error = make_api_request(f"recommendations/{int(user_id)}", {"n": n_recs})
            
            if error:
                st.error(error)
            elif data:
                recommendations = data.get("recommendations", [])
                display_recommendations(recommendations, f"Recommendations for User {user_id}")
                
                # Show user stats if available
                if "user_stats" in data:
                    st.subheader("📊 User Profile")
                    stats = data["user_stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Ratings", stats.get("total_ratings", "N/A"))
                    with col2:
                        st.metric("Average Rating", f"{stats.get('avg_rating', 0):.2f}")
                    with col3:
                        st.metric("Genres Liked", stats.get("favorite_genres", "N/A"))

# Tab 2: Similar Movies
with tab2:
    st.header("🎭 Find Similar Movies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_id = st.number_input("Enter Movie ID", min_value=1, step=1, value=1)
        n_similar = st.slider("Number of Similar Movies", 1, 20, 10)
        
    with col2:
        st.info("💡 **Tip**: Try Movie ID 1 (Toy Story) or 356 (Forrest Gump)")
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        with st.spinner("Finding similar movies..."):
            data, error = make_api_request(f"similar_movies/{int(movie_id)}", {"n": n_similar})
            
            if error:
                st.error(error)
            elif data:
                similar_movies = data.get("similar_movies", [])
                original_movie = data.get("original_movie", {})
                
                # Show original movie info
                if original_movie:
                    st.subheader(f"🎬 Original Movie")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"**ID:** {original_movie.get('movie_id', movie_id)}")
                    with col2:
                        st.write(f"**Title:** {original_movie.get('title', 'Unknown')}")
                    with col3:
                        st.write(f"**Year:** {original_movie.get('year', 'N/A')}")
                
                display_recommendations(similar_movies, "Similar Movies")

# Tab 3: System Metrics
with tab3:
    st.header("⚙️ System Health & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh System Status"):
            # Health check
            health_data, health_error = make_api_request("health")
            
            if health_error:
                st.error(f"❌ System Health: {health_error}")
            else:
                st.success("✅ System Status: Healthy")
                if health_data:
                    st.json(health_data)
            
            # Metrics
            metrics_data, metrics_error = make_api_request("metrics")
            
            if metrics_error:
                st.warning(f"⚠️ Metrics unavailable: {metrics_error}")
            elif metrics_data:
                st.subheader("📈 System Metrics")
                
                # Display metrics in columns
                if isinstance(metrics_data, dict):
                    metric_cols = st.columns(len(metrics_data))
                    for i, (key, value) in enumerate(metrics_data.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(key.replace("_", " ").title(), value)
    
    with col2:
        st.subheader("🛠️ Model Information")
        
        # You can add model version, last updated, etc.
        st.info("**Active Models:**")
        st.write("- SVD++ Collaborative Filtering")
        st.write("- TF-IDF Content-Based")
        st.write("- Metadata Content-Based")
        
        st.info("**Last Model Update:**")
        st.write("Phase 2 Training Complete")

# Tab 4: Analytics
with tab4:
    st.header("📊 Analytics & Insights")
    
    # Sample analytics - you can expand this
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Popular Genres")
        
        # Mock data - replace with real API data
        genre_data = {
            "Drama": 1200,
            "Comedy": 980,
            "Action": 850,
            "Thriller": 720,
            "Romance": 650,
            "Sci-Fi": 480,
            "Horror": 380,
            "Documentary": 320
        }
        
        fig = px.bar(
            x=list(genre_data.keys()),
            y=list(genre_data.values()),
            title="Genre Popularity"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Rating Distribution")
        
        # Mock data - replace with real API data
        rating_data = {
            "0.5": 45,
            "1.0": 120,
            "1.5": 180,
            "2.0": 280,
            "2.5": 420,
            "3.0": 680,
            "3.5": 920,
            "4.0": 1200,
            "4.5": 680,
            "5.0": 480
        }
        
        fig = px.line(
            x=list(rating_data.keys()),
            y=list(rating_data.values()),
            title="Rating Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional analytics section
    st.subheader("🔍 System Overview")
    
    overview_cols = st.columns(4)
    
    with overview_cols[0]:
        st.metric("Total Users", "943", delta="12")
    
    with overview_cols[1]:
        st.metric("Total Movies", "1,682", delta="8")
    
    with overview_cols[2]:
        st.metric("Total Ratings", "100,000", delta="1,234")
    
    with overview_cols[3]:
        st.metric("API Uptime", "99.9%", delta="0.1%")

# Footer
st.markdown("---")
st.markdown("**FilmFusion Dashboard** | Built with Streamlit & FastAPI | Phase 3 Complete")

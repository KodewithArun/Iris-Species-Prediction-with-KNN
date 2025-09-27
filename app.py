import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 3rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .header-container p {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Section Cards */
    .section-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Input Form Styling */
    .input-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .input-group {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .input-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    
    /* Button Styling */
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    
    .predict-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
        border: 1px solid #68d391;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .result-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2f855a;
        margin-bottom: 1rem;
    }
    
    .result-species {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 1rem;
    }
    
    /* Info Cards */
    .info-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .info-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    /* Statistics Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #718096;
        margin-top: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .input-grid {
            grid-template-columns: 1fr;
        }
        
        .header-container h1 {
            font-size: 2rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("knn_iris_model.pkl")
        scaler = joblib.load("scaler_iris.pkl")
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

model, scaler, models_loaded = load_model_and_scaler()

# Header Section
st.markdown("""
<div class="header-container">
    <h1>🌸 Iris Species Predictor</h1>
    <p>Professional Machine Learning Classification System</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Model files not found. Please ensure 'knn_iris_model.pkl' and 'scaler_iris.pkl' are in the same directory.")
    st.stop()

# Main Application Layout
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Info", "📈 Dataset"])

with tab1:
    # Input Section
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Input Flower Measurements</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌿 Sepal Measurements")
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.4,
            step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.4,
            step=0.1,
            help="Width of the sepal in centimeters"
        )
    
    with col2:
        st.markdown("### 🌺 Petal Measurements")
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.3,
            step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=0.2,
            step=0.1,
            help="Width of the petal in centimeters"
        )
    
    # Current Values Display
    st.markdown("---")
    st.markdown("### 📋 Current Input Values")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sepal Length", f"{sepal_length} cm")
    with col2:
        st.metric("Sepal Width", f"{sepal_width} cm")
    with col3:
        st.metric("Petal Length", f"{petal_length} cm")
    with col4:
        st.metric("Petal Width", f"{petal_width} cm")
    
    # Prediction Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Predict Species", type="primary", use_container_width=True):
            # Prepare input data
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            predicted_species = prediction[0].title()
            
            # Display results
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">🎯 Prediction Result</div>
                <div class="result-species">{predicted_species}</div>
                <p>The model predicts this flower belongs to the <strong>Iris {predicted_species}</strong> species.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Species descriptions
            species_descriptions = {
                'Setosa': {
                    'emoji': '🌸',
                    'description': 'Characterized by smaller petals and distinctive sepal proportions. Typically found in cooler climates.',
                    'features': 'Short, broad petals with wide sepals'
                },
                'Versicolor': {
                    'emoji': '🌺',
                    'description': 'Medium-sized flowers with balanced proportions. Common in temperate regions.',
                    'features': 'Moderate petal and sepal dimensions'
                },
                'Virginica': {
                    'emoji': '🌹',
                    'description': 'Largest flowers with long petals and sepals. Often found in warmer climates.',
                    'features': 'Long, narrow petals with large sepals'
                }
            }
            
            if predicted_species.lower() in [s.lower() for s in species_descriptions.keys()]:
                species_info = species_descriptions[predicted_species.title()]
                st.info(f"""
                **{species_info['emoji']} About Iris {predicted_species}:**
                
                {species_info['description']}
                
                **Key Features:** {species_info['features']}
                """)

with tab2:
    # Model Information
    st.markdown("### 🤖 Model Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Algorithm Details</div>
            <p><strong>Type:</strong> K-Nearest Neighbors (KNN)</p>
            <p><strong>Version:</strong> Scikit-learn Implementation</p>
            <p><strong>Preprocessing:</strong> StandardScaler Normalization</p>
            <p><strong>Features:</strong> 4 Numerical Features</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Performance Metrics</div>
            <p><strong>Accuracy:</strong> ~95% on test data</p>
            <p><strong>Classes:</strong> 3 (Setosa, Versicolor, Virginica)</p>
            <p><strong>Training Samples:</strong> 120 flowers</p>
            <p><strong>Test Samples:</strong> 30 flowers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Model Advantages</div>
            <ul>
                <li>Simple and interpretable</li>
                <li>No training period required</li>
                <li>Works well with small datasets</li>
                <li>Robust to noisy data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Use Cases</div>
            <ul>
                <li>Botanical classification</li>
                <li>Educational demonstrations</li>
                <li>Pattern recognition studies</li>
                <li>Machine learning tutorials</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Dataset Information
    st.markdown("### 📊 Iris Dataset Overview")
    
    # Dataset statistics
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">150</div>
            <div class="stat-label">Total Samples</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">3</div>
            <div class="stat-label">Species</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">1936</div>
            <div class="stat-label">Year Introduced</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📏 Feature Ranges")
        feature_ranges = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Min (cm)': [4.3, 2.0, 1.0, 0.1],
            'Max (cm)': [7.9, 4.4, 6.9, 2.5],
            'Average (cm)': [5.8, 3.1, 3.8, 1.2]
        })
        st.dataframe(feature_ranges, use_container_width=True)
    
    with col2:
        st.markdown("### 🌸 Species Distribution")
        species_data = pd.DataFrame({
            'Species': ['Setosa', 'Versicolor', 'Virginica'],
            'Samples': [50, 50, 50],
            'Percentage': ['33.3%', '33.3%', '33.3%']
        })
        st.dataframe(species_data, use_container_width=True)
    
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #718096; font-family: Inter, sans-serif;'>
    <p style='font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;'>
        🌸 Iris Species Predictor
    </p>
    <p style='font-size: 0.9rem;'>
        Built with Streamlit • Powered by Scikit-learn • Designed for Professional Use
    </p>
</div>
""", unsafe_allow_html=True)
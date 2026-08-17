import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import geopandas as gpd

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Yellow Taxi Premium Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------
# Premium CSS Overrides
# -------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Force Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    
    h1, h2, h3, h4, h5, h6, span, p, div {
        color: #C9D1D9;
    }
    
    /* Clean, luxurious cards */
    .metric-card {
        background-color: #1A1C20;
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #FBC02D;
    }
    .metric-title {
        color: #8B949E;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 36px;
        font-weight: 700;
        margin: 0;
    }
    .metric-accent {
        color: #FBC02D;
    }
    
    /* Dual AI Predictor Cards */
    .ai-card {
        background: #111315;
        border: 1px solid #2D3139;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .ai-card-winner {
        border: 1px solid #FBC02D;
        background: radial-gradient(circle at top, rgba(251, 192, 45, 0.05), #111315);
    }
    .ai-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background-color: #FBC02D;
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .ai-time {
        font-size: 56px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 16px 0;
    }
    .ai-time-winner {
        color: #FBC02D;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* Custom Scrollbar for side list */
    .scroll-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-container::-webkit-scrollbar-track {
        background: #08090A;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: #2D3139;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Data Loading Engine
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join("data", "yellow_tripdata_2026-01.parquet")
    zone_path = os.path.join("data", "taxi_zone_lookup.csv")
    shp_path = os.path.join("data", "taxi_zones", "taxi_zones.shp")
    
    if not os.path.exists(data_path) or not os.path.exists(zone_path) or not os.path.exists(shp_path):
        return pd.DataFrame(), pd.DataFrame(), None
        
    df = pd.read_parquet(data_path, columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'trip_distance', 'total_amount', 'tip_amount', 'payment_type'])
    zones_df = pd.read_csv(zone_path)
    zones_gdf = gpd.read_file(shp_path)
    
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day_name'] = df['tpep_pickup_datetime'].dt.day_name()
    
    return df, zones_df, zones_gdf

@st.cache_resource
def load_models():
    cb_path = os.path.join("models", "catboost_trip_duration.pkl")
    lr_path = os.path.join("models", "linear_regression_trip_duration.pkl")
    cb_model = joblib.load(cb_path) if os.path.exists(cb_path) else None
    lr_model = joblib.load(lr_path) if os.path.exists(lr_path) else None
    return cb_model, lr_model

with st.spinner("Initializing Enterprise Data Engine (3.7M Records)..."):
    df, zones_df, zones_gdf = load_data()
    cb_model, lr_model = load_models()

if zones_gdf is not None:
    # Required for Mapbox plotting
    zones_wgs84 = zones_gdf.to_crs(epsg=4326).copy()
    # Required for CatBoost Euclidean Distance calculations
    zones_projected = zones_gdf.to_crs(epsg=2263).copy()
    centroid_x = zones_projected.set_index("LocationID").geometry.centroid.x
    centroid_y = zones_projected.set_index("LocationID").geometry.centroid.y

# -------------------------------------------------------------------
# Header & KPIs
# -------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #111315 0%, #1A1C20 100%); padding: 40px; border-radius: 16px; border: 1px solid #2D3139; margin-bottom: 30px; position: relative; overflow: hidden;">
    <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: radial-gradient(circle, rgba(251, 192, 45, 0.1) 0%, rgba(0,0,0,0) 70%); border-radius: 50%;"></div>
    <h1 style="color: #FFFFFF; font-size: 42px; font-weight: 700; margin: 0; padding: 0;">NYC Yellow Taxi Intelligence</h1>
    <div style="display: flex; align-items: center; margin-top: 12px; gap: 16px;">
        <span style="background-color: rgba(251, 192, 45, 0.15); color: #FBC02D; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 700; letter-spacing: 1px; border: 1px solid rgba(251, 192, 45, 0.3);">🗓️ JAN 1 - JAN 31, 2026</span>
        <span style="color: #8B949E; font-size: 14px; font-weight: 600;">|</span>
        <span style="color: #8B949E; font-size: 14px; font-weight: 600;">ALL 5 BOROUGHS</span>
    </div>
    <p style="color: #8B949E; font-size: 16px; max-width: 800px; margin-top: 20px; line-height: 1.6;">
        Processing over 3.7 million high-frequency transport transactions to uncover spatial routing inefficiencies, temporal demand matrices, and tipping psychology across New York City.
    </p>
</div>
""", unsafe_allow_html=True)

if not df.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #FBC02D;"><div class="metric-title">Total Volume</div><div class="metric-value">{len(df)/1e6:.1f}<span class="metric-accent">M</span></div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #FBC02D;"><div class="metric-title">Gross Revenue</div><div class="metric-value"><span class="metric-accent">$</span>{df["total_amount"].sum()/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #FBC02D;"><div class="metric-title">Avg Distance</div><div class="metric-value">{df["trip_distance"].mean():.1f}<span class="metric-accent"> mi</span></div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #FBC02D;"><div class="metric-title">Zero-Tip Rides</div><div class="metric-value">{len(df[df["tip_amount"]==0]):,}</div></div>', unsafe_allow_html=True)
    
st.write("")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Geospatial Heatmap", "Native Analytics", "AI Routing Benchmark", "System Architecture"])

# ================= TAB 1: GEOSPATIAL MAPBOX =================
with tab1:
    if df.empty or zones_gdf is None:
        st.warning("Data unavailable.")
    else:
        st.markdown("<h3 style='margin-bottom: 20px;'>Macro Level Pickup Density</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns([7, 3])
        
        zone_demand = df['PULocationID'].value_counts().reset_index()
        zone_demand.columns = ['LocationID', 'Trips']
        gdf_mapped = zones_wgs84.merge(zone_demand, on='LocationID', how='left')
        gdf_mapped['Trips'] = gdf_mapped['Trips'].fillna(0)
        gdf_mapped = gdf_mapped.set_index('LocationID')
        
        with c1:
            with st.spinner("Rendering Polygons..."):
                fig_map = px.choropleth_mapbox(
                    gdf_mapped,
                    geojson=gdf_mapped.geometry,
                    locations=gdf_mapped.index,
                    color='Trips',
                    hover_name='zone',
                    hover_data={'Trips': True},
                    color_continuous_scale="Inferno",
                    mapbox_style="carto-darkmatter",
                    zoom=9.2, center={"lat": 40.7306, "lon": -73.9352},
                    opacity=0.75,
                )
                
                fig_map.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0}, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_family="Space Grotesk"
                )
                st.plotly_chart(fig_map, use_container_width=True, height=600, config={'scrollZoom': True})
            
        with c2:
            st.markdown("#### High-Density Nodes")
            st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
            top_zones = gdf_mapped.sort_values(by='Trips', ascending=False).head(20).reset_index()
            for idx, row in top_zones.iterrows():
                st.markdown(f"""
                <div style="background-color: #1A1C20; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #FBC02D;">
                    <span style="font-weight: 600; font-size: 14px;">{row['zone']}</span>
                    <span style="color: #FBC02D; font-weight: 700;">{row['Trips']:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: NATIVE ANALYTICS =================
with tab2:
    st.markdown("### Comprehensive EDA & ML Validation Repository")
    st.markdown("<p style='color: #8B949E; margin-bottom: 30px;'>Pristine Matplotlib renderings exported directly from the central Jupyter pipeline. Browse all generated visualizations below.</p>", unsafe_allow_html=True)
    
    cat_tabs = st.tabs(["Exploratory Data Analysis", "Feature Distributions", "Geospatial Profiles", "Machine Learning Evaluation"])
    
    directories = [
        ("img/EDA/*.png", cat_tabs[0]),
        ("img/distribution/*.png", cat_tabs[1]),
        ("img/geospatial/*.png", cat_tabs[2]),
        ("img/ML/*.png", cat_tabs[3])
    ]
    
    for path_pattern, t in directories:
        with t:
            images = glob.glob(path_pattern)
            if not images:
                st.info("No images found in this category.")
            else:
                cols = st.columns(2)
                for i, img_path in enumerate(images):
                    with cols[i % 2]:
                        name = os.path.basename(img_path).replace('.png', '').replace('_', ' ')
                        st.markdown(f"<div style='text-align: center; margin-bottom: 10px; font-weight: 600; color: #FBC02D;'>{name}</div>", unsafe_allow_html=True)
                        st.image(img_path, use_container_width=True)
                        st.write("")

# ================= TAB 3: PREDICTIVE ENGINE =================
with tab3:
    st.markdown("### AI Trip Estimator")
    st.markdown("<p style='color: #8B949E; margin-bottom: 30px;'>Input trip parameters to generate highly accurate ETA predictions using our advanced routing intelligence.</p>", unsafe_allow_html=True)
    
    if cb_model is None:
        st.error("⚠️ AI Model unavailable.")
    else:
        zone_dict = zones_df.drop_duplicates(subset=['Zone']).set_index('Zone').to_dict(orient='index')
        valid_zones = sorted([z for z in zones_df['Zone'].dropna().unique() if z != 'Unknown'])
        
        with st.form("benchmark_form"):
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                input_pu_zone = st.selectbox("Origin Location", options=valid_zones, index=valid_zones.index('Upper East Side South') if 'Upper East Side South' in valid_zones else 0)
                input_do_zone = st.selectbox("Destination Location", options=valid_zones, index=valid_zones.index('JFK Airport') if 'JFK Airport' in valid_zones else 1)
            with c_f2:
                input_passenger = st.number_input("Passenger Load", min_value=1, max_value=6, value=1)
                input_hour = st.slider("Departure Hour", 0, 23, 17)
            with c_f3:
                day_opts = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                input_day = st.selectbox("Departure Day", options=day_opts)
                day_to_num = {d: i for i, d in enumerate(day_opts)}
                
            submitted = st.form_submit_button("Calculate Estimated Time", type="primary", use_container_width=True)
            
        if submitted:
            pu_id = zone_dict[input_pu_zone]['LocationID']
            do_id = zone_dict[input_do_zone]['LocationID']
            pu_boro = zone_dict[input_pu_zone]['Borough']
            do_boro = zone_dict[input_do_zone]['Borough']
            
            try:
                est_dist = np.sqrt((centroid_x.loc[pu_id] - centroid_x.loc[do_id])**2 + (centroid_y.loc[pu_id] - centroid_y.loc[do_id])**2) / 5280
            except: est_dist = 0.0
                
            cb_df = pd.DataFrame([{
                "pickup_zone": input_pu_zone, "dropoff_zone": input_do_zone,
                "estimated_zone_distance_miles": est_dist, "pickup_hour": input_hour,
                "pickup_dayofweek": day_to_num[input_day], "passenger_count": input_passenger
            }])
            
            cb_pred = cb_model.predict(cb_df)[0]
            
            st.write("")
            col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
            
            with col_center:
                st.markdown(f"""
                <div class="ai-card ai-card-winner" style="padding: 40px; border-radius: 20px;">
                    <div class="ai-badge" style="background-color: #2EA043; color: white;">AI Optimized</div>
                    <div style="color: #8B949E; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">Estimated Trip Duration</div>
                    <div class="ai-time ai-time-winner" style="font-size: 72px; margin: 20px 0;">{cb_pred:.1f} <span style="font-size: 28px; color: #FBC02D;">MIN</span></div>
                    <div style="font-size: 14px; color: #8B949E; display: flex; justify-content: space-between; border-top: 1px solid #2D3139; padding-top: 16px; margin-top: 20px;">
                        <span>📍 {est_dist:.1f} Miles</span>
                        <span>🛣️ Inter-borough: {'Yes' if pu_boro != do_boro else 'No'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ================= TAB 4: ARCHITECTURE =================
with tab4:
    st.markdown("### Data Dictionary")
    st.markdown("<p style='color: #8B949E;'>Formal schema and spatial metadata defining the dataset boundaries.</p>", unsafe_allow_html=True)
    try:
        with open("Dataset_explanation.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except:
        pass

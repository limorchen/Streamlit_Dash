import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Set Streamlit config
st.set_page_config(layout="wide", page_title="Company Insights Dashboard")
st.title("Company Insights Dashboard")

uploaded_file = st.file_uploader("Upload your company data CSV", type=["csv"])

# --- Utility Functions ---
def parse_market_cap(value):
    try:
        value = str(value).strip().replace(",", "").replace("$", "")
        if pd.isna(value) or value.lower() == 'nan':
            return None
        if value.startswith(("<", ">")):
            value = value[1:]
        if "M" in value:
            return float(value.replace("M", "")) * 1e6
        elif "B" in value:
            return float(value.replace("B", "")) * 1e9
        return float(value)
    except Exception as e:
        print(f"Could not parse market cap value: {value} — {e}")
        return None

def reduce_categories(series, top_n=7, other_label="Other"):
    top_categories = series.value_counts().nlargest(top_n).index
    return series.apply(lambda x: x if x in top_categories else other_label)

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)

    exclude_cols = ['Company Name', 'Market Cap']
    cat_cols = [col for col in df.columns if col not in exclude_cols]

    if cat_cols:
        df = (
            df.groupby(cat_cols, dropna=False)
              .agg({'Company Name': lambda x: ', '.join(sorted(set(x.dropna()))),
                    'Market Cap': 'mean'})
              .reset_index()
        )[['Company Name', 'Market Cap'] + cat_cols]

    business_area_map = {
        'Exosome Therapy': 'Exosome-Based Therapy',
        'Exosome Therapeutics': 'Exosome-Based Therapy',
        'Extracellular Vesicles': 'Exosome-Based Therapy',
        'Cell Therapy': 'Cell-Based Therapy',
        'CAR-T Therapy': 'Cell-Based Therapy',
        'Gene Therapy': 'Gene & Nucleic Acid Therapies',
        'mRNA Therapeutics': 'Gene & Nucleic Acid Therapies',
        'siRNA': 'Gene & Nucleic Acid Therapies',
        'Rejuvenation': 'Longevity & Anti-aging',
        'Longevity': 'Longevity & Anti-aging',
    }

    if 'Business Area' in df.columns:
        df['Business Area'] = df['Business Area'].replace(business_area_map)

    for col in cat_cols:
        if df[col].dtype == 'object':
            df[col] = reduce_categories(df[col], top_n=5)

    st.subheader("Filtered Data")
    st.dataframe(df)

    # --- Filters ---
    with st.sidebar:
        st.header("Filters")
        filters = {}
        for col in ['Business Area', 'Location']:
            if col in df.columns:
                filters[col] = st.multiselect(col, options=df[col].dropna().unique())

    filtered_df = df.copy()
    for col, values in filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    # --- Metrics ---
    col1, col2 = st.columns(2)
    col1.metric("Total Companies", len(filtered_df))
    if 'Market Cap' in filtered_df.columns:
        avg_market_cap = filtered_df['Market Cap'].dropna().mean()
        col2.metric("Avg. Market Cap", f"${avg_market_cap:,.0f}")

    # --- Charts ---
    def plot_chart(title, chart_func):
        st.subheader(title)
        fig = chart_func()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if 'Business Area' in filtered_df.columns:
        plot_chart("Company Count by Business Area", lambda: px.bar(
            filtered_df['Business Area'].value_counts().reset_index(names=['Business Area', 'Count']),
            x='Business Area', y='Count'))

    if {'Location', 'Market Cap'}.issubset(filtered_df.columns):
        plot_chart("Market Cap by Location", lambda: px.box(filtered_df, x="Location", y="Market Cap"))

    if {'Stage of development', 'Market Cap'}.issubset(filtered_df.columns):
        plot_chart("Market Cap by Stage of Development", lambda: px.scatter(
            filtered_df.dropna(subset=['Market Cap']),
            x="Stage of development",
            y="Market Cap",
            size="Market Cap",
            color="Stage of development",
            hover_data=["Company Name"] if "Company Name" in df.columns else None))

    if 'product' in filtered_df.columns:
        plot_chart("Company Distribution by Product", lambda: (
            None if filtered_df['product'].dropna().empty else px.pie(
                filtered_df['product'].value_counts().reset_index(names=['product', 'Count']),
                names='product', values='Count', title='Companies by product', hole=0.3)
        ))

    # --- Optional ML Model ---
    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        model_df = df.dropna(subset=['Market Cap'])
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].fillna("Unknown")
            y = model_df['Market Cap']

            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)]
                )),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            st.write(f"Sample prediction result: ${y_pred[0]:,.0f}")
        else:
            st.warning("One or more required columns are missing for prediction.")

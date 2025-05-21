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

def parse_partnerships(partnership_str):
    if pd.isna(partnership_str) or partnership_str == '':
        return []
    partners = []
    for sep in [',', ';']:
        if isinstance(partnership_str, str) and sep in partnership_str:
            partners = [p.strip() for p in partnership_str.split(sep) if p.strip()]
            break
    if not partners and isinstance(partnership_str, str) and partnership_str.strip():
        partners = [partnership_str.strip()]
    return partners

def has_partnerships(partnership_list):
    return len(partnership_list) > 0

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)

    partnership_column = "Notable Partnerships/Deals"
    if partnership_column in df.columns:
        df['Parsed Partnerships'] = df[partnership_column].apply(parse_partnerships)
        df['Has Partnerships'] = df['Parsed Partnerships'].apply(has_partnerships)
        df['Partnership Count'] = df['Parsed Partnerships'].apply(len)

    exclude_cols = ['Company Name', 'Market Cap']
    exclude_from_groupby = exclude_cols + ['Parsed Partnerships', 'Partnership Count', 'Has Partnerships']
    cat_cols = [col for col in df.columns if col not in exclude_from_groupby]

    if cat_cols:
        agg_dict = {
            'Company Name': lambda x: ', '.join(sorted(set(x.dropna()))),
            'Market Cap': 'mean'
        }
        if 'Partnership Count' in df.columns:
            agg_dict['Partnership Count'] = 'mean'
        if 'Has Partnerships' in df.columns:
            agg_dict['Has Partnerships'] = 'mean'
        df = df.groupby(cat_cols, dropna=False).agg(agg_dict).reset_index()

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

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies", len(filtered_df))
        if 'Market Cap' in filtered_df.columns:
            avg_market_cap = filtered_df['Market Cap'].dropna().mean()
            col2.metric("Avg. Market Cap", f"${avg_market_cap:,.0f}")
        if 'Has Partnerships' in filtered_df.columns:
            partnership_ratio = filtered_df['Has Partnerships'].mean() * 100
            col3.metric("% with Partnerships", f"{partnership_ratio:.1f}%")

        def plot_chart(title, chart_func, key=None):
            st.subheader(title)
            fig = chart_func()
            st.plotly_chart(fig, use_container_width=True, key=key)

        if {'Location', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Market Cap by Location",
                lambda: px.box(filtered_df, x="Location", y="Market Cap"),
                key="marketcap_location"
            )

        if {'Stage of development', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Market Cap by Stage of Development",
                lambda: px.scatter(
                    filtered_df.dropna(subset=['Market Cap']),
                    x="Stage of development",
                    y="Market Cap",
                    size="Market Cap",
                    color="Stage of development",
                    hover_data=["Company Name"] if "Company Name" in df.columns else None),
                key="marketcap_stage"
            )

        if 'Has Partnerships' in filtered_df.columns and 'Market Cap' in filtered_df.columns:
            plot_chart(
                "Partnership Impact on Market Cap",
                lambda: px.box(
                    filtered_df.dropna(subset=['Market Cap']),
                    x="Has Partnerships",
                    y="Market Cap",
                    color="Has Partnerships",
                    labels={
                        "Has Partnerships": "Has Notable Partnerships",
                        "Market Cap": "Market Cap ($)"
                    },
                    title="Market Cap Comparison: Companies With vs. Without Partnerships"
                ),
                key="partnership_impact"
            )

        # --- NEW: Heatmap of Business Area Type vs. Notable Partnerships ---
        if 'Business Area' in filtered_df.columns and 'Parsed Partnerships' in filtered_df.columns:
            st.subheader("Heatmap: Business Area Type vs. Notable Partnerships")
            df_exploded = filtered_df.explode('Parsed Partnerships').dropna(subset=['Parsed Partnerships'])
            heatmap_data = pd.crosstab(
                df_exploded['Business Area'],
                df_exploded['Parsed Partnerships'],
                normalize='index'
            ) * 100
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Partnership", y="Business Area", color="Percentage (%)"),
                color_continuous_scale='YlGnBu',
                title='Partnership Types by Business Area (%)'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        model_df = df.dropna(subset=['Market Cap'])
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if 'Has Partnerships' in model_df.columns:
            required_cols.append('Has Partnerships')
        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].copy()
            for col in X.columns:
                if X[col].dtype != 'object':
                    X[col] = X[col].astype(str)
            X = X.fillna("Unknown")
            y = model_df['Market Cap']
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)])),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            st.write(f"Sample prediction result: ${y_pred[0]:,.0f}")
        else:
            st.error("One or more required columns are missing for prediction.")
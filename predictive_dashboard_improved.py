import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Streamlit app setup
st.set_page_config(layout="wide", page_title="Company Insights Dashboard")
st.title("Company Insights Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your company data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    def parse_market_cap(value):
        try:
            value = str(value).strip().replace(",", "").replace("$", "")
            if pd.isna(value) or value.lower() == 'nan' or not value:
                return None
            if value.startswith(("<", ">")):
                value = value[1:]
            if "M" in value:
                return float(value.replace("M", "")) * 1e6
            elif "B" in value:
                return float(value.replace("B", "")) * 1e9
            return float(value)
        except Exception:
            return None

    def parse_partnerships(partnership_str):
        if pd.isna(partnership_str) or partnership_str == '':
            return []
        for sep in [',', ';']:
            if isinstance(partnership_str, str) and sep in partnership_str:
                return [p.strip() for p in partnership_str.split(sep) if p.strip()]
        return [partnership_str.strip()] if isinstance(partnership_str, str) else []

    def has_partnerships(partnership_list):
        return len(partnership_list) > 0

    def reduce_categories(series, top_n=7, other_label="Other"):
        top_categories = series.value_counts().nlargest(top_n).index
        return series.apply(lambda x: x if x in top_categories else other_label)

    df['Market Cap'] = df['Market Cap'].apply(parse_market_cap) if 'Market Cap' in df.columns else None
    if 'Notable Partnerships/Deals' in df.columns:
        df['Parsed Partnerships'] = df['Notable Partnerships/Deals'].apply(parse_partnerships)
        df['Has Partnerships'] = df['Parsed Partnerships'].apply(has_partnerships)
        df['Partnership Count'] = df['Parsed Partnerships'].apply(len)

    exclude_cols = ['Company Name', 'Market Cap', 'Parsed Partnerships', 'Partnership Count', 'Has Partnerships']
    cat_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']

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
        df[col] = reduce_categories(df[col], top_n=5)

    st.subheader("Filtered Data")
    st.dataframe(df)

    # Sidebar filters
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
                    labels={"Has Partnerships": "Has Notable Partnerships", "Market Cap": "Market Cap ($)"},
                ),
                key="partnership_impact"
            )

        if 'product' in filtered_df.columns:
            product_counts = filtered_df['product'].value_counts().reset_index()
            product_counts.columns = ['product', 'Count']
            plot_chart(
                "Company Distribution by Product",
                lambda: px.pie(
                    product_counts,
                    names='product',
                    values='Count',
                    title='Companies by product',
                    hole=0.3
                ) if not filtered_df['product'].dropna().empty else px.pie(names=['No data'], values=[1], title='No product data available'),
                key="product_distribution"
            )

    # --- Heatmap: Business Area vs. Notable Partnerships ---
    if 'Business Area' in df.columns and 'Parsed Partnerships' in df.columns:
        st.subheader("Heatmap: Business Area vs. Notable Partnerships")

        df_heatmap = df.copy()
        df_heatmap['Business Area'] = df_heatmap['Business Area'].astype(str).str.split(',')
        df_heatmap['Business Area'] = df_heatmap['Business Area'].apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else [])

        df_heatmap['Parsed Partnerships'] = df_heatmap['Parsed Partnerships'].astype(str).str.split(',')
        df_heatmap['Parsed Partnerships'] = df_heatmap['Parsed Partnerships'].apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else [])

        df_exploded = df_heatmap.explode('Business Area').explode('Parsed Partnerships')
        df_exploded = df_exploded.dropna(subset=['Business Area', 'Parsed Partnerships'])

        if not df_exploded.empty:
            heatmap_data = pd.crosstab(
                df_exploded['Business Area'],
                df_exploded['Parsed Partnerships'],
                normalize='index'
            ) * 100

            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Partnership Type", y="Business Area", color="Percentage (%)"),
                color_continuous_scale='YlGnBu',
                title='Distribution of Notable Partnerships Across Business Areas (%)'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available to display the Business Area vs. Partnership heatmap.")










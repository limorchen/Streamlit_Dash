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
    """Parse partnership string into a list of partners"""
    if pd.isna(partnership_str) or partnership_str == '':
        return []
    
    # Handle different separators and clean up
    partners = []
    for sep in [',', ';']:
        if sep in partnership_str:
            partners = [p.strip() for p in partnership_str.split(sep) if p.strip()]
            break
    
    # If no separators found, treat as a single partner
    if not partners and partnership_str.strip():
        partners = [partnership_str.strip()]
        
    return partners

def has_partnerships(partnership_list):
    """Check if a company has any partnerships"""
    return len(partnership_list) > 0

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)
    
    # Process Notable partnerships if the column exists
    if 'Notable partnerships' in df.columns:
        df['Parsed Partnerships'] = df['Notable partnerships'].apply(parse_partnerships)
        df['Has Partnerships'] = df['Parsed Partnerships'].apply(has_partnerships)
        df['Partnership Count'] = df['Parsed Partnerships'].apply(len)

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

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies", len(filtered_df))
        if 'Market Cap' in filtered_df.columns:
            avg_market_cap = filtered_df['Market Cap'].dropna().mean()
            col2.metric("Avg. Market Cap", f"${avg_market_cap:,.0f}")
        if 'Has Partnerships' in filtered_df.columns:
            partnership_ratio = filtered_df['Has Partnerships'].mean() * 100
            col3.metric("% with Partnerships", f"{partnership_ratio:.1f}%")

        # --- Charts ---
        def plot_chart(title, chart_func, key=None):
            st.subheader(title)
            fig = chart_func()
            st.plotly_chart(fig, use_container_width=True, key=key)

        if 'Business Area' in filtered_df.columns:
            vc = filtered_df['Business Area'].value_counts()
            ba_counts = pd.DataFrame({
                'Business Area': vc.index,
                'Count': vc.values
            })
            st.write("Business Area counts DataFrame:", ba_counts.head())
            plot_chart(
                "Company Count by Business Area",
                lambda: px.bar(ba_counts, x='Business Area', y='Count'),
                key="business_area_chart"
            )

        # --- Business Area vs Partnerships Analysis ---
        if {'Business Area', 'Partnership Count'}.issubset(filtered_df.columns):
            st.subheader("Business Area vs. Partnerships Analysis")
            
            # Average partnerships by business area
            partnership_by_area = filtered_df.groupby('Business Area')['Partnership Count'].agg(
                ['mean', 'count']).reset_index()
            partnership_by_area.columns = ['Business Area', 'Avg Partnerships', 'Company Count']
            partnership_by_area = partnership_by_area.sort_values('Avg Partnerships', ascending=False)
            
            # Only include areas with at least 2 companies for better reliability
            reliable_data = partnership_by_area[partnership_by_area['Company Count'] >= 2]
            
            if not reliable_data.empty:
                fig = px.bar(
                    reliable_data,
                    x='Business Area',
                    y='Avg Partnerships',
                    color='Company Count',
                    labels={
                        'Avg Partnerships': 'Average Number of Partnerships',
                        'Business Area': 'Business Area',
                        'Company Count': 'Number of Companies'
                    },
                    title='Average Partnerships by Business Area'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a heatmap showing the distribution of partnership counts
                if len(filtered_df['Business Area'].unique()) > 1:
                    # Create partnership bins
                    filtered_df['Partnership Bins'] = pd.cut(
                        filtered_df['Partnership Count'], 
                        bins=[0, 1, 3, 5, float('inf')],
                        labels=['0', '1-3', '4-5', '6+']
                    )
                    
                    # Create a crosstab for the heatmap
                    heatmap_data = pd.crosstab(
                        filtered_df['Business Area'], 
                        filtered_df['Partnership Bins'],
                        normalize='index'
                    ) * 100  # Convert to percentage
                    
                    # Only include business areas with enough data
                    if not heatmap_data.empty:
                        fig = px.imshow(
                            heatmap_data,
                            labels=dict(x="Number of Partnerships", y="Business Area", color="Percentage (%)"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            color_continuous_scale='YlGnBu',
                            title='Partnership Distribution by Business Area (%)'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display the data table
                st.write("Partnership Statistics by Business Area:")
                st.dataframe(partnership_by_area)
            else:
                st.info("Not enough data to create reliable partnership analysis by business area. " +
                       "Each area needs at least 2 companies.")

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

        # --- Additional chart - Partnership influence on Market Cap ---
        if {'Has Partnerships', 'Market Cap'}.issubset(filtered_df.columns):
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

        def empty_pie_chart():
            fig = px.pie(
                names=['No data'],
                values=[1],
                title='No product data available'
            )
            fig.update_traces(textinfo='none')  # Optional: hide labels
            return fig

        if 'product' in filtered_df.columns:
            product_counts = filtered_df['product'].value_counts().reset_index()
            product_counts.columns = ['product', 'Count']

            plot_chart(
                "Company Distribution by Product",
                lambda: px.pie(
                    product_counts,
                    names='product', values='Count', title='Companies by product', hole=0.3
                ) if not filtered_df['product'].dropna().empty else empty_pie_chart(),
                key="product_distribution"
            )


    # --- Optional ML Model ---
    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        model_df = df.dropna(subset=['Market Cap'])
        
        # Add partnership indicator to the model if available
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if 'Has Partnerships' in model_df.columns:
            required_cols.append('Has Partnerships')
            
        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].copy()
            # Convert any non-object columns that shouldn't be one-hot encoded
            for col in X.columns:
                if X[col].dtype != 'object':
                    X[col] = X[col].astype(str)
            X = X.fillna("Unknown")
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
            st.error("One or more required columns are missing for prediction.")

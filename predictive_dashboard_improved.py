import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Set Streamlit config for wider layout and page title
st.set_page_config(layout="wide", page_title="Company Insights Dashboard")
st.title("Company Insights Dashboard")

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload your company data CSV", type=["csv"])

# --- Utility Functions ---

def parse_market_cap(value):
    """
    Parses market capitalization strings (e.g., "$1.2B", "500M") into float numbers.
    Handles missing values and various formats.
    """
    try:
        # Clean the string by removing commas and dollar signs, and stripping whitespace
        value = str(value).strip().replace(",", "").replace("$", "")
        # Return None for NaN or empty string values
        if pd.isna(value) or value.lower() == 'nan' or not value:
            return None
        # Remove leading "<" or ">" characters if present
        if value.startswith(("<", ">")):
            value = value[1:]
        # Convert "M" (million) and "B" (billion) suffixes to numerical values
        if "M" in value:
            return float(value.replace("M", "")) * 1e6
        elif "B" in value:
            return float(value.replace("B", "")) * 1e9
        # Return as float if no suffix
        return float(value)
    except Exception as e:
        # Print error for unparseable values and return None
        print(f"Could not parse market cap value: {value} — {e}")
        return None

def reduce_categories(series, top_n=7, other_label="Other"):
    """
    Reduces the number of unique categories in a Series by grouping less frequent
    ones into an 'Other' category.
    """
    # Get the top N most frequent categories
    top_categories = series.value_counts().nlargest(top_n).index
    # Apply the reduction: keep top categories, mark others as 'Other'
    return series.apply(lambda x: x if x in top_categories else other_label)

def parse_partnerships(partnership_str):
    """
    Parses a string of partnerships, splitting by common delimiters (comma, semicolon).
    Returns a list of cleaned partnership strings.
    """
    if pd.isna(partnership_str) or partnership_str == '':
        return []
    partners = []
    # Try splitting by comma or semicolon
    for sep in [',', ';']:
        if isinstance(partnership_str, str) and sep in partnership_str:
            partners = [p.strip() for p in partnership_str.split(sep) if p.strip()]
            break # Stop after the first successful split
    # If no delimiters found, treat the whole string as a single partnership
    if not partners and isinstance(partnership_str, str) and partnership_str.strip():
        partners = [partnership_str.strip()]
    return partners

st.write("Parsed column sample:")
st.dataframe(df[['Notable Partnerships/Deals', 'Parsed Partnerships']].head(10))


def has_partnerships(partnership_list):
    """Checks if a list of partnerships is not empty."""
    return len(partnership_list) > 0

def empty_pie_chart():
    """Returns a placeholder pie chart for when no product data is available."""
    fig = px.pie(
        names=['No data'],
        values=[1],
        title='No product data available'
    )
    fig.update_traces(textinfo='none') # Hide text info for placeholder
    return fig

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Process 'Market Cap' column if it exists
    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)

    # Process 'Notable Partnerships/Deals' column if it exists
    partnership_column = "Notable Partnerships/Deals"
    if partnership_column in df.columns:
        df['Parsed Partnerships'] = df[partnership_column].apply(parse_partnerships)
        df['Has Partnerships'] = df['Parsed Partnerships'].apply(has_partnerships)
        df['Partnership Count'] = df['Parsed Partnerships'].apply(len)

    # Define columns to exclude from general processing (like groupby)
    exclude_cols = ['Company Name', 'Market Cap']
    exclude_from_groupby = exclude_cols + ['Parsed Partnerships', 'Partnership Count', 'Has Partnerships']
    # Identify categorical columns for grouping
    cat_cols = [col for col in df.columns if col not in exclude_from_groupby]

    # Group data by categorical columns to aggregate company information
    if cat_cols:
        agg_dict = {
            'Company Name': lambda x: ', '.join(sorted(set(x.dropna()))), # Aggregate company names
            'Market Cap': 'mean' # Calculate mean market cap
        }
        # Add partnership aggregations if columns exist
        if 'Partnership Count' in df.columns:
            agg_dict['Partnership Count'] = 'mean'
        if 'Has Partnerships' in df.columns:
            agg_dict['Has Partnerships'] = 'mean'
        # Perform groupby and reset index to make grouped columns regular columns
        df = df.groupby(cat_cols, dropna=False).agg(agg_dict).reset_index()

    # Map similar business area names to a standardized format
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

    # Apply the business area mapping
    if 'Business Area' in df.columns:
        df['Business Area'] = df['Business Area'].replace(business_area_map)

    # Reduce categories for all identified categorical columns to manage cardinality
    for col in cat_cols:
        if df[col].dtype == 'object': # Only apply to object (string) columns
            df[col] = reduce_categories(df[col], top_n=5)

    # Display the processed (filtered and grouped) data
    st.subheader("Filtered Data")
    st.dataframe(df)

    # --- Filters ---
    # Sidebar for interactive filtering
    with st.sidebar:
        st.header("Filters")
        filters = {}
        # Create multiselect filters for 'Business Area' and 'Location'
        for col in ['Business Area', 'Location']:
            if col in df.columns:
                filters[col] = st.multiselect(col, options=df[col].dropna().unique())

    # Apply selected filters to create a filtered DataFrame
    filtered_df = df.copy()
    for col, values in filters.items():
        if values: # Apply filter only if values are selected
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    # Display metrics or a warning if no data matches filters
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies", len(filtered_df))
        if 'Market Cap' in filtered_df.columns:
            avg_market_cap = filtered_df['Market Cap'].dropna().mean()
            col2.metric("Avg. Market Cap", f"${avg_market_cap:,.0f}")
        if 'Has Partnerships' in filtered_df.columns:
            partnership_ratio = filtered_df['Has Partnerships'].mean() * 100
            col3.metric("% with Partnerships", f"{partnership_ratio:.1f}%")

        # Helper function to plot charts
        def plot_chart(title, chart_func, key=None):
            st.subheader(title)
            fig = chart_func()
            st.plotly_chart(fig, use_container_width=True, key=key)

        # Plot Market Cap by Location (Box Plot)
        if {'Location', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Market Cap by Location",
                lambda: px.box(filtered_df, x="Location", y="Market Cap"),
                key="marketcap_location"
            )

        # Plot Market Cap by Stage of Development (Scatter Plot)
        if {'Stage of development', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Market Cap by Stage of Development",
                lambda: px.scatter(
                    filtered_df.dropna(subset=['Market Cap']), # Drop NaN Market Cap for plotting
                    x="Stage of development",
                    y="Market Cap",
                    size="Market Cap", # Size markers by Market Cap
                    color="Stage of development", # Color by stage
                    hover_data=["Company Name"] if "Company Name" in df.columns else None), # Show company name on hover
                key="marketcap_stage"
            )

        # Plot Partnership Impact on Market Cap (Box Plot)
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

        # Plot Company Distribution by Product (Pie Chart)
        if 'product' in filtered_df.columns:
            product_counts = filtered_df['product'].value_counts().reset_index()
            product_counts.columns = ['product', 'Count'] # Rename columns for clarity

            plot_chart(
                "Company Distribution by Product",
                # Use empty_pie_chart if no product data, otherwise create pie chart
                lambda: px.pie(
                    product_counts,
                    names='product', values='Count', title='Companies by product', hole=0.3
                ) if not filtered_df['product'].dropna().empty else empty_pie_chart(),
                key="product_distribution"
            )

        # Heatmap: Business Area Type vs. Notable Partnerships
        if 'Business Area' in filtered_df.columns and 'Parsed Partnerships' in filtered_df.columns:
            st.subheader("Heatmap: Business Area Type vs. Notable Partnerships")
            # Explode the 'Parsed Partnerships' list into separate rows, then drop NaNs
            df_exploded = filtered_df.explode('Parsed Partnerships').dropna(subset=['Parsed Partnerships'])
            st.write("Exploded shape:", df_exploded.shape)
            st.dataframe(df_exploded.head())

            # Only display heatmap if there is data after exploding and dropping NaNs
            if not df_exploded.empty:
                # Calculate cross-tabulation for heatmap data, normalized by index (Business Area)
                heatmap_data = pd.crosstab(
                    df_exploded['Business Area'],
                    df_exploded['Parsed Partnerships'],
                    normalize='index'
                ) * 100 # Convert to percentage

                # Create and display the heatmap
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Partnership", y="Business Area", color="Percentage (%)"),
                    color_continuous_scale='YlGnBu',
                    title='Distribution of Notable Partnerships Across Business Areas (%)'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Display info message if no data for heatmap
                st.info("No data available to display the Business Area vs. Partnership heatmap.")

    # Simple Market Cap Prediction Model
    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        # Filter out rows with missing 'Market Cap' for model training
        model_df = df.dropna(subset=['Market Cap'])
        # Define required columns for the prediction model
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if 'Has Partnerships' in model_df.columns:
            required_cols.append('Has Partnerships')

        # Check if all required columns are present
        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].copy()
            # Convert non-object columns to string for OneHotEncoder if necessary
            for col in X.columns:
                if X[col].dtype != 'object':
                    X[col] = X[col].astype(str)
            X = X.fillna("Unknown") # Fill any remaining NaNs with 'Unknown'
            y = model_df['Market Cap']

            # Create a pipeline with OneHotEncoder and RandomForestRegressor
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)])),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train) # Train the model
            y_pred = pipeline.predict(X_test) # Make predictions

            # Display a sample prediction result
            st.write(f"Sample prediction result: ${y_pred[0]:,.0f}")
        else:
            st.error("One or more required columns are missing for prediction.")
    else:
        st.info("Market Cap column not found in the uploaded data, cannot train prediction model.")





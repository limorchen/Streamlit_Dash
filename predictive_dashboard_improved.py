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
        # Removed 'Partnership Count' calculation

    exclude_cols = ['Company Name', 'Market Cap', 'Parsed Partnerships', 'Has Partnerships'] # Removed 'Partnership Count' from exclude
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

        # --- Predict Market Cap (More Comprehensive Model) ---
        st.subheader("Predict Market Cap (Including Categorical Features)")

        model_df_full = df.copy().dropna(subset=['Market Cap'])

        if 'Market Cap' in model_df_full.columns and len(model_df_full) >= 10:
            # Select features for the model
            features = [] # Removed 'Partnership Count'
            categorical_features = ['Business Area', 'product', 'Location', 'vertical', 'Stage of development']
            all_available_categorical = [col for col in categorical_features if col in model_df_full.columns]
            features.extend(all_available_categorical)

            model_df_prepared = model_df_full[features + ['Market Cap']].dropna()

            if not model_df_prepared.empty:
                # One-hot encode categorical features
                model_df_encoded = pd.get_dummies(model_df_prepared, columns=all_available_categorical, dummy_na=False)

                X = model_df_encoded.drop(columns=['Market Cap'])
                y = model_df_encoded['Market Cap']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_full = RandomForestRegressor(n_estimators=100, random_state=42)
                model_full.fit(X_train, y_train)

                st.markdown("### Predict Market Cap from Categorical Features") # Updated title
                user_inputs_full = {}

                # Categorical features
                # Categorical features
                for cat_col in all_available_categorical:
                    unique_values = model_df_full[cat_col].dropna().unique()
                    selected_value = st.selectbox(f"Select {cat_col}", options=unique_values, key=f"{cat_col}_full")
                    user_inputs_full[cat_col] = selected_value
                    
                # Prepare input DataFrame for prediction
                input_data = {} # Removed 'Partnership Count'
                for cat_col in all_available_categorical:
                    input_data[cat_col] = [user_inputs_full.get(cat_col, None)]
                input_df_full = pd.DataFrame(input_data)

                # One-hot encode the input
                input_df_encoded = pd.get_dummies(input_df_full, columns=all_available_categorical, dummy_na=False)

                # Ensure the input has the same columns as the training data
                missing_cols = set(X.columns) - set(input_df_encoded.columns)
                for c in missing_cols:
                    input_df_encoded[c] = 0
                input_df_encoded = input_df_encoded[X.columns]

                prediction_full = model_full.predict(input_df_encoded)[0]
                st.success(f"Predicted Market Cap (from categorical features): ${prediction_full:,.0f}") # Updated success message

            else:
                st.warning("Not enough data with selected features and 'Market Cap' to train the model.")

        else:
            st.warning("Not enough data with 'Market Cap' to train a prediction model with more features.")

        # --- Previous Simple Market Cap Prediction (Numerical Only) ---
        st.subheader("Predict Market Cap (Simple Model - Numerical Only)")
        # Prepare dataset: only numeric columns and drop NA
        model_df_numeric = df.select_dtypes(include=[np.number]).dropna()

        if 'Market Cap' in model_df_numeric.columns and len(model_df_numeric) >= 10:
            # Remove 'Partnership Count' if it's numeric
            columns_to_drop = ['Market Cap']
            if 'Partnership Count' in model_df_numeric.columns:
                columns_to_drop.append('Partnership Count')
            X_numeric = model_df_numeric.drop(columns=columns_to_drop, errors='ignore')
            y_numeric = model_df_numeric['Market Cap']

            if not X_numeric.empty: # Check if there are any remaining numerical features
                X_train_numeric, X_test_numeric, y_train_numeric, y_test_numeric = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)
                model_numeric = RandomForestRegressor(n_estimators=100, random_state=42)
                model_numeric.fit(X_train_numeric, y_train_numeric)

                st.markdown("### Predict Market Cap from Numerical Features (Excluding Partnership Count)")
                user_inputs_numeric = {}
                for feature in X_numeric.columns:
                    min_val, max_val = float(X_numeric[feature].min()), float(X_numeric[feature].max())
                    mean_val = float(X_numeric[feature].mean())
                    user_inputs_numeric[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, key=f"{feature}_numeric")

                input_df_numeric = pd.DataFrame([user_inputs_numeric])
                prediction_numeric = model_numeric.predict(input_df_numeric)[0]
                st.success(f"Predicted Market Cap: ${prediction_numeric:,.0f}")
            else:
                st.warning("No remaining numerical features to train the simple prediction model (excluding Market Cap and Partnership Count).")
        else:
            st.warning("Not enough numeric data with 'Market Cap' to train a prediction model.")







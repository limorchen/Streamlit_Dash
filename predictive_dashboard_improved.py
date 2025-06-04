import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(layout="wide", page_title="Company Insights Dashboard")
st.title("Company Insights Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your company data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # --- Helper functions ---
    def parse_market_cap(value):
        try:
            value = str(value).strip().replace(",", "").replace("$", "")
            if not value or value.lower() == 'nan':
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
        if pd.isna(partnership_str) or not partnership_str: # Handle NaN and empty strings
            return []

        # Step 1: Unify delimiters by replacing all semicolons with commas
        temp_str = str(partnership_str).replace(';', ',')

        # Step 2: Split by comma and clean up each part (strip whitespace, remove empty strings)
        parts = [p.strip() for p in temp_str.split(',') if p.strip()]

        return parts

    def reduce_categories(series, top_n=7, other_label="Other"):
        top = series.value_counts().nlargest(top_n).index
        return series.apply(lambda x: x if x in top else other_label)

    # --- Preprocess ---
    if 'Market Cap' in df.columns:
        df['Market Cap'] = df['Market Cap'].apply(parse_market_cap)

    if 'Notable Partnerships/Deals' in df.columns:
        df['Parsed Partnerships'] = df['Notable Partnerships/Deals'].apply(parse_partnerships)
        df['Has Partnerships'] = df['Parsed Partnerships'].apply(lambda x: len(x) > 0)

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

    exclude_cols = ['Company Name', 'Market Cap', 'Parsed Partnerships', 'Has Partnerships']
    cat_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']
    for col in cat_cols:
        df[col] = reduce_categories(df[col], top_n=5)

    st.subheader("Filtered Data")
    st.dataframe(df)

    # --- Filters ---
    with st.sidebar:
        st.header("Filters")
        filters = {
            col: st.multiselect(col, options=df[col].dropna().unique())
            for col in ['Business Area', 'Location'] if col in df.columns
        }

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

        def plot_chart(title, chart_func):
            st.subheader(title)
            st.plotly_chart(chart_func(), use_container_width=True)

        if {'Location', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart("Market Cap by Location", lambda: px.box(filtered_df, x="Location", y="Market Cap"))

        if {'Stage of development', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Market Cap by Stage of Development",
                lambda: px.scatter(
                    filtered_df.dropna(subset=['Market Cap']),
                    x="Stage of development",
                    y="Market Cap",
                    size="Market Cap",
                    color="Stage of development",
                    hover_data=["Company Name"] if "Company Name" in df.columns else None,
                ),
            )

        if {'Has Partnerships', 'Market Cap'}.issubset(filtered_df.columns):
            plot_chart(
                "Partnership Impact on Market Cap",
                lambda: px.box(
                    filtered_df.dropna(subset=['Market Cap']),
                    x="Has Partnerships",
                    y="Market Cap",
                    color="Has Partnerships",
                    labels={"Has Partnerships": "Has Notable Partnerships", "Market Cap": "Market Cap ($)"},
                ),
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
                    hole=0.3
                ) if not product_counts.empty else px.pie(names=['No data'], values=[1]),
            )

        # --- Heatmap ---
        # --- Heatmap ---
        if {'Business Area', 'Parsed Partnerships'}.issubset(df.columns):
            st.subheader("Heatmap: Business Area vs. Notable Partnerships")

            df_heatmap = df.copy()
            # Ensure Business Area is also a list of strings if it can contain multiple
            # If 'Business Area' can contain comma-separated values, keep this.
            # If it's always single values, you can remove this split too, or just make sure it's consistent.
            df_heatmap['Business Area'] = df_heatmap['Business Area'].astype(str).str.split(',')

            # Remove or comment out this line:
            # df_heatmap['Parsed Partnerships'] = df_heatmap['Parsed Partnerships'].astype(str).str.split(',')

            # The 'Parsed Partnerships' column is ALREADY a list of strings from your parse_partnerships function.
            # Just ensure it's treated as a list for explode.
            # If there are any NaN values that might cause issues with explode, handle them (e.g., fill with empty list)
            df_heatmap['Parsed Partnerships'] = df_heatmap['Parsed Partnerships'].apply(lambda x: x if isinstance(x, list) else [])


            df_exploded = df_heatmap.explode('Business Area').explode('Parsed Partnerships')
            df_exploded = df_exploded.dropna(subset=['Business Area', 'Parsed Partnerships'])

            if not df_exploded.empty:
                heatmap_data = pd.crosstab(
                    df_exploded['Business Area'].str.strip(),
                    df_exploded['Parsed Partnerships'].str.strip(),
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
                st.info("No data to display in heatmap.")
        # --- Market Cap Prediction ---
        st.subheader("Predict Market Cap (Including Categorical Features)")
        model_df = df.dropna(subset=['Market Cap'])

        if len(model_df) >= 10:
            cat_features = ['Business Area', 'product', 'Location', 'vertical', 'Stage of development']
            used_features = [col for col in cat_features if col in model_df.columns]
            model_df = model_df[used_features + ['Market Cap']].dropna()

            if not model_df.empty:
                model_df_encoded = pd.get_dummies(model_df, columns=used_features)
                X = model_df_encoded.drop(columns=['Market Cap'])
                y_original = model_df_encoded['Market Cap']
                y_transformed = np.log1p(y_original) # <--- INSERTED LINES

                X_train, X_test, y_train, y_test_transformed = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

                model = XGBRegressor(random_state=42)
                model.fit(X_train, y_train)

                # Predict on the test set (in log scale)
                y_pred_transformed = model.predict(X_test) # Original prediction

                # Inverse transform the predictions
                y_pred = np.expm1(y_pred_transformed) # <--- INSERTED LINE
                y_test = np.expm1(y_test_transformed) # <--- INSERTED LINE

                # Evaluation metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                st.subheader("Model Evaluation Metrics")
                st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.2f}")
                st.write(f"**R-squared (R²):** {r2:.2f}")

                # Visual Diagnostics
                st.subheader("Visual Diagnostics: Actual vs. Predicted Market Cap")
                fig_scatter, ax_scatter = plt.subplots(figsize=(3, 3)) # Pass figsize to subplots
                ax_scatter.scatter(y_test, y_pred)
                ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax_scatter.set_xlabel("Actual Market Cap")
                ax_scatter.set_ylabel("Predicted Market Cap")
                ax_scatter.set_title("Actual vs. Predicted Market Cap (XGBoost)")
                st.pyplot(fig_scatter)

                user_input = {}
                for col in used_features:
                    options = df[col].dropna().unique()
                    user_input[col] = st.selectbox(f"Select {col}", options=options)

                input_df = pd.DataFrame([user_input])
                input_encoded = pd.get_dummies(input_df, columns=used_features)

                # Align columns
                for col in X.columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[X.columns]

                predicted_log = model.predict(input_encoded)[0]
                prediction = np.expm1(predicted_log)
                st.success(f"Predicted Market Cap: ${prediction:,.0f}")
            else:
                st.warning("Not enough data to train prediction model.")
        else:
            st.warning("Insufficient data with 'Market Cap' for modeling.")








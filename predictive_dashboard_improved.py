import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Streamlit app
st.set_page_config(layout="wide")
st.title("Exosome Company Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Create 'Has Partnerships' column
    if 'Notable Partnerships/Deals' in df.columns:
        df['Has Partnerships'] = df['Notable Partnerships/Deals'].notna() & df['Notable Partnerships/Deals'].astype(str).str.strip().ne("")
    else:
        df['Has Partnerships'] = False  # Default to False if column is missing

    st.subheader("Raw Data")
    st.dataframe(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    location_filter = st.sidebar.multiselect("Filter by Location", options=df['Location'].dropna().unique(), default=df['Location'].dropna().unique())
    dev_filter = st.sidebar.multiselect("Filter by Development Stage", options=df['Stage of development'].dropna().unique(), default=df['Stage of development'].dropna().unique())

    # Filtered data
    filtered_df = df[df['Location'].isin(location_filter) & df['Stage of development'].isin(dev_filter)]

    st.subheader("Filtered Data Summary")
    st.write(f"Total Companies: {filtered_df.shape[0]}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Companies by Business Area")
        fig1 = px.histogram(filtered_df, x="Business Area", title="Business Area Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Companies by Development Stage")
        fig2 = px.histogram(filtered_df, x="Stage of development", title="Development Stage Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Companies by Location")
        fig3 = px.histogram(filtered_df, x="Location", title="Location Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Companies with Partnerships")
        fig4 = px.histogram(filtered_df, x="Has Partnerships", title="Partnerships Presence")
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation Heatmap (if enough numeric data)
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig5, ax = plt.subplots()
        sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig5)

    # Market Cap Prediction
    st.subheader("Predict Market Cap (Simple Model)")
    if 'Market Cap' in df.columns:
        model_df = df.dropna(subset=['Market Cap'])

        # Required columns
        required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
        if 'Has Partnerships' in model_df.columns:
            required_cols.append('Has Partnerships')

        if all(col in model_df.columns for col in required_cols):
            X = model_df[required_cols].copy()

            # Convert all non-object columns to string (e.g., bool to str)
            for col in X.columns:
                if X[col].dtype != 'object':
                    X[col] = X[col].astype(str)
            X = X.fillna("Unknown")
            y = model_df['Market Cap']

            # ML pipeline
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)
                ])),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Display sample prediction
            st.write(f"Sample prediction result: ${y_pred[0]:,.0f}")
        else:
            st.error("One or more required columns are missing for prediction.")
    else:
        st.info("Market Cap column not found in the uploaded data, cannot train prediction model.")
else:
    st.warning("Please upload a CSV file to begin.")










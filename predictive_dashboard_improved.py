import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
st.title("Exosome Companies Landscape Dashboard")

uploaded_file = st.file_uploader("Upload your exosome companies CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Filter Companies")

    # Sidebar filters
    filters = {}
    for col in ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            selected_vals = st.sidebar.multiselect(f"Select {col}", unique_vals, default=unique_vals)
            filters[col] = selected_vals

    # Apply filters
    filtered_df = df.copy()
    for col, selected_vals in filters.items():
        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.subheader("Filtered Companies Table")
    st.dataframe(filtered_df, use_container_width=True)

    # Plot: Companies by Business Area
    if 'Business Area' in filtered_df.columns:
        st.subheader("Companies by Business Area")
        fig1, ax = plt.subplots()
        filtered_df['Business Area'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig1)

    # Plot: Companies by Stage of Development
    if 'Stage of development' in filtered_df.columns:
        st.subheader("Companies by Stage of Development")
        fig2, ax = plt.subplots()
        filtered_df['Stage of development'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig2)

    # Plot: Companies by Location
    if 'Location' in filtered_df.columns:
        st.subheader("Companies by Location")
        fig3, ax = plt.subplots()
        filtered_df['Location'].value_counts().head(15).plot(kind='bar', ax=ax, color='green')
        st.pyplot(fig3)

    # Plot: Companies with/without partnerships
    if 'Has Partnerships' in filtered_df.columns:
        st.subheader("Companies With vs Without Notable Partnerships")
        fig4, ax = plt.subplots()
        filtered_df['Has Partnerships'].value_counts().plot(kind='bar', ax=ax, color='purple')
        st.pyplot(fig4)

    # Correlation Heatmap
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = filtered_df[numeric_cols].dropna(axis=1, how='all')
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")
        fig5, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig5)
    else:
        st.info("Not enough usable numeric columns for correlation heatmap.")

    # Predict Market Cap
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
    else:
        st.info("Market Cap column not found in the uploaded data, cannot train prediction model.")
else:
    st.warning("Please upload a CSV file to begin.")











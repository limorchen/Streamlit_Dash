# Heatmap: Business Area Type vs. Notable Partnerships
if 'Business Area' in df.columns and 'Parsed Partnerships' in df.columns:
    st.subheader("Heatmap: Business Area Type vs. Notable Partnerships")

    # Show column names and sample values for debugging
    st.write("✅ DataFrame Columns:", df.columns.tolist())
    st.write("📌 Sample 'Business Area' values:", df['Business Area'].dropna().head())
    st.write("📌 Sample 'Parsed Partnerships' values:", df['Parsed Partnerships'].dropna().head())

    # Convert to string, split, and strip values
    df['Business Area'] = df['Business Area'].astype(str).str.split(',')
    df['Business Area'] = df['Business Area'].apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else [])

    df['Parsed Partnerships'] = df['Parsed Partnerships'].astype(str).str.split(',')
    df['Parsed Partnerships'] = df['Parsed Partnerships'].apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else [])

    # Explode into long format
    df_exploded = df.explode('Business Area')
    df_exploded = df_exploded.explode('Parsed Partnerships')
    df_exploded = df_exploded.dropna(subset=['Business Area', 'Parsed Partnerships'])

    # Debug: check exploded data
    st.write("🔍 Exploded Data Preview:")
    st.dataframe(df_exploded.head())

    if not df_exploded.empty:
        # Generate crosstab and convert to percentage
        heatmap_data = pd.crosstab(
            df_exploded['Business Area'],
            df_exploded['Parsed Partnerships'],
            normalize='index'
        ) * 100

        st.write("🔢 Crosstab Preview:")
        st.dataframe(heatmap_data.head())

        # Plot the heatmap using Plotly
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

# ===============================
# Prediction Block: Market Cap
# ===============================
st.subheader("Predict Market Cap (Simple Model)")
if 'Market Cap' in df.columns:
    model_df = df.dropna(subset=['Market Cap'])

    required_cols = ['Business Area', 'Cell type (source/target)', 'Stage of development', 'Location']
    if 'Has Partnerships' in model_df.columns:
        required_cols.append('Has Partnerships')

    if all(col in model_df.columns for col in required_cols):
        X = model_df[required_cols].copy()
        
        # Ensure consistent string dtype for OneHotEncoder
        for col in X.columns:
            if X[col].dtype != 'object':
                X[col] = X[col].astype(str)
        X = X.fillna("Unknown")
        y = model_df['Market Cap']

        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Split and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)

        st.success(f"Model R² Score: {score:.2f}")

        # UI for prediction
        st.markdown("### Try the Market Cap Prediction")
        user_input = {}
        for col in required_cols:
            options = sorted(model_df[col].dropna().unique())
            user_input[col] = st.selectbox(f"Select {col}", options)

        input_df = pd.DataFrame([user_input])
        predicted_cap = pipeline.predict(input_df)[0]
        st.metric("Predicted Market Cap", f"${predicted_cap:,.0f}")
    else:
        st.warning("Some required columns are missing for the prediction model.")










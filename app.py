import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import joblib

# ============================
# Default Parameters
# ============================
TEST_TRAIN_RATIO = 0.2
TARGET_COLUMN = None
COLUMN_NAMES = []
MODEL_TYPE = "Classification"
MODEL_CHOICE = "Random Forest"
NULL_HANDLING = "Drop Rows"

# ============================
# Streamlit setup
# ============================
st.set_page_config(layout="wide")
st.title("AutoML")

# Two main columns for inputs
left_col, right_col = st.columns(2)
df = None
model = None
train_clicked = False

# ============================
# Left column: Dataset upload + Target + Dataset Info
# ============================
with left_col:
    st.text("Upload Dataset")
    dataset = st.file_uploader("", label_visibility="collapsed", type=["csv"])

    if dataset:
        df = pd.read_csv(dataset)
        COLUMN_NAMES = df.columns.tolist()

        # Target selection (radio horizontal)
        st.markdown("### Select Target Column")
        TARGET_COLUMN = st.radio(
            "Target Column",
            options=COLUMN_NAMES,
            horizontal=True
        )
        st.text(f"Target column: {TARGET_COLUMN}")

        # Dataset info
        total_rows = df.shape[0]
        empty_rows = df.isnull().any(axis=1).sum()

        # Determine type of data (use .shape to get column count)
        num_cols = df.select_dtypes(include=np.number).columns.size
        cat_cols = df.select_dtypes(include="object").columns.size
        if num_cols > 0 and cat_cols > 0:
            data_type = "Mixed"
        elif num_cols > 0:
            data_type = "Numerical"
        elif cat_cols > 0:
            data_type = "Categorical"
        else:
            data_type = "Unknown"

        st.markdown(
            f"**Number of rows:** {total_rows}  \n"
            f"**Number of rows with missing values:** {empty_rows}  \n"
            f"**Data Type:** {data_type}"
        )

# ============================
# Right column: Model options + Train
# ============================
with right_col:
    if df is not None:
        st.markdown("### Model Options")

        # Test/train split
        test_ratio = st.number_input(
            "Test Train Split Ratio (0-1)",
            min_value=0.05, max_value=0.5, value=TEST_TRAIN_RATIO, step=0.05
        )

        # Model type
        model_type = st.selectbox(
            "Model Type",
            ["Classification", "Regression"],
            index=["Classification", "Regression"].index(MODEL_TYPE)
        )

        # Model choice depends on type
        if model_type == "Classification":
            options = ["Logistic Regression", "Random Forest", "XGBoost"]
            default_choice = "Random Forest" if "Random Forest" in options else options[0]
            model_choice = st.selectbox("Model", options, index=options.index(default_choice))
            use_pca = st.checkbox("Use PCA (2 components)")
        else:
            options = ["Linear Regression", "Random Forest Regressor", "XGBoost"]
            default_choice = "Linear Regression"
            model_choice = st.selectbox("Model", options, index=options.index(default_choice))
            use_pca = False
        
        #else:
        #    options = ["KMeans", "DBSCAN"]
        #    default_choice = "KMeans"
        #    model_choice = st.selectbox("Model", options, index=options.index(default_choice))
        #    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=3, step=1)
        #    use_pca = False
        
        # Normalization option (applies ONLY to numeric columns)
        normalization = st.selectbox(
            "Normalization Technique",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            index=0
        )

        # Handle nulls
        null_handling = st.selectbox(
            "Handle Nulls",
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
            index=["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"].index(NULL_HANDLING)
        )

        # Buttons side by side
        btn_col1, btn_col2 = st.columns([1, 1])
        train_clicked = btn_col1.button("Train Model", type="primary")

# ============================
# Training and Metrics
# ============================
if train_clicked and df is not None and TARGET_COLUMN is not None:
    df_copy = df.copy()

    # Missing value handling
    if null_handling == "Drop Rows":
        df_copy = df_copy.dropna()
    elif null_handling == "Fill with Mean":
        df_copy = df_copy.fillna(df_copy.mean(numeric_only=True))
    elif null_handling == "Fill with Median":
        df_copy = df_copy.fillna(df_copy.median(numeric_only=True))
    elif null_handling == "Fill with Mode":
        df_copy = df_copy.fillna(df_copy.mode().iloc[0])

    # Separate target & features
    y_raw = df_copy[TARGET_COLUMN]
    X = df_copy.drop(columns=[TARGET_COLUMN])

    # Encode target ONCE for classification; keep numeric for regression
    if model_type == "Classification":
        y_le = LabelEncoder()
        y = y_le.fit_transform(pd.Series(y_raw).astype(str).fillna("NA"))
    else:
        # Regression: keep target numeric (coerce if needed)
        y = pd.to_numeric(y_raw, errors="coerce")

    # Detect original feature types once
    original_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Treat object/category/bool as categorical to avoid scaling booleans
    original_cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # --- Numeric pipeline: coerce to numeric, drop all-NaN columns, then scale if selected ---
    if len(original_num_cols) > 0:
        X_num = X[original_num_cols].apply(pd.to_numeric, errors="coerce")
        keep_mask = ~X_num.isna().all(axis=0)
        X_num = X_num.loc[:, keep_mask]
    else:
        X_num = pd.DataFrame(index=X.index)

    if normalization != "None" and X_num.shape[1] > 0:
        if normalization == "StandardScaler":
            scaler = StandardScaler()
        elif normalization == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif normalization == "RobustScaler":
            scaler = RobustScaler()
        X_num_scaled = pd.DataFrame(
            scaler.fit_transform(X_num.values.astype(float)),
            index=X_num.index,
            columns=X_num.columns
        )
    else:
        X_num_scaled = X_num

    # --- Categorical pipeline: encode AFTER numeric scaling, never scale categoricals ---
    if len(original_cat_cols) > 0:
        X_cat = X[original_cat_cols].copy()
        for col in X_cat.columns:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col].astype(str).fillna("NA"))
    else:
        X_cat = pd.DataFrame(index=X.index)

    # --- Combine back numeric + categorical ---
    if X_num_scaled.size > 0 and X_cat.size > 0:
        X = pd.concat([X_num_scaled, X_cat], axis=1)
    elif X_num_scaled.size > 0:
        X = X_num_scaled
    else:
        X = X_cat  # rare case: all-categorical dataset

    # Keep a numeric-only copy for clustering visualization (after preprocessing)
    X_numeric_all = X.select_dtypes(include=[np.number]).copy()

    # --- PCA: only on scaled numeric features (ignore encoded categoricals) ---
    if use_pca and model_type == "Classification":
        if X_num_scaled.shape >= 2:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_num_scaled)
            # Replace features with just PCs for simplicity
            X = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X.index)
        else:
            st.warning("Not enough numeric features for PCA. Skipping PCA.")

    # Split for Classification/Regression
    if model_type in ["Classification", "Regression"]:
        stratify_arg = y if model_type == "Classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42, stratify=stratify_arg
            )
        except ValueError:
            # Fallback if stratify fails due to rare classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42
            )

    # Select model
    model = None
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor()
    elif model_choice == "XGBoost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if model_type == "Classification":
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            else:
                model = XGBRegressor()
        except Exception:
            st.error("XGBoost is not installed. Please install xgboost to use this model.")
            model = None
    elif model_choice == "KMeans":
        # For sklearn<1.4 replace n_init="auto" with n_init=10
        try:
            model = KMeans(n_clusters=n_clusters, n_init="auto")
        except TypeError:
            model = KMeans(n_clusters=n_clusters, n_init=10)
    elif model_choice == "DBSCAN":
        model = DBSCAN()

    # Train model (show spinner)
    if model is not None:
        with st.spinner("Training model, please wait..."):
            if model_type in ["Classification", "Regression"]:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:  # Clustering
                # Fit on numeric features
                X_clust = X_numeric_all if 'X_numeric_all' in locals() else X.select_dtypes(include=np.number)
                model.fit(X_clust)

        st.success("✅ Model Trained!")

        # ====================
        # Metrics & Visualizations
        # ====================
        if model_type == "Classification":
            st.markdown("<h3 style='text-align: center;'>Model Metrics</h3>", unsafe_allow_html=True)
            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                # Class imbalance detection on full y
                class_counts = pd.Series(y).value_counts()
                imbalance_ratio = class_counts.max() / class_counts.sum()
                if imbalance_ratio > 0.7:
                    st.warning(
                        f"⚠️ Dataset is imbalanced! Class `{class_counts.idxmax()}` has "
                        f"{class_counts.max()} out of {class_counts.sum()} samples."
                    )

                # Overall metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                st.write(f"**Accuracy:** {acc:.3f}")
                st.write(f"**Precision:** {prec:.3f}")
                st.write(f"**Recall:** {rec:.3f}")

                # Cross-validation metrics
                try:
                    acc_cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    st.write(f"**5-Fold CV Accuracy:** {acc_cv.mean():.3f} ± {acc_cv.std():.3f}")
                except Exception:
                    st.info("Could not run cross-validation for this model/dataset.")

                # Per-class metrics
                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.subheader("Per-Class Metrics")
                    st.dataframe(report_df)
                except Exception:
                    st.info("Per-class report not available.")

                # Confusion matrix (Plotly)
                try:
                    cm_df = pd.DataFrame(
                        confusion_matrix(y_test, y_pred),
                        index=[f"Actual {i}" for i in np.unique(y_test)],
                        columns=[f"Pred {i}" for i in np.unique(y_test)]
                    )
                    cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', aspect="auto")
                    cm_fig.update_layout(width=400, height=300, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
                    st.subheader("Confusion Matrix")
                    st.plotly_chart(cm_fig, use_container_width=True)
                except Exception:
                    st.info("Could not compute confusion matrix.")

                # Class distribution
                dist_fig = px.bar(
                    x=class_counts.index, y=class_counts.values, text=class_counts.values,
                    labels={'x': 'Class', 'y': 'Count'}, title="Class Distribution"
                )
                st.plotly_chart(dist_fig, use_container_width=True)

        elif model_type == "Regression":
            st.markdown("<h3 style='text-align: center;'>Model Metrics</h3>", unsafe_allow_html=True)
            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                # RMSE (root mean squared error)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.write(f"**RMSE:** {rmse:.3f}")

                # CV RMSE (preferred scorer, fallback)
                try:
                    rmse_cv = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
                    st.write(f"**5-Fold CV RMSE:** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")
                except Exception:
                    try:
                        mse_cv = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                        rmse_cv = np.sqrt(np.abs(mse_cv))
                        st.write(f"**5-Fold CV RMSE (approx):** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")
                    except Exception:
                        st.info("Could not compute cross-validated RMSE.")

                # Regression Fit visualization
                st.subheader("Regression Fit")
                if X_test.shape[1] == 1:
                    xx = X_test.iloc[:, 0]
                    fig = px.scatter(x=xx, y=y_test, labels={"x": X_test.columns, "y": TARGET_COLUMN},
                                     title="Actual vs Predicted with Regression Line")
                    fig.add_scatter(x=xx, y=y_pred, mode="lines", name="Predicted")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                                     title="Predicted vs Actual")
                    fig.add_scatter(x=y_test, y=y_test, mode="lines", name="Ideal Fit")
                    st.plotly_chart(fig, use_container_width=True)

        else:  # Clustering
            # Build numeric matrix and labels
            X_clust = X_numeric_all if 'X_numeric_all' in locals() else X.select_dtypes(include=np.number)
            if hasattr(model, "labels_"):
                labels = model.labels_
            else:
                try:
                    labels = model.predict(X_clust)
                except Exception:
                    labels = getattr(model, "labels_", np.zeros(len(X_clust), dtype=int))

            st.markdown("<h3 style='text-align: center;'>Cluster Visualizations</h3>", unsafe_allow_html=True)
            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                # Cluster size distribution
                cl_counts = pd.Series(labels).value_counts().sort_index()
                dist_fig = px.bar(
                    x=cl_counts.index.astype(str),
                    y=cl_counts.values,
                    text=cl_counts.values,
                    labels={"x": "Cluster", "y": "Count"},
                    title="Cluster Sizes"
                )
                st.plotly_chart(dist_fig, use_container_width=True)

                # 2D visualization: direct if exactly 2 numeric features; else PCA(2)
                plot_df = X_clust.copy()
                use_direct = plot_df.shape[1] == 2
                centers_2d = None
                if use_direct:
                    f1, f2 = plot_df.columns[:2].tolist()
                    emb_df = plot_df.rename(columns={f1: "F1", f2: "F2"})
                    emb_cols = ["F1", "F2"]
                    subtitle = "Original Feature Space"
                    if hasattr(model, "cluster_centers_") and isinstance(model, KMeans):
                        centers = model.cluster_centers_
                        centers_2d = pd.DataFrame(centers, columns=emb_cols)
                else:
                    pca_vis = PCA(n_components=2, random_state=42)
                    emb = pca_vis.fit_transform(plot_df.values)
                    emb_cols = ["PC1", "PC2"]
                    subtitle = "PCA(2) Projection"
                    emb_df = pd.DataFrame(emb, columns=emb_cols, index=plot_df.index)
                    if hasattr(model, "cluster_centers_") and isinstance(model, KMeans):
                        centers = model.cluster_centers_
                        centers_2d = pd.DataFrame(pca_vis.transform(centers), columns=emb_cols)

                emb_df["cluster"] = labels.astype(int)

                scatter = px.scatter(
                    emb_df,
                    x=emb_cols[0],
                    y=emb_cols,
                    color=emb_df["cluster"].astype(str),
                    opacity=0.85,
                    title=f"Clusters in 2D ({subtitle})",
                    labels={emb_cols: emb_cols, emb_cols: emb_cols, "color": "cluster"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                # KMeans centers overlay
                if centers_2d is not None:
                    scatter.add_scatter(
                        x=centers_2d[emb_cols],
                        y=centers_2d[emb_cols],
                        mode="markers+text",
                        marker=dict(symbol="x", size=12, color="black"),
                        text=[f"C{k}" for k in range(centers_2d.shape)],
                        textposition="top center",
                        name="Centers"
                    )

                st.plotly_chart(scatter, use_container_width=True)

                # Pairwise feature scatters for small-dimensional data (2–4 features)
                if X_clust.shape >= 2 and X_clust.shape <= 4:
                    st.subheader("Pairwise Feature Scatter (Colored by Cluster)")
                    cols = X_clust.columns.tolist()
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            pair_df = pd.DataFrame({
                                cols[i]: X_clust[cols[i]].values,
                                cols[j]: X_clust[cols[j]].values,
                                "cluster": labels.astype(int).astype(str)
                            })
                            fig_pair = px.scatter(
                                pair_df,
                                x=cols[i],
                                y=cols[j],
                                color="cluster",
                                opacity=0.8,
                                title=f"{cols[i]} vs {cols[j]}",
                                color_discrete_sequence=px.colors.qualitative.Safe
                            )
                            st.plotly_chart(fig_pair, use_container_width=True)

                # Optional silhouette score (exclude DBSCAN noise)
                try:
                    from sklearn.metrics import silhouette_score
                    unique_labels = np.unique(labels)
                    if -1 in unique_labels:
                        valid_mask = labels != -1
                    else:
                        valid_mask = np.ones_like(labels, dtype=bool)
                    labeled_points = np.sum(valid_mask)
                    labeled_unique = np.unique(labels[valid_mask])
                    if labeled_points >= 10 and labeled_unique.size >= 2:
                        sil = silhouette_score(X_clust[valid_mask], labels[valid_mask])
                        st.write(f"Silhouette score (excluding noise): {sil:.3f}")
                except Exception:
                    pass

            # Download clustering model
            try:
                with open("trained_model.pkl", "wb") as f:
                    joblib.dump(model, f)
                with open("trained_model.pkl", "rb") as f:
                    st.download_button(
                        label="Download Trained Model",
                        data=f,
                        file_name="trained_model.pkl",
                        type="secondary"
                    )
            except Exception:
                st.info("Could not save model to disk in this environment.")
    else:
        st.info("No model selected or failed to initialize.")

# ============================
# Dataset Visualizations
# ============================
if df is not None:
    st.markdown("---")
    st.header("Dataset Visualizations")
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Target + correlation heatmap
    col1, col2 = st.columns(2)
    with col1:
        if TARGET_COLUMN:
            st.subheader("Target Distribution")
            fig = px.histogram(df, x=TARGET_COLUMN, height=200, width=300)
            fig.update_layout(margin=dict(l=10, r=10, t=25, b=20), font=dict(size=8))
            st.plotly_chart(fig, use_container_width=False)
    with col2:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True).round(2)
        if corr.size > 0:
            fig = ff.create_annotated_heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.columns), colorscale="Blues", showscale=True
            )
            for ann in fig.layout.annotations:
                ann.font.size = 12
            fig.update_layout(height=400, width=500, margin=dict(l=10, r=10, t=25, b=20), font=dict(size=10))
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.info("Not enough numeric columns to compute correlation heatmap.")

    # Histograms
    st.subheader("Histograms")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns to show histograms.")
    else:
        cols_per_row = 6
        row = None
        for i, col in enumerate(numeric_cols):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row)
            with row[i % cols_per_row]:
                fig = px.histogram(df, x=col, nbins=20, height=200, width=300)
                fig.update_layout(margin=dict(l=5, r=5, t=20, b=20), font=dict(size=7), showlegend=False)
                st.plotly_chart(fig, use_container_width=False)

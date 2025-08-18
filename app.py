import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
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
st.title("AutoML Demo")

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

        # Target selection
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

        # Determine type of data
        num_cols = df.select_dtypes(include=np.number).shape[1]
        cat_cols = df.select_dtypes(include="object").shape[1]
        if num_cols > 0 and cat_cols > 0:
            data_type = "Mixed"
        elif num_cols > 0:
            data_type = "Numerical"
        elif cat_cols > 0:
            data_type = "Categorical"
        else:
            data_type = "Unknown"

        st.markdown(f"**Number of rows:** {total_rows}  \n"
                    f"**Number of rows with missing values:** {empty_rows}  \n"
                    f"**Data Type:** {data_type}")

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
            ["Classification", "Regression", "Clustering"],
            index=["Classification","Regression","Clustering"].index(MODEL_TYPE)
        )

        # Model choice depends on type
        if model_type == "Classification":
            model_choice = st.selectbox(
                "Model",
                ["Logistic Regression", "Random Forest", "XGBoost"],
                index=["Logistic Regression", "Random Forest", "XGBoost"].index(MODEL_CHOICE)
            )
            use_pca = st.checkbox("Use PCA (2 components)")
        elif model_type == "Regression":
            model_choice = st.selectbox(
                "Model",
                ["Linear Regression", "Random Forest", "XGBoost"],
                index=["Linear Regression","Random Forest","XGBoost"].index(MODEL_CHOICE)
            )
            use_pca = False
        else:
            model_choice = st.selectbox(
                "Model",
                ["KMeans", "DBSCAN"],
                index=["KMeans","DBSCAN"].index(MODEL_CHOICE)
            )
            n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=3, step=1)
            use_pca = False

        # Handle nulls
        null_handling = st.selectbox(
            "Handle Nulls",
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
            index=["Drop Rows","Fill with Mean","Fill with Median","Fill with Mode"].index(NULL_HANDLING)
        )

        # Buttons side by side
        btn_col1, btn_col2 = st.columns([1,1])
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
    y = df_copy[TARGET_COLUMN]
    X = df_copy.drop(columns=[TARGET_COLUMN])

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if model_type == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Apply PCA if selected
    if use_pca and model_type=="Classification":
        X_num = X.select_dtypes(include=np.number)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_num)
        X = pd.DataFrame(X_pca, columns=["PC1","PC2"])

    # Split for Classification/Regression
    if model_type in ["Classification", "Regression"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    # Select model
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor()
    elif model_choice == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        if model_type == "Classification":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        else:
            model = XGBRegressor()
    elif model_choice == "KMeans":
        model = KMeans(n_clusters=n_clusters)

    # Train model
    if model_type in ["Classification","Regression"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:  # Clustering
        X_train = X.select_dtypes(include=np.number)
        model.fit(X_train)

    st.success("Model Trained!")

    # ====================
    # Centered Metrics Container
    # ====================
    st.markdown("<h3 style='text-align: center;'>Model Metrics</h3>", unsafe_allow_html=True)
    center_col1, center_col2, center_col3 = st.columns([1,2,1])
    with center_col2:
        if model_type == "Classification":
            # Class imbalance detection
            class_counts = pd.Series(y).value_counts()
            imbalance_ratio = class_counts.max()/class_counts.sum()
            if imbalance_ratio>0.7:
                st.warning(f"⚠️ Dataset is imbalanced! Class `{class_counts.idxmax()}` has {class_counts.max()} out of {class_counts.sum()} samples.")

            # Overall metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            st.write(f"**Accuracy:** {acc:.3f}")
            st.write(f"**Precision:** {prec:.3f}")
            st.write(f"**Recall:** {rec:.3f}")

            # Cross-validation metrics
            acc_cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            st.write(f"**5-Fold CV Accuracy:** {acc_cv.mean():.3f} ± {acc_cv.std():.3f}")

            # Per-class metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.subheader("Per-Class Metrics")
            st.dataframe(report_df)

            # Confusion matrix (Plotly)
            cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                 index=[f"Actual {i}" for i in np.unique(y_test)],
                                 columns=[f"Pred {i}" for i in np.unique(y_test)])
            cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', aspect="auto")
            cm_fig.update_layout(width=400, height=300, margin=dict(l=20,r=20,t=40,b=20), coloraxis_showscale=False)
            st.subheader("Confusion Matrix")
            st.plotly_chart(cm_fig, use_container_width=True)

            # Class distribution
            dist_fig = px.bar(x=class_counts.index, y=class_counts.values, text=class_counts.values,
                              labels={'x':'Class','y':'Count'}, title="Class Distribution")
            st.plotly_chart(dist_fig, use_container_width=True)

        elif model_type=="Regression":
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.write(f"**RMSE:** {rmse:.3f}")
            rmse_cv = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
            st.write(f"**5-Fold CV RMSE:** {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f}")

        else:  # Clustering
            st.write(f"Cluster centers:\n{model.cluster_centers_}")

        # Download button (secondary color)
        with open("trained_model.pkl","wb") as f:
            joblib.dump(model, f)
        with open("trained_model.pkl","rb") as f:
            st.download_button(
                label="Download Trained Model",
                data=f,
                file_name="trained_model.pkl",
                type="secondary"
            )

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
            fig.update_layout(margin=dict(l=10,r=10,t=25,b=20), font=dict(size=8))
            st.plotly_chart(fig, use_container_width=False)
    with col2:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True).round(2)
        fig = ff.create_annotated_heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.columns), colorscale="Blues", showscale=True
        )
        for ann in fig.layout.annotations: ann.font.size = 12
        fig.update_layout(height=400,width=500,margin=dict(l=10,r=10,t=25,b=20), font=dict(size=10))
        st.plotly_chart(fig, use_container_width=False)

    # Histograms
    st.subheader("Histograms")
    numeric_cols = df.select_dtypes(include="number").columns
    for i, col in enumerate(numeric_cols):
        if i % 6 == 0:
            row = st.columns(6)
        with row[i%6]:
            fig = px.histogram(df, x=col, nbins=20, height=200, width=300)
            fig.update_layout(margin=dict(l=5,r=5,t=20,b=20), font=dict(size=7), showlegend=False)
            st.plotly_chart(fig, use_container_width=False)

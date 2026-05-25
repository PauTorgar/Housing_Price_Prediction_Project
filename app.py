from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Housing Price Prediction Dashboard",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "train.csv"
ACCENT = "#d8ff62"
PURPLE = "#8f7cff"
BORDER = "#2a2a2a"
TEXT = "#f5f5f5"
SUBTLE = "#a0a0a0"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(143, 124, 255, 0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(216, 255, 98, 0.09), transparent 24%),
                    linear-gradient(180deg, #151515 0%, #0d0d0d 100%);
                color: {TEXT};
            }}
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"] {{
                background: #0d0d0d !important;
            }}
            div[data-testid="stDecoration"] {{
                background: transparent !important;
            }}
            .block-container {{
                padding-top: 1.6rem;
                padding-bottom: 2rem;
                max-width: 1380px;
            }}
            .hero-shell {{
                background: rgba(20, 20, 20, 0.96);
                border: 1px solid {BORDER};
                border-radius: 28px;
                padding: 32px 24px 22px 24px;
                margin-bottom: 18px;
                box-shadow: 0 16px 60px rgba(0, 0, 0, 0.32);
            }}
            .eyebrow {{
                color: {SUBTLE};
                font-size: 0.86rem;
            }}
            .hero-title {{
                font-size: 2.7rem;
                line-height: 1;
                font-weight: 700;
                margin: 0 0 0.55rem 0;
            }}
            .hero-sub {{
                color: #d7d7d7;
                max-width: 800px;
                font-size: 0.98rem;
                margin-bottom: 0;
            }}
            .tag {{
                display: inline-flex;
                align-items: center;
                padding: 6px 12px;
                border-radius: 999px;
                background: rgba(143, 124, 255, 0.17);
                color: #cabdff;
                border: 1px solid rgba(143, 124, 255, 0.25);
                font-size: 0.78rem;
                margin-top: 0.65rem;
            }}
            .card {{
                background: linear-gradient(180deg, #1a1a1a 0%, #151515 100%);
                border: 1px solid {BORDER};
                border-radius: 24px;
                padding: 18px 18px 16px 18px;
                height: 100%;
            }}
            .card.accent {{
                background: linear-gradient(145deg, rgba(216,255,98,0.98), rgba(194,255,90,0.94));
                color: #121212;
                border-color: rgba(216,255,98,0.45);
            }}
            .outputs-card {{
                margin-bottom: 1.35rem;
            }}
            .prediction-card {{
                margin-bottom: 1.35rem;
            }}
            .metric-label {{
                font-size: 0.82rem;
                color: {SUBTLE};
                margin-bottom: 8px;
            }}
            .accent .metric-label {{
                color: rgba(16,16,16,0.72);
            }}
            .metric-value {{
                font-size: 2rem;
                line-height: 1;
                font-weight: 700;
                margin-bottom: 6px;
            }}
            .metric-delta {{
                font-size: 0.82rem;
                color: #d0d0d0;
            }}
            .accent .metric-delta {{
                color: rgba(16,16,16,0.76);
            }}
            .panel-title {{
                font-size: 0.96rem;
                font-weight: 700;
                margin-bottom: 2px;
            }}
            .panel-subtitle {{
                font-size: 0.82rem;
                color: {SUBTLE};
                margin-bottom: 10px;
            }}
            .insight-box {{
                background: #171717;
                border: 1px solid {BORDER};
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 12px;
            }}
            .insight-label {{
                font-size: 0.76rem;
                color: {SUBTLE};
                margin-bottom: 4px;
            }}
            .insight-value {{
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 4px;
            }}
            .insight-copy {{
                font-size: 0.85rem;
                color: #d9d9d9;
                line-height: 1.45;
                margin: 0;
            }}
            .plot-shell {{
                background: #171717;
                border: 1px solid {BORDER};
                border-radius: 20px;
                padding: 12px;
            }}
            div[data-testid="stRadio"] > label {{
                display: none;
            }}
            div[data-testid="stRadio"] {{
                margin-bottom: 1.9rem;
            }}
            div[role="radiogroup"] {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px 26px;
            }}
            div[role="radiogroup"] label {{
                background: #1d1d1d !important;
                border: 1px solid {BORDER} !important;
                border-radius: 999px !important;
                padding: 10px 16px !important;
                color: {TEXT} !important;
                min-height: auto !important;
            }}
            div[role="radiogroup"] label:has(input:checked) {{
                background: {ACCENT} !important;
                color: #111 !important;
                border-color: rgba(216,255,98,0.45) !important;
                font-weight: 700 !important;
            }}
            .stSelectbox label, .stSlider label {{
                color: #d9d9d9 !important;
                font-size: 0.86rem !important;
            }}
            div[data-baseweb="select"] > div,
            .stSlider [data-baseweb="slider"] {{
                background: #171717 !important;
                border-radius: 14px !important;
                border-color: {BORDER} !important;
            }}
            .stButton > button {{
                width: 100%;
                border-radius: 14px;
                border: 1px solid {BORDER};
                font-weight: 700;
                padding: 0.7rem 1rem;
            }}
            div[data-testid="stDownloadButton"] > button {{
                width: 100%;
                border-radius: 14px;
                background: {ACCENT};
                color: #111;
                border: none;
                font-weight: 700;
                padding: 0.7rem 1rem;
            }}
            @media (max-width: 900px) {{
                .hero-title {{
                    font-size: 2.1rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    engineered = dataframe.copy()
    engineered["HouseAge"] = engineered["YrSold"] - engineered["YearBuilt"]
    engineered["RemodAge"] = engineered["YrSold"] - engineered["YearRemodAdd"]
    engineered["TotalSF"] = (
        engineered["TotalBsmtSF"].fillna(0)
        + engineered["1stFlrSF"].fillna(0)
        + engineered["2ndFlrSF"].fillna(0)
    )
    engineered["TotalBath"] = (
        engineered["FullBath"].fillna(0)
        + 0.5 * engineered["HalfBath"].fillna(0)
        + engineered["BsmtFullBath"].fillna(0)
        + 0.5 * engineered["BsmtHalfBath"].fillna(0)
    )
    engineered["HasGarage"] = np.where(engineered["GarageArea"].fillna(0) > 0, 1, 0)
    return engineered


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_assets() -> dict:
    raw_df = load_dataset()
    df = raw_df.copy()
    outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
    df = df.loc[~outlier_mask].copy()
    df = engineer_features(df)

    target = "SalePrice"
    features = df.drop(columns=[target, "Id"], errors="ignore")
    y = df[target].copy()

    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_valid, y_train, y_valid = train_test_split(
        features, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    trained_pipelines = {}
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)

        results.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_valid, preds),
                "RMSE": np.sqrt(mean_squared_error(y_valid, preds)),
                "R2": r2_score(y_valid, preds),
            }
        )
        trained_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    best_pipeline = trained_pipelines[best_model_name]

    base_row = X_train.mode(dropna=False).iloc[0].copy()
    for col in numeric_features:
        base_row[col] = float(X_train[col].median())
    for col in categorical_features:
        mode_values = X_train[col].mode(dropna=True)
        if not mode_values.empty:
            base_row[col] = mode_values.iloc[0]

    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    best_model = best_pipeline.named_steps["model"]
    raw_importance = (
        best_model.feature_importances_
        if hasattr(best_model, "feature_importances_")
        else np.abs(np.ravel(best_model.coef_))
    )
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": raw_importance})
        .sort_values("importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    return {
        "raw_df": raw_df,
        "model_df": df,
        "features": features,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "importance_df": importance_df,
        "base_row": base_row,
        "outliers_removed": int(outlier_mask.sum()),
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def metric_card(title: str, value: str, delta: str, accent: bool = False) -> None:
    card_class = "card accent" if accent else "card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 16, "color": TEXT}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#171717",
        "font": {"color": TEXT, "family": "Arial"},
        "margin": {"l": 20, "r": 20, "t": 46, "b": 20},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.06)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.06)"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
    }


def get_prediction_defaults(base_row: pd.Series) -> None:
    if "prediction_defaults" not in st.session_state:
        st.session_state.prediction_defaults = {
            "overall_qual": int(base_row["OverallQual"]),
            "gr_liv_area": int(base_row["GrLivArea"]),
            "garage_cars": int(base_row["GarageCars"]),
            "full_bath": int(base_row["FullBath"]),
            "year_built": int(base_row["YearBuilt"]),
            "total_bsmt_sf": int(base_row["TotalBsmtSF"]),
            "garage_area": int(base_row["GarageArea"]),
            "half_bath": int(base_row["HalfBath"]),
            "year_remod_add": int(base_row["YearRemodAdd"]),
            "yr_sold": int(base_row["YrSold"]),
            "neighborhood": base_row["Neighborhood"],
            "house_style": base_row["HouseStyle"],
            "kitchen_qual": base_row["KitchenQual"],
        }


def reset_prediction_inputs() -> None:
    for key, value in st.session_state.prediction_defaults.items():
        st.session_state[key] = value


def build_prediction_input(base_row: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
    get_prediction_defaults(base_row)
    defaults = st.session_state.prediction_defaults
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        <div class="card prediction-card">
            <div class="panel-title">Live Price Estimate</div>
            <div class="panel-subtitle">Adjust the property profile to generate a real-time prediction with the strongest model.</div>
        """,
        unsafe_allow_html=True,
    )

    action_col_1, action_col_2 = st.columns(2)
    with action_col_1:
        if st.button("Reset values", key="reset_prediction"):
            reset_prediction_inputs()
            st.rerun()
    with action_col_2:
        st.download_button(
            "Download Report",
            data="Housing Price Intelligence dashboard export\n",
            file_name="housing_dashboard_report.txt",
            mime="text/plain",
            key="download_report",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        overall_qual = st.slider("Overall quality", 1, 10, key="overall_qual")
        gr_liv_area = st.slider("Above-ground living area", 400, 4500, key="gr_liv_area")
        garage_cars = st.slider("Garage capacity", 0, 5, key="garage_cars")
        full_bath = st.slider("Full bathrooms", 0, 4, key="full_bath")
        year_built = st.slider("Year built", 1875, 2010, key="year_built")
    with col_b:
        total_bsmt_sf = st.slider("Basement area", 0, 3200, key="total_bsmt_sf")
        garage_area = st.slider("Garage area", 0, 1500, key="garage_area")
        half_bath = st.slider("Half bathrooms", 0, 3, key="half_bath")
        year_remod_add = st.slider("Remodel year", 1950, 2010, key="year_remod_add")
        yr_sold = st.slider("Sale year", 2006, 2010, key="yr_sold")

    neighborhood_options = sorted(data["Neighborhood"].dropna().unique().tolist())
    house_style_options = sorted(data["HouseStyle"].dropna().unique().tolist())
    kitchen_quality_options = sorted(data["KitchenQual"].dropna().unique().tolist())

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        neighborhood = st.selectbox(
            "Neighborhood",
            neighborhood_options,
            index=neighborhood_options.index(defaults["neighborhood"]),
            key="neighborhood",
        )
    with col_d:
        house_style = st.selectbox(
            "House style",
            house_style_options,
            index=house_style_options.index(defaults["house_style"]),
            key="house_style",
        )
    with col_e:
        kitchen_qual = st.selectbox(
            "Kitchen quality",
            kitchen_quality_options,
            index=kitchen_quality_options.index(defaults["kitchen_qual"]),
            key="kitchen_qual",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    user_row = base_row.copy()
    user_row["OverallQual"] = overall_qual
    user_row["GrLivArea"] = gr_liv_area
    user_row["GarageCars"] = garage_cars
    user_row["FullBath"] = full_bath
    user_row["YearBuilt"] = year_built
    user_row["TotalBsmtSF"] = total_bsmt_sf
    user_row["GarageArea"] = garage_area
    user_row["HalfBath"] = half_bath
    user_row["YearRemodAdd"] = year_remod_add
    user_row["YrSold"] = yr_sold
    user_row["Neighborhood"] = neighborhood
    user_row["HouseStyle"] = house_style
    user_row["KitchenQual"] = kitchen_qual
    user_row["HouseAge"] = user_row["YrSold"] - user_row["YearBuilt"]
    user_row["RemodAge"] = user_row["YrSold"] - user_row["YearRemodAdd"]
    user_row["TotalSF"] = (
        user_row["TotalBsmtSF"] + user_row["1stFlrSF"] + user_row["2ndFlrSF"]
    )
    user_row["TotalBath"] = (
        user_row["FullBath"]
        + 0.5 * user_row["HalfBath"]
        + user_row["BsmtFullBath"]
        + 0.5 * user_row["BsmtHalfBath"]
    )
    user_row["HasGarage"] = 1 if user_row["GarageArea"] > 0 else 0
    return pd.DataFrame([user_row])


def render_top_navigation(best_model_name: str) -> str:
    nav_col_1, nav_col_2 = st.columns([4, 1])
    with nav_col_1:
        current_view = st.radio(
            "Sections",
            ["Overview", "Insights", "Analytics", "Prediction", "Reports"],
            horizontal=True,
            key="current_view",
            label_visibility="collapsed",
        )
    with nav_col_2:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end;">
                <div class="tag">Best model: {best_model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return current_view


def render_filter_bar() -> str:
    return st.radio(
        "Filters",
        ["All", "Exploration", "Modeling", "Prediction"],
        horizontal=True,
        key="content_filter",
        label_visibility="collapsed",
    )


def render_overview(raw_df: pd.DataFrame, model_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    selected_filter = render_filter_bar()

    if selected_filter in ["All", "Exploration"]:
        chart_left, chart_mid = st.columns([1.25, 1])
        with chart_left:
            dist_fig = px.histogram(raw_df, x="SalePrice", nbins=36, opacity=0.85)
            dist_fig.update_traces(marker_color=PURPLE)
            dist_fig.update_layout(**chart_layout("Sale price distribution"))
            st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
            st.plotly_chart(dist_fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_mid:
            trend_df = model_df.groupby("YrSold", as_index=False)["SalePrice"].median()
            trend_fig = go.Figure()
            trend_fig.add_trace(
                go.Scatter(
                    x=trend_df["YrSold"],
                    y=trend_df["SalePrice"],
                    mode="lines+markers",
                    line={"color": ACCENT, "width": 4},
                    marker={"size": 10, "color": ACCENT},
                    fill="tozeroy",
                    fillcolor="rgba(216,255,98,0.08)",
                )
            )
            trend_fig.update_layout(**chart_layout("Median sale price by year sold"))
            st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
            st.plotly_chart(trend_fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

    if selected_filter in ["All", "Modeling"]:
        chart_bottom_left, chart_bottom_right = st.columns([1.25, 1])
        with chart_bottom_left:
            scatter_fig = px.scatter(
                model_df,
                x="GrLivArea",
                y="SalePrice",
                color="OverallQual",
                color_continuous_scale=[PURPLE, ACCENT],
                hover_data=["Neighborhood", "YearBuilt"],
                opacity=0.72,
            )
            scatter_fig.update_layout(**chart_layout("Living area vs sale price"))
            st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
            st.plotly_chart(scatter_fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_bottom_right:
            comparison_fig = go.Figure()
            comparison_fig.add_trace(
                go.Bar(
                    x=results_df["Model"],
                    y=results_df["RMSE"],
                    marker_color=[ACCENT, PURPLE, "#606060"],
                    text=[money(v) for v in results_df["RMSE"]],
                    textposition="auto",
                )
            )
            comparison_fig.update_layout(**chart_layout("Model comparison by RMSE"))
            st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
            st.plotly_chart(comparison_fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)


def render_insights(model_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    left, right = st.columns([1.2, 1])
    with left:
        neighborhood_df = (
            model_df.groupby("Neighborhood", as_index=False)["SalePrice"]
            .median()
            .sort_values("SalePrice", ascending=False)
            .head(12)
        )
        fig = px.bar(
            neighborhood_df.sort_values("SalePrice"),
            x="SalePrice",
            y="Neighborhood",
            orientation="h",
            color="SalePrice",
            color_continuous_scale=[[0, PURPLE], [1, ACCENT]],
        )
        fig.update_layout(**chart_layout("Top neighborhoods by median price"))
        fig.update_coloraxes(showscale=False)
        st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div class="panel-title">What Drives Price the Most?</div>
                <div class="panel-subtitle">Interpretation panel for the strongest signals in the model.</div>
                <div class="insight-box">
                    <div class="insight-label">Top feature</div>
                    <div class="insight-value">{importance_df.iloc[0]["feature"]}</div>
                    <p class="insight-copy">This transformed feature carries the most weight in the current best model.</p>
                </div>
                <div class="insight-box">
                    <div class="insight-label">Secondary feature</div>
                    <div class="insight-value">{importance_df.iloc[1]["feature"]}</div>
                    <p class="insight-copy">It adds another strong pricing signal and often interacts with overall quality and home size.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analytics(results_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        importance_fig = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0, PURPLE], [1, ACCENT]],
        )
        importance_fig.update_layout(**chart_layout("Feature importance"))
        importance_fig.update_coloraxes(showscale=False)
        st.markdown('<div class="plot-shell">', unsafe_allow_html=True)
        st.plotly_chart(importance_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="panel-title">Executive Summary</div>
                <div class="panel-subtitle">Validation metrics for the three trained regression models.</div>
            """,
            unsafe_allow_html=True,
        )
        summary_df = results_df.copy()
        summary_df["MAE"] = summary_df["MAE"].map(money)
        summary_df["RMSE"] = summary_df["RMSE"].map(money)
        summary_df["R2"] = summary_df["R2"].map(lambda x: f"{x:.3f}")
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_panel(
    base_row: pd.Series, model_df: pd.DataFrame, best_pipeline: Pipeline, best_model_name: str, importance_df: pd.DataFrame
) -> None:
    left, right = st.columns([2.1, 1], gap="large")
    with left:
        prediction_input = build_prediction_input(base_row, model_df)
    with right:
        predicted_price = float(best_pipeline.predict(prediction_input)[0])
        median_price = float(model_df["SalePrice"].median())
        premium_pct = ((predicted_price - median_price) / median_price) * 100
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="insight-label">Predicted sale price</div>
                <div class="insight-value">{money(predicted_price)}</div>
                <p class="insight-copy">{premium_pct:+.1f}% compared with the dataset median using <strong>{best_model_name}</strong>.</p>
            </div>
            <div class="insight-box">
                <div class="insight-label">Best model</div>
                <div class="insight-value">{best_model_name}</div>
                <p class="insight-copy">The app recalculates each estimate using the strongest model selected by validation RMSE.</p>
            </div>
            <div class="insight-box">
                <div class="insight-label">Top signal</div>
                <div class="insight-value">{importance_df.iloc[0]['feature']}</div>
                <p class="insight-copy">This is the highest-impact input in the model scoring pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_reports(results_df: pd.DataFrame, model_df: pd.DataFrame, best_model_name: str) -> None:
    report_df = results_df.copy()
    report_df["MAE"] = report_df["MAE"].round(2)
    report_df["RMSE"] = report_df["RMSE"].round(2)
    report_df["R2"] = report_df["R2"].round(4)

    st.markdown(
        """
        <div class="card outputs-card">
            <div class="panel-title">Downloadable Outputs</div>
            <div class="panel-subtitle">Export a CSV with model KPIs and a concise business summary.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    summary_text = (
        f"Best model: {best_model_name}\n"
        f"Dataset rows after cleaning: {model_df.shape[0]}\n"
        f"Median sale price: {money(float(model_df['SalePrice'].median()))}\n"
    )
    col_1, col_2 = st.columns(2)
    with col_1:
        st.download_button(
            "Download metrics CSV",
            data=csv_bytes,
            file_name="housing_model_metrics.csv",
            mime="text/csv",
        )
    with col_2:
        st.download_button(
            "Download summary TXT",
            data=summary_text,
            file_name="housing_summary.txt",
            mime="text/plain",
        )


def render_dashboard() -> None:
    inject_styles()
    try:
        assets = train_assets()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    raw_df = assets["raw_df"]
    model_df = assets["model_df"]
    results_df = assets["results_df"]
    importance_df = assets["importance_df"]
    best_model_name = assets["best_model_name"]
    best_pipeline = assets["best_pipeline"]

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">Housing Price Intelligence</div>
            <div class="hero-sub">A live machine learning dashboard for exploratory analysis, model benchmarking, and instant property valuation with the Kaggle House Prices dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_view = render_top_navigation(best_model_name)

    st.write("")
    pred_col, rmse_col, r2_col, rows_col = st.columns([1.15, 1, 1, 1])
    baseline_price = float(model_df["SalePrice"].median())
    with pred_col:
        metric_card("Median Sale Price", money(baseline_price), "Benchmark from the cleaned training dataset")
    with rmse_col:
        metric_card("Validation RMSE", money(float(results_df.loc[0, "RMSE"])), "Lower values indicate stronger model performance")
    with r2_col:
        metric_card("Best Model R-squared", f"{float(results_df.loc[0, 'R2']):.3f}", "Explained variance on the holdout split")
    with rows_col:
        metric_card("Cleaned Training Rows", f"{model_df.shape[0]:,}", f"{assets['outliers_removed']} outliers removed", accent=True)

    st.write("")

    if current_view == "Overview":
        render_overview(raw_df, model_df, results_df)
    elif current_view == "Insights":
        render_insights(model_df, importance_df)
    elif current_view == "Analytics":
        render_analytics(results_df, importance_df)
    elif current_view == "Prediction":
        render_prediction_panel(assets["base_row"], model_df, best_pipeline, best_model_name, importance_df)
    elif current_view == "Reports":
        render_reports(results_df, model_df, best_model_name)


render_dashboard()

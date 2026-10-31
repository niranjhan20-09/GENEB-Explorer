
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from itertools import product
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# -----------------------------------------------
# WEBSITE CONFIGURATION
# -----------------------------------------------

st.set_page_config(
    page_title="GENEB Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #edf2ff;
}
[data-testid="stMetric"] {
    background-color: #151d35;
    border: 1px solid #283754;
    border-radius: 12px;
    padding: 15px;
}
h1, h2, h3 {
    color: #d6cbff;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

DATASETS = {
    "Human Non-TATA Promoters": "promoters",
    "Human Enhancers (Cohn)": "enhancers",
    "Coding vs Intergenomic DNA": "coding",
    "Human vs Worm": "species"
}

# Class meanings from the upstream Genomic Benchmarks task metadata.
# The numeric codes follow the alphabetical class order in the source datasets.
BIOLOGICAL_LABELS = {
    "promoters": {0: "Non-promoter DNA sequence", 1: "Human non-TATA promoter sequence"},
    "enhancers": {0: "Non-enhancer (background) sequence", 1: "Human enhancer sequence"},
    "coding": {0: "Protein-coding sequence", 1: "Intergenic DNA sequence"},
    "species": {0: "Human DNA (Homo sapiens)", 1: "Worm DNA (Caenorhabditis elegans)"},
}

MODEL_NAMES = [
    "Logistic Regression Original",
    "Logistic Regression Modified",
    "Random Forest Original",
    "Random Forest Modified",
    "SVM Original",
    "SVM Modified",
    "MLP Original",
    "MLP Modified"
]

EXPLANATIONS = {
    "Logistic Regression": {
        "principle": "Uses a sigmoid function to model class probability.",
        "original": "64 DNA k-mer features and regularized logistic regression.",
        "modified": "Selects 32 features and tunes regularization strength.",
        "finding": "The modified version had lower MCC on all four datasets."
    },
    "Random Forest": {
        "principle": "Combines predictions from multiple decision trees.",
        "original": "100 decision trees with default depth constraints.",
        "modified": "200 trees, restricted depth and leaf size, balanced classes.",
        "finding": "Modification showed mixed results across datasets."
    },
    "SVM": {
        "principle": "Finds a decision boundary that separates classes.",
        "original": "Linear-kernel Support Vector Machine.",
        "modified": "RBF kernel with optimized C and gamma.",
        "finding": "Modified SVM achieved the highest MCC on all four datasets."
    },
    "MLP": {
        "principle": "Learns patterns using interconnected artificial neurons.",
        "original": "One hidden layer containing 100 neurons.",
        "modified": "Two hidden layers (128, 64), L2 regularization and early stopping.",
        "finding": "Modified MLP improved MCC on every dataset."
    }
}

@st.cache_data
def read_results():
    frames = []

    for label, folder in DATASETS.items():
        path = RESULTS / f"{folder}.csv"
        df = pd.read_csv(path)
        df["Dataset"] = label
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

df = read_results()

# -----------------------------------------------
# NAVIGATION
# -----------------------------------------------

with st.sidebar:
    st.title("🧬 GENEB Explorer")

    page = st.radio(
        "Explore",
        [
            "Overview",
            "Model Laboratory",
            "Performance Explorer",
            "Live DNA Predictor",
            "Research Conclusions"
        ]
    )

    st.divider()
    st.caption("Genomic ML comparative study")
    st.caption("4 datasets | 4 algorithms | 32 experiments")

# -----------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------

if page == "Overview":

    st.title("🧬 GENEB Explorer")

    st.subheader(
        "Interactive Comparative Analysis of "
        "Machine Learning Models for Genomic DNA Classification"
    )

    st.write(
        "An experimental study comparing four existing machine "
        "learning algorithms and their modified versions across "
        "four genomic classification datasets."
    )

    a, b, c, d = st.columns(4)

    a.metric("Datasets", 4)
    b.metric("ML Algorithms", 4)
    c.metric("Model Configurations", 8)
    d.metric("Training Runs", 32)

    st.divider()

    st.header("How the system works")

    st.info(
        "DNA Sequence → 3-mer Extraction → "
        "64 Numerical Features → ML Classifier → Prediction"
    )

    st.header("Datasets")

    for dataset in DATASETS:
        st.markdown(f"**• {dataset}**")

    st.header("Key Finding")

    st.success(
        "The modified SVM achieved the highest "
        "MCC on all four evaluated datasets."
    )

    st.caption(
        "This project is a simplified extension of the GENEB "
        "research, using 3-mer frequencies and conventional "
        "ML classifiers instead of pretrained foundation-model embeddings."
    )

# -----------------------------------------------
# PAGE 2: MODEL LABORATORY
# -----------------------------------------------

elif page == "Model Laboratory":

    st.title("🔬 Model Laboratory")

    st.write(
        "Explore how the four algorithms operate "
        "and what was modified in each."
    )

    algorithm = st.selectbox(
        "Choose an ML algorithm",
        list(EXPLANATIONS.keys())
    )

    info = EXPLANATIONS[algorithm]

    st.subheader(algorithm)

    st.markdown("### Working Principle")
    st.write(info["principle"])

    left, right = st.columns(2)

    with left:
        st.markdown("### Original Model")
        st.info(info["original"])

    with right:
        st.markdown("### Modified Model")
        st.success(info["modified"])

    st.markdown("### Experimental Observation")
    st.write(info["finding"])

    st.markdown("### MCC Across Datasets")

    subset = df[
        df["Model"].str.startswith(algorithm)
    ]

    fig = px.bar(
        subset,
        x="Dataset",
        y="MCC",
        color="Model",
        barmode="group",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------
# PAGE 3: PERFORMANCE EXPLORER
# -----------------------------------------------

elif page == "Performance Explorer":

    st.title("📊 Performance Explorer")

    dataset = st.selectbox(
        "Select genomic dataset",
        list(DATASETS.keys())
    )

    available_metrics = [
        metric for metric in
        ["MCC", "F1", "Accuracy", "Time (s)"]
        if metric in df.columns
    ]

    metric = st.selectbox(
        "Select performance metric",
        available_metrics
    )

    selected = df[df["Dataset"] == dataset]

    fig = px.bar(
        selected,
        x="Model",
        y=metric,
        color="Model",
        template="plotly_dark",
        title=f"{metric} Comparison — {dataset}"
    )

    fig.update_layout(xaxis_tickangle=-40)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Results")
    st.dataframe(
        selected.drop(columns=["Dataset"]),
        use_container_width=True
    )

    st.subheader("Original vs Modified — MCC Change")

    changes = []

    for algorithm in EXPLANATIONS:
        original = selected.loc[
            selected["Model"] == f"{algorithm} Original",
            "MCC"
        ].iloc[0]

        modified = selected.loc[
            selected["Model"] == f"{algorithm} Modified",
            "MCC"
        ].iloc[0]

        changes.append({
            "Algorithm": algorithm,
            "Original": original,
            "Modified": modified,
            "MCC Change": modified - original
        })

    st.dataframe(
        pd.DataFrame(changes),
        use_container_width=True
    )

# -----------------------------------------------
# PAGE 4: LIVE PREDICTOR
# -----------------------------------------------

elif page == "Live DNA Predictor":

    st.title("🧬 Live DNA Sequence Classifier")

    st.write(
        "Enter a DNA sequence and classify it "
        "using one of the previously trained models."
    )

    dataset = st.selectbox(
        "Classification task",
        list(DATASETS.keys())
    )

    model_name = st.selectbox(
        "Choose model",
        MODEL_NAMES
    )

    dna = st.text_area(
        "Enter DNA sequence (A, C, G, T)",
        value="ATGCGTACCTGATCGATGCTAGCTAGCTAGC",
        height=150
    )

    def sequence_to_features(seq):
        vocabulary = [
            "".join(chars)
            for chars in product("ACGT", repeat=3)
        ]

        def dna_to_kmers(sequence):
            return [
                sequence[i:i+3]
                for i in range(len(sequence)-2)
                if set(sequence[i:i+3]) <= set("ACGT")
            ]

        vectorizer = CountVectorizer(
            analyzer=dna_to_kmers,
            vocabulary=vocabulary,
            lowercase=False
        )

        x = vectorizer.transform([seq.upper()])
        return normalize(x, norm="l1").toarray()

    if st.button("Predict DNA Classification", type="primary"):

        cleaned = "".join(dna.split()).upper()

        if len(cleaned) < 3:
            st.error("Enter a DNA sequence at least 3 bases long.")

        elif not set(cleaned).issubset(set("ACGT")):
            st.error("Only A, C, G and T nucleotides are supported.")

        else:
            folder = DATASETS[dataset]
            filename = model_name.replace(" ", "_") + ".joblib"
            model_path = MODELS / folder / filename

            if not model_path.exists():
                st.error(f"Model file not found: {model_path.name}")

            else:
                # Only load model files generated by your own
                # trusted training notebook.
                model = load(model_path)

                features = sequence_to_features(cleaned)

                prediction = model.predict(features)[0]

                # Preserve the original model output; only its displayed
                # meaning changes. This does not relabel or retrain the model.
                try:
                    numeric_label = int(prediction)
                except (ValueError, TypeError):
                    numeric_label = None

                biological_label = BIOLOGICAL_LABELS[folder].get(numeric_label)
                if biological_label is None:
                    st.warning(f"Unrecognized model class: {prediction!s}")
                else:
                    st.success(f"Predicted classification: {biological_label}")
                    st.caption(f"Model output: class {numeric_label}")

                st.info(
                    "This is a sequence-pattern classification based on a "
                    "research dataset, not a validated biological or "
                    "clinical diagnosis. Use task-relevant DNA sequences."
                )

# -----------------------------------------------
# PAGE 5: RESEARCH CONCLUSIONS
# -----------------------------------------------

elif page == "Research Conclusions":

    st.title("📚 Research Conclusions")

    st.markdown("### 1. Task-dependent performance")

    st.write(
        "Model performance varied significantly across the "
        "four genomic classification datasets."
    )

    st.markdown("### 2. Nonlinear classification")

    st.write(
        "The modified RBF SVM produced the highest MCC "
        "across all four selected datasets."
    )

    st.markdown("### 3. Neural network modifications")

    st.write(
        "The modified MLP improved MCC on every dataset, "
        "suggesting the combined changes were beneficial "
        "in these experiments."
    )

    st.markdown("### 4. Feature selection")

    st.write(
        "Selecting only 32 of 64 k-mer features reduced "
        "Logistic Regression MCC across all four tasks."
    )

    st.markdown("### 5. Limitations")

    st.write(
        "Experiments used four datasets, one random seed "
        "and handcrafted 3-mer frequencies. The individual "
        "effects of each model modification were not isolated."
    )

    st.markdown("### Connection to GENEB")

    st.write(
        "Our results illustrate the importance of "
        "task-aware evaluation, consistent with the main "
        "motivation of the GENEB research paper."
    )

    st.caption(
        "Reference: Ledneva et al. (2026), "
        "GENEB: Why Genomic Models Are Hard to Compare."
    )

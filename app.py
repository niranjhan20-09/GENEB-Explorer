"""GENEB-inspired classroom comparison of conventional genomic classifiers."""
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load

from dna_features import MAX_BASES, clean_sequence, sequence_to_features

st.set_page_config(page_title="GENEB Explorer", page_icon="🧬", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"] {border:1px solid #344467;border-radius:12px;padding:16px;}
textarea {font-family:monospace !important;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
METADATA = json.loads((ROOT / "training_metadata.json").read_text())
DEMO_SEQUENCES = json.loads((ROOT / "demo_sequences.json").read_text())
DATASETS = {
    "Human Non-TATA Promoters": "promoters",
    "Human Enhancers (Cohn)": "enhancers",
    "Coding vs Intergenomic DNA": "coding",
    "Human vs Worm": "species",
}
TASKS = {
    "promoters": "Distinguish promoter sequences from the dataset's negative examples.",
    "enhancers": "Distinguish enhancer sequences from the dataset's negative examples.",
    "coding": "Distinguish coding sequences from intergenic sequences.",
    "species": "Distinguish human sequences from worm sequences.",
}
ALGORITHMS = ["Logistic Regression", "Random Forest", "SVM", "MLP"]
MODEL_NAMES = [f"{algorithm} {variant}" for algorithm in ALGORITHMS
               for variant in ["Original", "Modified"]]
COLORS = {"Baseline": "#a5b4fc", "Modified": "#2dd4bf"}
EXPLANATIONS = {
    "Logistic Regression": (
        "Maps a weighted sum of features through a sigmoid to estimate a binary class probability.",
        "StandardScaler → LogisticRegression; C=1, max_iter=1000, random_state=42.",
        "SelectKBest(f_classif, k=32) → StandardScaler → LogisticRegression; C searched over [0.1, 1, 10], three-fold CV scored by MCC.",
    ),
    "Random Forest": (
        "Combines the predictions of many decision trees.",
        "100 trees; unrestricted max_depth; random_state=42, n_jobs=-1.",
        "200 trees; max_depth=15, min_samples_leaf=4, max_features=sqrt, class_weight=balanced, random_state=42.",
    ),
    "SVM": (
        "Learns a separating boundary; an RBF kernel can represent nonlinear boundaries in feature space.",
        "StandardScaler → SVC(kernel=linear); C=1 (default).",
        "StandardScaler → SVC(kernel=rbf); C ∈ [0.1, 1, 10], gamma ∈ [scale, 0.01]; three-fold CV scored by MCC.",
    ),
    "MLP": (
        "Learns nonlinear relationships through interconnected layers of artificial neurons.",
        "StandardScaler → MLP(100); max_iter=300, random_state=42.",
        "StandardScaler → MLP(128, 64); alpha=0.01, early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=42.",
    ),
}
METRIC_HELP = {
    "MCC": "Matthews correlation coefficient: −1 is total disagreement, 0 is no correlation, and +1 is perfect agreement. Higher is better.",
    "F1": "Macro-F1: unweighted mean of class-wise F1 scores, combining precision and recall. Higher is better; each class contributes equally.",
    "Accuracy": "Fraction of examples classified correctly. Higher is better; interpret alongside class balance and MCC.",
    "Time (s)": "Training fit time in seconds, including GridSearchCV search and refit where used. Excludes feature extraction and test prediction. Colab hardware was not recorded, so comparisons are descriptive; this is not inference latency.",
}
SYNTHETIC = {
    "Balanced repeat · synthetic": "ACGT" * 50,
    "Varied sequence · synthetic": (
        "ATGCGTACGTTAGCCTAGGCTAACGTTCGATCGGATCCATGCTAGCTTACGATC"
        "GATGGCATTCGATCGTAGCTAGGCTAATCGGATCGTACGCTAGCATG" * 2
    ),
    "GC-rich repeat · synthetic": "GCGCGCAT" * 25,
}


@st.cache_data
def read_results():
    frames = []
    for dataset, folder in DATASETS.items():
        frame = pd.read_csv(ROOT / "results" / f"{folder}.csv")
        required = {"Model", "Accuracy", "F1", "MCC", "Time (s)"}
        if not required.issubset(frame.columns) or set(frame["Model"]) != set(MODEL_NAMES) or len(frame) != 8:
            raise ValueError(f"Incomplete experiment results for {dataset}.")
        frame["Dataset"] = dataset
        frame["Algorithm"] = frame["Model"].str.replace(r" (Original|Modified)$", "", regex=True)
        frame["Configuration"] = frame["Model"].str.split().str[-1].replace({"Original": "Baseline"})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@st.cache_resource(max_entries=8)
def load_model(folder, model_name):
    # Load only the repository owner's trusted, pre-trained artifacts.
    path = ROOT / "models" / folder / (model_name.replace(" ", "_") + ".joblib")
    if not path.is_file():
        raise FileNotFoundError("The selected trained model is unavailable.")
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise FileNotFoundError("The trained model has not been downloaded from Git LFS.")
    return load(path)


def remember(widget_key, permanent_key):
    st.session_state[permanent_key] = st.session_state[widget_key]


def choose(label, options, key, **kwargs):
    st.session_state.setdefault(key, options[0])
    widget = "_" + key
    st.session_state[widget] = st.session_state[key]
    return st.selectbox(label, options, key=widget, on_change=remember,
                        args=(widget, key), **kwargs)


def style_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1020",
                      plot_bgcolor="#0b1020", font=dict(color="#edf2ff", size=14),
                      legend_title_text="Configuration", margin=dict(l=10, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displaylogo": False})


def changes_for(frame):
    table = frame.pivot(index="Algorithm", columns="Configuration", values="MCC").reindex(ALGORITHMS)
    table["MCC change"] = table["Modified"] - table["Baseline"]
    return table.reset_index()


def methodology():
    with st.expander("Methodology and reproducibility"):
        st.write("Source: darlednik/geneb-tasks, using each task's released train.csv and test.csv. "
                 "The notebook consumes numeric labels directly; it does not re-encode them.")
        st.write("Overlapping canonical DNA 3-mers are L1-normalized into 64 frequencies. Promoters use a "
                 "vocabulary learned from training data; the other tasks use a fixed A/C/G/T vocabulary. "
                 "The inspected promoter training snapshot contains all 64 3-mers in the same order.")
        st.write("Modified logistic regression and SVM use GridSearchCV(cv=3, scoring=matthews_corrcoef). "
                 "Feature selection and StandardScaler are inside the pipeline, so they are fitted within "
                 "training folds during tuning. The shown code does not use the supplied test labels for tuning.")
        st.write("Random-state-controlled estimators use seed 42. Modified MLP reserves 15% of its training "
                 "data for early stopping. Evaluation uses accuracy, macro-F1 and MCC on the provided test split.")
        st.write("Time measures model.fit, including search and refit for tuned models. It excludes feature "
                 "extraction and test prediction. There are 32 reported evaluations, with additional fits during tuning.")
        st.caption("Dataset counts and lengths were checked against source revision "
                   "005717000cfffd8d506d1bdd9c5c3c458895d428. The original notebook did not pin a "
                   "data revision or record hardware. These checks do not establish chromosome-level or "
                   "sequence-similarity independence between source splits.")
        st.warning("One seed, no uncertainty intervals and no ablations. The notebook also records a "
                   "convergence warning during modified MLP training on Human vs Worm.")
        rows = []
        for label, folder in DATASETS.items():
            meta = METADATA[folder]
            rows.append({"Task": label, "Train": meta["train"]["count"], "Test": meta["test"]["count"],
                         "Sequence length (bases)": meta["train"]["min_length"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.link_button("Source task collection", "https://huggingface.co/datasets/darlednik/geneb-tasks")


try:
    df = read_results()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Experiment results could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.title("🧬 GENEB Explorer")
    page = st.radio("Explore", ["Overview", "Model Laboratory", "Performance Explorer",
                                "Live DNA Predictor", "Research Conclusions"])
    st.divider()
    st.caption("GENEB-inspired comparative study")
    st.caption("4 datasets · 4 algorithms · 32 evaluations")
    st.caption("Experimental university project")

if page == "Overview":
    st.title("🧬 GENEB Explorer")
    st.subheader("A GENEB-inspired comparison of DNA classifiers")
    st.write("How do four conventional machine learning algorithms and their modified configurations "
             "perform on four genomic classification tasks using the same 3-mer representation?")
    for column, label, value in zip(st.columns(4), ["Datasets", "Algorithms", "Configurations", "Evaluations"], [4, 4, 8, len(df)]):
        column.metric(label, value)
    st.header("From DNA to prediction")
    st.info("DNA sequence → overlapping 3-mers → 64 normalized frequencies → saved classifier → predicted label")
    with st.expander("Worked example: ACGTAC"):
        st.code("ACG · CGT · GTA · TAC\nEach occurs once: frequency = 1/4 = 0.25")
        st.write("There are 4³ = 64 possible 3-mers. Frequencies summarize short local patterns; "
                 "they do not preserve the positions of those patterns or long-range sequence relationships.")
    st.header("Four biological questions")
    columns = st.columns(2)
    for index, (label, folder) in enumerate(DATASETS.items()):
        with columns[index % 2].container(border=True):
            st.subheader(label)
            st.write(TASKS[folder])
            meta = METADATA[folder]
            st.caption(f"Source snapshot: {meta['train']['count']:,} train · {meta['test']['count']:,} test · {meta['train']['min_length']} bases")
    st.header("Results at a glance")
    matrix = df.pivot(index="Model", columns="Dataset", values="MCC").reindex(index=MODEL_NAMES, columns=list(DATASETS))
    matrix.index = matrix.index.str.replace(" Original", " Baseline", regex=False)
    fig = px.imshow(matrix, text_auto=".3f", aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu", labels={"color": "MCC"}, height=480)
    fig.update_xaxes(side="top")
    style_chart(fig)
    st.caption("One reported value per configuration and task; no uncertainty intervals are available.")
    st.write("Use Performance Explorer to examine improvements and their recorded time costs.")
    methodology()
    st.caption("Inspired by GENEB's emphasis on consistent, task-aware evaluation. This project uses "
               "conventional classifiers on handcrafted features rather than reproducing its foundation-model benchmark.")

elif page == "Model Laboratory":
    st.title("🔬 Model Laboratory")
    st.write("Compare baseline and modified configurations, and inspect the observed effect on MCC.")
    algorithm = choose("Choose an ML algorithm", ALGORITHMS, "lab_algorithm")
    principle, baseline, modified = EXPLANATIONS[algorithm]
    st.subheader(algorithm)
    st.write(principle)
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Baseline")
        st.write(baseline)
    with right.container(border=True):
        st.subheader("Modified")
        st.write(modified)
    st.caption("Configurations and search grids verified in GENEB_ML_PROJECT.ipynb. Baseline corresponds "
               "to Original in saved artifacts. Search grids are not the same as selected fitted parameters.")
    with st.expander("Verified selected parameters"):
        st.write("Saved modified logistic regression artifacts: C=1 for promoters; C=0.1 for enhancers, coding and species. "
                 "Notebook outputs: modified SVM has C=1, gamma=scale for coding and species. "
                 "Selected SVM settings for the other two tasks were not retained in the inspected outputs.")
    subset = df[df["Algorithm"] == algorithm]
    fig = px.bar(subset, y="Dataset", x="MCC", color="Configuration", barmode="group",
                 orientation="h", color_discrete_map=COLORS, text_auto=".3f", height=440,
                 category_orders={"Configuration": ["Baseline", "Modified"]})
    fig.update_xaxes(range=[min(0, subset["MCC"].min()), 1])
    style_chart(fig)
    effects = subset.pivot(index="Dataset", columns="Configuration", values="MCC")
    improved = int((effects["Modified"] > effects["Baseline"]).sum())
    st.write(f"The modified configuration improved observed MCC on {improved} of {len(effects)} tasks.")
    st.caption("Changes were combined; this comparison cannot identify which individual change caused the difference.")

elif page == "Performance Explorer":
    st.title("📊 Performance Explorer")
    dataset = choose("Select genomic dataset", list(DATASETS), "performance_dataset")
    metric = choose("Select performance metric", list(METRIC_HELP), "performance_metric")
    st.caption(METRIC_HELP[metric])
    selected = df[df["Dataset"] == dataset]
    best = selected.loc[selected["MCC"].idxmax()]
    first, second = st.columns(2)
    first.metric("Highest observed MCC", f"{best['MCC']:.3f}")
    second.metric("Recorded time for that configuration", f"{best['Time (s)']:,.2f} s")
    st.caption(best["Model"].replace(" Original", " Baseline"))
    fig = px.bar(selected, y="Algorithm", x=metric, color="Configuration", barmode="group",
                 orientation="h", text_auto=".3f" if metric != "Time (s)" else ".2f",
                 color_discrete_map=COLORS, height=460,
                 category_orders={"Configuration": ["Baseline", "Modified"], "Algorithm": ALGORITHMS},
                 title=f"{metric} comparison — {dataset}")
    if metric != "Time (s)":
        fig.update_xaxes(range=[min(0, selected[metric].min()), 1])
    style_chart(fig)
    st.subheader("Detailed results")
    visible = selected[["Model", "Accuracy", "F1", "MCC", "Time (s)"]].copy()
    visible["Model"] = visible["Model"].str.replace(" Original", " Baseline", regex=False)
    st.dataframe(visible.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}", "Time (s)": "{:,.2f}"}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download these results (CSV)", visible.to_csv(index=False),
                       file_name=f"{DATASETS[dataset]}_results.csv", mime="text/csv")
    st.subheader("Baseline versus modified: MCC change")
    changes = changes_for(selected)
    st.dataframe(changes.style.format({"Baseline": "{:.4f}", "Modified": "{:.4f}", "MCC change": "{:+.4f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Positive change means higher observed MCC. Small differences are not evidence of statistical significance.")
    with st.expander("Performance versus recorded time"):
        style_chart(px.scatter(selected, x="Time (s)", y="MCC", color="Configuration", symbol="Algorithm",
                               hover_name="Model", color_discrete_map=COLORS, log_x=True,
                               title="Higher MCC and lower recorded time are preferable"))
        st.caption("Logarithmic time axis. Timing comparisons require consistent hardware and timing boundaries.")

elif page == "Live DNA Predictor":
    st.title("🧬 Live DNA Predictor")
    st.write("Explore how a saved classifier responds to a DNA sequence. Predictions are experimental "
             "and do not establish biological function, species identity or a diagnosis.")
    dataset = choose("Classification task", list(DATASETS), "prediction_dataset")
    st.caption(TASKS[DATASETS[dataset]])
    model_name = choose("Choose model", MODEL_NAMES, "prediction_model",
                        format_func=lambda name: name.replace(" Original", " Baseline"))
    examples = {f"Benchmark test row {row['row_index']} · label {row['label']}": row["sequence"]
                for row in DEMO_SEQUENCES[DATASETS[dataset]]}
    examples.update(SYNTHETIC)
    with st.expander("Load an example sequence"):
        example = st.selectbox("Example sequence", list(examples))
        if st.button("Use this example"):
            st.session_state["dna_text"] = examples[example]
            st.session_state.pop("prediction_result", None)
        st.caption("Benchmark examples come from the inspected source test snapshot and retain their numeric labels. "
                   "The original training download was not revision-pinned. Synthetic examples have no known class; "
                   "a single demonstration does not measure accuracy.")
        st.link_button("Example provenance", METADATA[DATASETS[dataset]]["source_url"])
    st.session_state.setdefault("dna_text", SYNTHETIC[next(iter(SYNTHETIC))])
    st.session_state["_dna_input"] = st.session_state["dna_text"]
    dna = st.text_area("Enter DNA sequence (A, C, G, T)", key="_dna_input", height=150,
                       on_change=remember, args=("_dna_input", "dna_text"),
                       help=f"One plain sequence, up to {MAX_BASES:,} bases. Lowercase and whitespace are normalized. FASTA headers and ambiguous bases are unsupported.")
    cleaned = "".join(dna.split()).upper()
    valid_alphabet = bool(cleaned) and set(cleaned) <= set("ACGT")
    if valid_alphabet:
        length, gc = st.columns(2)
        length.metric("Sequence length", f"{len(cleaned):,} bases")
        gc.metric("GC content", f"{100 * (cleaned.count('G') + cleaned.count('C')) / len(cleaned):.1f}%")
    expected_length = METADATA[DATASETS[dataset]]["train"]["min_length"]
    st.caption(f"The inspected benchmark snapshot uses {expected_length}-base sequences for this task.")
    if valid_alphabet and len(cleaned) != expected_length:
        st.warning(f"This input has {len(cleaned):,} bases; the source task uses {expected_length}. "
                   "The app can compute its features, but the prediction is outside the task's length setting.")
    if valid_alphabet and len(set(cleaned[i:i+3] for i in range(max(0, len(cleaned)-2)))) < 8:
        st.caption("This sequence has very few distinct 3-mers. Repetitive examples are useful for "
                   "demonstrating the interface but may not resemble the benchmark's biological inputs.")
    signature = (dataset, model_name, cleaned)
    if st.button("Predict DNA classification", type="primary"):
        st.session_state.pop("prediction_result", None)
        try:
            sequence = clean_sequence(dna)
            with st.spinner("Running the selected classifier…"):
                model = load_model(DATASETS[dataset], model_name)
                prediction = model.predict(sequence_to_features(sequence))[0]
            st.session_state["prediction_result"] = {"signature": signature, "label": str(prediction)}
        except ValueError as exc:
            st.error(f"Prediction could not be completed: {exc}")
        except FileNotFoundError as exc:
            st.error(f"{exc} Please try another model or ask the project owner to restore the artifact.")
        except Exception:
            st.error("The selected model could not complete the prediction. Please ask the project owner to check its saved preprocessing and library versions.")
    result = st.session_state.get("prediction_result")
    if result and result["signature"] == signature:
        st.subheader("Prediction result")
        class_name = METADATA[DATASETS[dataset]]["label_names"].get(result["label"])
        if class_name:
            st.success(f"Predicted class: {class_name} · dataset label {result['label']}")
        else:
            st.info(f"Predicted dataset label: {result['label']}")
        st.write(f"**Task:** {dataset} · **Model:** {model_name.replace(' Original', ' Baseline')}")
        if METADATA[DATASETS[dataset]]["label_mapping_status"] != "source_verified":
            st.caption("Class names use the mapping configured in this project. Its numeric-to-biological "
                       "correspondence still needs independent source verification.")
        st.caption("A class label is not a probability or confidence score.")
    elif result:
        st.caption("Input or model changed. Run prediction again to see a matching result.")

else:
    st.title("📚 Research Conclusions")
    st.subheader("What these experiments show")
    winners = df.loc[df.groupby("Dataset")["MCC"].idxmax(), ["Dataset", "Model", "MCC"]]
    st.dataframe(winners.style.format({"MCC": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.write("The modified RBF SVM has the highest observed MCC on all four evaluated datasets. "
             "This finding applies to these configurations and reported evaluations, not to all genomic tasks.")
    st.write("The modified MLP improved observed MCC on all four tasks. Logistic regression's modified "
             "configuration had lower MCC on all four. Random forest changes produced mixed outcomes.")
    st.subheader("What these experiments do not establish")
    st.write("The effects of feature selection, regularization, architecture and tuning were not isolated. "
             "For example, the logistic-regression comparison does not prove that feature selection alone caused the decrease.")
    st.write("No repeated-seed estimates or significance tests are provided. Results do not establish "
             "statistical superiority, reliable predictions on arbitrary DNA, or clinical utility.")
    st.subheader("Next experiments")
    st.markdown("- Verify the split and preprocessing procedure; keep the test set untouched during tuning.\n"
                "- Repeat with multiple seeds and report uncertainty.\n"
                "- Run ablations that change one component at a time.\n"
                "- Add held-out examples with verified labels and confusion matrices from saved predictions.")
    methodology()
    st.subheader("Connection to GENEB")
    st.write("GENEB studies frozen representations from 40 genomic foundation models across 100 tasks. "
             "This smaller project is inspired by its motivation for consistent, task-aware comparison; "
             "it compares conventional classifiers using handcrafted 3-mer frequencies.")
    st.link_button("Read GENEB — Ledneva et al. (2026)", "https://proceedings.mlr.press/v306/ledneva26b.html")
    st.link_button("Genomic Benchmarks dataset reference", "https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks")

"""Regression tests; the UI fixture uses a synthetic model, not project weights."""
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dna_features import MAX_BASES, VOCABULARY, clean_sequence, sequence_to_features

@pytest.mark.parametrize('sequence', ['ACG', 'ACGT'*50, 'A'*300, 'GATTACA'*41])
def test_original_feature_compatibility(sequence):
    def analyzer(seq):
        return [seq[i:i+3] for i in range(len(seq)-2) if set(seq[i:i+3]) <= set('ACGT')]
    original = CountVectorizer(analyzer=analyzer, vocabulary=[''.join(c) for c in product('ACGT', repeat=3)], lowercase=False)
    expected = normalize(original.transform([sequence.upper()]), norm='l1').toarray()
    np.testing.assert_array_equal(sequence_to_features(sequence), expected)
    assert expected.shape == (1,64)
    assert expected.sum() == pytest.approx(1)

def test_worked_example():
    features = sequence_to_features(' acg\nta c ')[0]
    for kmer in ['ACG','CGT','GTA','TAC']:
        assert features[VOCABULARY.index(kmer)] == .25
    assert clean_sequence(' acg\nt ') == 'ACGT'

@pytest.mark.parametrize('sequence', ['', 'AC', 'ACGN', '>id\nACGT', 'A'*(MAX_BASES+1)])
def test_input_validation(sequence):
    with pytest.raises(ValueError):
        clean_sequence(sequence)

@pytest.fixture
def app_path(tmp_path):
    for name in ['app.py','dna_features.py','training_metadata.json','demo_sequences.json']:
        shutil.copy(ROOT/name, tmp_path/name)
    shutil.copytree(ROOT/'results', tmp_path/'results')
    target = tmp_path/'models'/'promoters'
    target.mkdir(parents=True)
    model = DummyClassifier(strategy='constant', constant=1).fit(np.zeros((2,64)), [0,1])
    dump(model, target/'Logistic_Regression_Original.joblib')
    return tmp_path/'app.py'

def test_all_sections_and_chart_choices(app_path):
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    for page in ['Model Laboratory','Performance Explorer','Research Conclusions','Overview']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page == 'Model Laboratory':
            for name in ['Logistic Regression','Random Forest','SVM','MLP']:
                app.selectbox[0].select(name).run()
                assert not app.exception
        if page == 'Performance Explorer':
            for dataset in list(app.selectbox[0].options):
                app.selectbox[0].select(dataset).run()
                for metric in ['MCC','F1','Accuracy','Time (s)']:
                    app.selectbox[1].select(metric).run()
                    assert not app.exception
                    assert len(app.dataframe[0].value) == 8

def test_predictor_and_state(app_path):
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    app.sidebar.radio[0].set_value('Live DNA Predictor').run()
    app.text_area[0].input('AC').run()
    app.button[1].click().run()
    assert 'at least 3' in app.error[0].value
    app.text_area[0].input('acgt'*50).run()
    app.button[1].click().run()
    assert not app.exception and not app.error
    assert any('dataset label 1' in item.value for item in app.success)
    app.text_area[0].input('GATTACA'*40).run()
    assert not app.success
    app.sidebar.radio[0].set_value('Overview').run()
    app.sidebar.radio[0].set_value('Live DNA Predictor').run()
    assert app.text_area[0].value == 'GATTACA'*40
    app.selectbox[1].select('SVM Modified').run()
    app.button[1].click().run()
    assert not app.exception
    assert 'unavailable' in app.error[0].value


def test_species_label_regression():
    # Exact matches against semantic source labels showed that alphabetical
    # naming reverses the GENEB encoding for this task.
    import json
    metadata = json.loads((ROOT / "training_metadata.json").read_text())
    assert metadata["species"]["label_names"]["0"] == "Worm DNA (Caenorhabditis elegans)"
    assert metadata["species"]["label_names"]["1"] == "Human DNA (Homo sapiens)"

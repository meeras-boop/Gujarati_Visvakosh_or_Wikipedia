"""
streamlit_UI.py
================
Deployment UI for the 17 dual-task Gujarati models created by
Gujarati_Visvakosh_Wikipedia_17_Models_Colab.py.

Repository layout expected:
    streamlit_UI.py
    requirements.txt
    trained_models/
        AdaBoost.pkl
        BernoulliNB.pkl
        ... 17 model bundles ...

Each .pkl bundle must contain:
    source_model       -> 0=Visvakosh, 1=Wikipedia
    category_model     -> one of seven categories
    feature_artifacts  -> fitted scalers + word/char TF-IDF vectorizers
    model_feature_names
    feature_view       -> handcrafted_62 or full

IMPORTANT EXPLANATION PRINCIPLE
-------------------------------
The trained estimator makes the prediction. The explanation code NEVER changes
or overrides that prediction. Explanations are generated from model-specific
learned evidence:
  * linear models: local x_j * coefficient contribution
  * BernoulliNB: class-conditional Bernoulli log evidence
  * DecisionTree: exact path followed in the learned tree
  * tree ensembles: learned feature importance + local feature ablation
  * KNN: actual nearest-neighbour labels/distances + ablation
  * LDA / MLP / RBF-SVC: actual local prediction sensitivity by feature-block
    and handcrafted-feature ablation
"""

import os
import glob
import json
import math
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack, issparse

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PAGE
# =============================================================================

st.set_page_config(
    page_title="Gujarati Vishwakosh vs Wikipedia",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
<style>
.main-title {font-size:2.35rem;font-weight:800;text-align:center;color:#173b63;margin-bottom:.2rem}
.sub-title {text-align:center;color:#5e6875;margin-bottom:1.5rem}
.result-v {padding:16px;border-radius:12px;background:#eef8f0;border-left:7px solid #2e7d32}
.result-w {padding:16px;border-radius:12px;background:#eef5ff;border-left:7px solid #1565c0}
.result-cat {padding:16px;border-radius:12px;background:#f8f5ff;border-left:7px solid #6a1b9a}
.rule-box {padding:14px;border-radius:10px;background:#fafafa;border:1px solid #e5e7eb}
.small-note {font-size:.88rem;color:#667085}
.reason {padding:7px 10px;margin:4px 0;background:#f7f8fa;border-radius:8px;border-left:3px solid #9aa4b2}
.exact {border-left-color:#2e7d32}
.sensitivity {border-left-color:#ef6c00}
.info-chip {display:inline-block;padding:3px 8px;margin:2px;border-radius:10px;background:#eef1f5;font-size:.78rem}
.model-head {font-size:1.05rem;font-weight:700}
[data-testid="stMetricValue"] {font-size:1.45rem}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📚 Gujarati Vishwakosh vs Wikipedia Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Rule-based style analysis + 17 trained dual-task ML models + model-specific explanations</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# 2. EXACT TRAINING CONSTANTS
# =============================================================================

CATEGORIES = [
    "Person Biography",
    "Computer Science & Information Technology",
    "Engineering & Technology",
    "Medical & Health Sciences",
    "Education & Research",
    "Language, Literature & Communication",
    "Science, Industry & Society",
]

CATEGORY_KEYWORDS = {
    "Person Biography": {
        "visvakosh": [
            'જન્મ', 'અવસાન', 'પદવી', 'શિક્ષણ', 'યુનિવર્સિટી', 'પ્રોફેસર',
            'વૈજ્ઞાનિક', 'ગણિતશાસ્ત્રી', 'ઇજનેર', 'સંશોધક', 'સંશોધન',
            'કારકિર્દી', 'પ્રદાન', 'પુરસ્કાર', 'ઍવૉર્ડ', 'મેડલ', 'એનાયત',
            'ફેલો', 'સભ્ય', 'સ્થાપક', 'ડિરેક્ટર', 'પીએચ.ડી.', 'હિન્ટન',
            'એકર્ટ', 'એડા', 'ટ્યૂરિંગ', 'બૅબેજ', 'ચિદંબરમ્',
        ],
        "wikipedia": [
            'જન્મ', 'મૃત્યુ', 'જીવન', 'જીવનચરિત્ર', 'કારકિર્દી', 'શિક્ષણ',
            'કાર્ય', 'યોગદાન', 'સંશોધન', 'વૈજ્ઞાનિક', 'ગણિતશાસ્ત્રી',
            'ઇજનેર', 'પ્રોગ્રામર', 'પ્રોફેસર', 'યુનિવર્સિટી', 'પુરસ્કાર',
            'સન્માન', 'પદવી', 'જાણીતા', 'સ્થાપક',
            'researcher', 'scientist', 'engineer', 'mathematician',
        ],
    },
    "Computer Science & Information Technology": {
        "visvakosh": [
            'કોમ્પ્યૂટર', 'કમ્પ્યુટર', 'સૉફ્ટવૅર', 'હાર્ડવેર', 'પ્રોગ્રામ',
            'પ્રોગ્રામિંગ', 'અલ્ગોરિધમ', 'માહિતી', 'ડેટા', 'નિવેશ',
            'ઇનપુટ', 'નિર્ગમ', 'આઉટપુટ', 'CPU', 'મેમરી', 'સ્ટોરેજ',
            'ડિસ્ક', 'ઇન્ટરનેટ', 'નેટવર્ક', 'વેબ', 'વેબસાઇટ', 'સર્વર',
            'ક્લાયન્ટ', 'DNS', 'ઇ-મેઇલ', 'કૃત્રિમ બુદ્ધિમત્તા', 'ચેટબોટ',
            'મશીન લર્નિંગ', 'ડેટાબેઇઝ', 'દ્વિઅંકી',
        ],
        "wikipedia": [
            'કમ્પ્યુટર', 'computer', 'software', 'hardware', 'program',
            'programming', 'algorithm', 'data', 'database', 'memory',
            'processor', 'CPU', 'internet', 'network', 'web', 'server',
            'browser', 'email', 'ઇ-મેઇલ', 'AI', 'કૃત્રિમ બુદ્ધિમત્તા',
            'machine learning', 'deep learning', 'ChatGPT', 'OpenAI',
            'chatbot', 'information technology', 'IT', 'code',
        ],
    },
    "Engineering & Technology": {
        "visvakosh": [
            'ઇજનેરી', 'યંત્ર', 'સાધન', 'ડિઝાઇન', 'ઇલેકટ્રોનિક્સ',
            'ઇલેક્ટ્રોનિક', 'વિદ્યુત', 'પરિપથ', 'ટ્રાન્ઝિસ્ટર',
            'અર્ધવાહક', 'IC', 'માઇક્રોવેવ', 'તરંગ', 'આવૃત્તિ', 'રેડિયો',
            'સિગ્નલ', 'યાન', 'યાન-નયન', 'કૉકપિટ', 'મોટર', 'વાહન',
            'ઑટોમોબાઇલ', 'ઉત્પાદન', 'બીબું', 'વૉશિંગ મશીન',
            'સંદેશાવ્યવહાર', 'પ્રસારણ', 'ઉપગ્રહ', 'નિયંત્રણ',
        ],
        "wikipedia": [
            'engineering', 'technology', 'ઇજનેરી', 'યંત્ર', 'machine',
            'device', 'system', 'design', 'electronic', 'electronics',
            'electrical', 'circuit', 'semiconductor', 'transistor',
            'microwave', 'frequency', 'signal', 'navigation', 'vehicle',
            'automobile', 'cockpit', 'motor', 'manufacturing', 'production',
            'communication', 'transmission', 'satellite', 'control system',
        ],
    },
    "Medical & Health Sciences": {
        "visvakosh": [
            'ઔષધ', 'ઔષધો', 'દવા', 'ફાર્મસી', 'રોગ', 'દર્દી', 'નિદાન',
            'સારવાર', 'શરીર', 'શ્રવણ', 'શ્રવણસહાયક', 'કાન', 'વિકિરણ',
            'કિરણોત્સર્ગી', 'વિકિરણશીલ', 'સમસ્થાનિક', 'રેડિયો સમસ્થાનિક',
            'ગૅમા', 'થાઇરૉઇડ', 'ચિત્રણ', 'સ્કૅન', 'PET', 'સ્મૃતિ',
            'સ્મૃતિલોપ', 'વિસ્મૃતિ', 'યાદ', 'મગજ', 'દીર્ઘકાલીન',
            'અલ્પકાલીન', 'ઈજા',
        ],
        "wikipedia": [
            'medical', 'medicine', 'health', 'ઔષધ', 'દવા', 'રોગ', 'disease',
            'patient', 'દર્દી', 'diagnosis', 'નિદાન', 'treatment', 'સારવાર',
            'pharmacy', 'hearing', 'hearing aid', 'કાન', 'radioisotope',
            'isotope', 'radiation', 'scan', 'imaging', 'PET', 'thyroid',
            'memory', 'સ્મૃતિ', 'amnesia', 'સ્મૃતિલોપ', 'brain', 'મગજ',
        ],
    },
    "Education & Research": {
        "visvakosh": [
            'શિક્ષણ', 'પ્રાથમિક', 'શાળા', 'વિદ્યાર્થી', 'શિક્ષક',
            'અધ્યાપક', 'અધ્યાપન', 'બોધન', 'અધ્યયન', 'અભ્યાસ',
            'અભ્યાસક્રમ', 'પરીક્ષા', 'તાલીમ', 'વિશ્વવિદ્યાલય',
            'ઉચ્ચ શિક્ષણ', 'અનુદાન', 'આયોગ', 'UGC', 'શિક્ષણનીતિ',
            'સંશોધન', 'પ્રયોગશાળા', 'PRL', 'સંસ્થા', 'વ્યવસ્થાપન',
            'IIM', 'ભૌતિકવિજ્ઞાન', 'યુનિવર્સિટી', 'શૈક્ષણિક',
        ],
        "wikipedia": [
            'education', 'શિક્ષણ', 'school', 'શાળા', 'student', 'વિદ્યાર્થી',
            'teacher', 'શિક્ષક', 'teaching', 'બોધન', 'learning', 'અધ્યયન',
            'curriculum', 'પાઠ્યક્રમ', 'university', 'વિશ્વવિદ્યાલય',
            'higher education', 'research', 'સંશોધન', 'laboratory',
            'પ્રયોગશાળા', 'UGC', 'grant', 'અનુદાન', 'commission', 'institute',
            'PRL', 'IIM', 'academic',
        ],
    },
    "Language, Literature & Communication": {
        "visvakosh": [
            'લેખન', 'લખાણ', 'શબ્દ', 'ભાષા', 'લિપિ', 'અક્ષર', 'શૈલી',
            'પ્રૂફ', 'પ્રૂફરીડિંગ', 'ભૂલો', 'સંપાદન', 'પ્રકાશન', 'કોશ',
            'પર્યાયકોશ', 'પર્યાય', 'નિઘંટુ', 'દસ્તાવેજ', 'ડૉક્યુમેન્ટેશન',
            'લેખ્યસૂચિ', 'લેખ્યસૂચીકરણ', 'ગ્રંથાલય', 'સુલેખન', 'લેખિની',
            'કલમ', 'શાહી', 'કાગળ', 'પ્રતિલિપિ', 'અક્ષરમાળા',
        ],
        "wikipedia": [
            'language', 'ભાષા', 'literature', 'સાહિત્ય', 'writing', 'લેખન',
            'text', 'લખાણ', 'word', 'શબ્દ', 'script', 'લિપિ', 'letter',
            'અક્ષર', 'proofreading', 'પ્રૂફરીડિંગ', 'editing', 'સંપાદન',
            'dictionary', 'શબ્દકોશ', 'thesaurus', 'પર્યાયકોશ', 'synonym',
            'પર્યાય', 'documentation', 'દસ્તાવેજ', 'calligraphy', 'સુલેખન',
            'publication', 'પ્રકાશન',
        ],
    },
    "Science, Industry & Society": {
        "visvakosh": [
            'વિજ્ઞાન', 'ઔદ્યોગિક', 'ઉદ્યોગ', 'વિકાસ', 'ઉત્પાદન',
            'સેવા-ઉદ્યોગ', 'આંકડાશાસ્ત્ર', 'આંકડા', 'આંકડાશાસ્ત્રીય',
            'નમૂના', 'પ્રમાણ', 'હીરા', 'હીરો', 'હીરાઉદ્યોગ', 'કૅરેટ',
            'ખાણ', 'કિમ્બરલાઇટ', 'કાર્બન', 'વૃદ્ધિ', 'જનસંખ્યા',
            'સંસાધન', 'પર્યાવરણ', 'પગરખાં', 'ચામડું', 'હવામાન',
            'વાતાવરણ', 'તાપમાન', 'દબાણ', 'ભેજ', 'વરસાદ', 'પવન',
            'ચક્રવાત', 'આગાહી',
        ],
        "wikipedia": [
            'science', 'વિજ્ઞાન', 'industry', 'ઉદ્યોગ', 'industrial',
            'ઔદ્યોગિક', 'development', 'વિકાસ', 'statistics', 'આંકડાશાસ્ત્ર',
            'statistical', 'service industry', 'diamond', 'હીરા', 'carat',
            'કૅરેટ', 'mine', 'ખાણ', 'growth', 'વૃદ્ધિ', 'population',
            'જનસંખ્યા', 'resources', 'environment', 'પર્યાવરણ', 'weather',
            'હવામાન', 'temperature', 'તાપમાન', 'rainfall', 'વરસાદ',
            'climate', 'forecast',
        ],
    },
}

CATEGORY_KEYWORD_REGEX: Dict[str, Dict[str, Any]] = {}
for _cat, _sets in CATEGORY_KEYWORDS.items():
    CATEGORY_KEYWORD_REGEX[_cat] = {}
    for _src in ("visvakosh", "wikipedia"):
        _phrases = sorted(set(k for k in _sets[_src] if k), key=len, reverse=True)
        CATEGORY_KEYWORD_REGEX[_cat][_src] = (
            re.compile("|".join(re.escape(k) for k in _phrases), flags=re.IGNORECASE)
            if _phrases else None
        )

STYLE_FEATURE_NAMES = [
    'word_count', 'log_word_count', 'char_count', 'log_char_count',
    'sentence_count', 'avg_word_length',
    'avg_sentence_length', 'std_sentence_length',
    'max_sentence_length', 'min_sentence_length',
    'type_token_ratio', 'hapax_ratio',
    'v_markers_per_1000', 'w_markers_per_1000',
    'marker_diff_per_1000', 'marker_ratio',
    'passive_per_1000',
    'english_char_ratio', 'gujarati_char_ratio', 'script_ratio',
    'colon_per_1000', 'comma_per_1000', 'paren_per_1000',
    'space_comma_per_1000', 'hyphen_per_1000', 'danda_per_1000',
    'citation_count', 'wiki_heading_count',
    'colon_in_first_200', 'def_in_first_200',
    'cnt_તથા', 'cnt_વળી', 'cnt_કહેવાય_છે', 'cnt_એટલે', 'cnt_કરાય_છે',
    'cnt_શામેલ', 'cnt_દ્વારા', 'cnt_સક્ષમ', 'cnt_ઉલ્લેખ',
    'cnt_કરવામાં_આવે_છે',
    'translit_style_ratio',
]
assert len(STYLE_FEATURE_NAMES) == 41

CATEGORY_KEYWORD_FEATURE_NAMES: List[str] = []
for _cat in CATEGORIES:
    CATEGORY_KEYWORD_FEATURE_NAMES.extend([
        f"catkw::{_cat}::visvakosh_per_1000",
        f"catkw::{_cat}::wikipedia_per_1000",
        f"catkw::{_cat}::total_per_1000",
    ])

VISVAKOSH_STYLE_MARKERS = [
    'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
    'અલબત્ત', 'તદુપરાંત', 'દા.ત.', 'જુઓ', 'એટલે કે', 'કહેવાય છે',
    'દા. ત.', 'તેમજ', 'ઉપરાંત', 'વિશેષ', 'અત્રે', 'તેવી જ રીતે',
    'એટલે', 'કહેવાય', 'કરાય છે', 'થાય છે', 'ઓળખાય છે', 'ગણાય છે',
]
WIKIPEDIA_STYLE_MARKERS = [
    'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
    'મોડેલ', 'સોફ્ટવેર', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે', 'દ્વારા',
    'સંદર્ભ', 'બાહ્ય કડીઓ', 'સ્રોત', 'ટીકા', 'વિવાદ', 'સક્ષમ',
    'સમાવેશ', 'ઉલ્લેખ', 'પ્રોગ્રામ', 'ક્લસ્ટર',
    'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
    'માનવામાં આવે છે',
]
VISVAKOSH_PASSIVE = ['ગણાય છે', 'કરાય છે', 'કહેવાય છે', 'થાય છે', 'ઓળખાય છે']
WIKIPEDIA_PASSIVE = [
    'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
    'માનવામાં આવે છે', 'કરવામાં આવ્યા હતા', 'કરવામાં આવ્યું હતું'
]
DEFINITION_MARKERS = ['એટલે', 'કહેવાય', 'ગણાય', 'રૂપે ઓળખાય', 'એટલે કે']
TRADITIONAL_TRANSLIT = ['ૉ', 'ૅ', 'ઑ', 'ઍ']
MODERN_TRANSLIT = ['ો', 'ે', 'ૈ']
WIKI_HEADINGS = [
    'મુખ્ય લેખ', 'આ પણ જુઓ', 'સંદર્ભ', 'બાહ્ય કડીઓ', 'બાહ્ય લિંક્સ',
    'વધુ વાંચન', 'નોંધ', 'ટીકા', 'જીવન', 'કારકિર્દી', 'ઇતિહાસ'
]

WORD_RE = re.compile(r'[\u0A80-\u0AFF]+|[A-Za-z]+|[0-9]+')
GUJ_RE = re.compile(r'[\u0A80-\u0AFF]')
ENG_RE = re.compile(r'[A-Za-z]')

EXPECTED_MODELS = [
    'AdaBoost', 'BernoulliNB', 'DecisionTree', 'ExtraTrees', 'KNN', 'LDA',
    'LinearSVC', 'LogisticRegression', 'LogisticRegression_L1', 'MLP',
    'PassiveAggressive', 'Perceptron', 'RandomForest', 'RidgeClassifier',
    'SGDClassifier', 'SVC_Linear', 'SVC_RBF'
]


# =============================================================================
# 3. FEATURE EXTRACTION — EXACTLY MATCHES TRAINING
# =============================================================================

def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text or "")


def tokenize_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace('।', '.')
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 2]


def _count_markers(text: str, markers: List[str]) -> int:
    return int(sum(text.count(m) for m in markers if m))


def extract_41_style_features(text: str) -> Dict[str, float]:
    text = "" if text is None else str(text)
    words = tokenize_words(text)
    sentences = tokenize_sentences(text)
    wc = max(len(words), 1)
    cc = max(len(text), 1)

    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    sentence_lengths = [x for x in sentence_lengths if x > 0]

    if words:
        avg_word_len = float(np.mean([len(w) for w in words]))
        uniq = set(words)
        ttr = len(uniq) / len(words)
        freq = Counter(words)
        hapax = sum(1 for _, c in freq.items() if c == 1)
        hapax_ratio = hapax / max(len(uniq), 1)
    else:
        avg_word_len = ttr = hapax_ratio = 0.0

    if sentence_lengths:
        avg_sl = float(np.mean(sentence_lengths))
        std_sl = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        max_sl = float(max(sentence_lengths))
        min_sl = float(min(sentence_lengths))
    else:
        avg_sl = std_sl = max_sl = min_sl = 0.0

    v_markers = _count_markers(text, VISVAKOSH_STYLE_MARKERS)
    w_markers = _count_markers(text, WIKIPEDIA_STYLE_MARKERS)
    total_markers = v_markers + w_markers
    passive = _count_markers(text, VISVAKOSH_PASSIVE + WIKIPEDIA_PASSIVE)
    eng_chars = len(ENG_RE.findall(text))
    guj_chars = len(GUJ_RE.findall(text))
    citations = len(re.findall(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]', text))
    wiki_heading_count = sum(1 for h in WIKI_HEADINGS if h in text)
    first_200 = text[:200]
    trad = sum(text.count(x) for x in TRADITIONAL_TRANSLIT)
    modern = sum(text.count(x) for x in MODERN_TRANSLIT)
    translit_ratio = trad / (trad + modern) if (trad + modern) > 0 else 0.5

    feats = {
        'word_count': float(len(words)),
        'log_word_count': float(np.log1p(len(words))),
        'char_count': float(len(text)),
        'log_char_count': float(np.log1p(len(text))),
        'sentence_count': float(len(sentences)),
        'avg_word_length': avg_word_len,
        'avg_sentence_length': avg_sl,
        'std_sentence_length': std_sl,
        'max_sentence_length': max_sl,
        'min_sentence_length': min_sl,
        'type_token_ratio': float(ttr),
        'hapax_ratio': float(hapax_ratio),
        'v_markers_per_1000': 1000.0 * v_markers / wc,
        'w_markers_per_1000': 1000.0 * w_markers / wc,
        'marker_diff_per_1000': 1000.0 * (v_markers - w_markers) / wc,
        'marker_ratio': v_markers / total_markers if total_markers > 0 else 0.5,
        'passive_per_1000': 1000.0 * passive / wc,
        'english_char_ratio': eng_chars / cc,
        'gujarati_char_ratio': guj_chars / cc,
        'script_ratio': guj_chars / max(guj_chars + eng_chars, 1),
        'colon_per_1000': 1000.0 * text.count(':') / wc,
        'comma_per_1000': 1000.0 * text.count(',') / wc,
        'paren_per_1000': 1000.0 * (text.count('(') + text.count(')')) / wc,
        'space_comma_per_1000': 1000.0 * text.count(' ,') / wc,
        'hyphen_per_1000': 1000.0 * text.count('-') / wc,
        'danda_per_1000': 1000.0 * text.count('।') / wc,
        'citation_count': float(citations),
        'wiki_heading_count': float(wiki_heading_count),
        'colon_in_first_200': float(':' in first_200),
        'def_in_first_200': float(any(m in first_200 for m in DEFINITION_MARKERS)),
        'cnt_તથા': float(text.count('તથા')),
        'cnt_વળી': float(text.count('વળી')),
        'cnt_કહેવાય_છે': float(text.count('કહેવાય છે')),
        'cnt_એટલે': float(text.count('એટલે')),
        'cnt_કરાય_છે': float(text.count('કરાય છે')),
        'cnt_શામેલ': float(text.count('શામેલ')),
        'cnt_દ્વારા': float(text.count('દ્વારા')),
        'cnt_સક્ષમ': float(text.count('સક્ષમ')),
        'cnt_ઉલ્લેખ': float(text.count('ઉલ્લેખ')),
        'cnt_કરવામાં_આવે_છે': float(text.count('કરવામાં આવે છે')),
        'translit_style_ratio': float(translit_ratio),
    }
    return {name: float(feats[name]) for name in STYLE_FEATURE_NAMES}


def extract_category_keyword_features(text: str) -> Dict[str, float]:
    text = "" if text is None else str(text)
    wc = max(len(tokenize_words(text)), 1)
    out: Dict[str, float] = {}
    for cat in CATEGORIES:
        vrx = CATEGORY_KEYWORD_REGEX[cat]["visvakosh"]
        wrx = CATEGORY_KEYWORD_REGEX[cat]["wikipedia"]
        v = sum(1 for _ in vrx.finditer(text)) if vrx is not None else 0
        w = sum(1 for _ in wrx.finditer(text)) if wrx is not None else 0
        out[f"catkw::{cat}::visvakosh_per_1000"] = 1000.0 * v / wc
        out[f"catkw::{cat}::wikipedia_per_1000"] = 1000.0 * w / wc
        out[f"catkw::{cat}::total_per_1000"] = 1000.0 * (v + w) / wc
    return out


def category_keyword_matches(text: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for cat in CATEGORIES:
        v_words = [(w, len(re.findall(re.escape(w), text, flags=re.IGNORECASE)))
                   for w in CATEGORY_KEYWORDS[cat]['visvakosh'] if re.search(re.escape(w), text, flags=re.IGNORECASE)]
        w_words = [(w, len(re.findall(re.escape(w), text, flags=re.IGNORECASE)))
                   for w in CATEGORY_KEYWORDS[cat]['wikipedia'] if re.search(re.escape(w), text, flags=re.IGNORECASE)]
        out[cat] = {
            'v_words': v_words,
            'w_words': w_words,
            'v_hits': sum(c for _, c in v_words),
            'w_hits': sum(c for _, c in w_words),
        }
    return out


def raw_feature_lookup(text: str) -> Dict[str, float]:
    style = extract_41_style_features(text)
    cat = extract_category_keyword_features(text)
    out = {f"style::{k}": v for k, v in style.items()}
    out.update(cat)
    return out


def transform_text(text: str, artifacts: Dict[str, Any]):
    style_names = artifacts.get('style_feature_names', STYLE_FEATURE_NAMES)
    cat_names = artifacts.get('category_keyword_feature_names', CATEGORY_KEYWORD_FEATURE_NAMES)

    sf = extract_41_style_features(text)
    cf = extract_category_keyword_features(text)
    s = np.asarray([[float(sf.get(k, 0.0)) for k in style_names]], dtype=np.float64)
    c = np.asarray([[float(cf.get(k, 0.0)) for k in cat_names]], dtype=np.float64)

    s_scaled = artifacts['style_scaler'].transform(s)
    c_scaled = artifacts['catkw_scaler'].transform(c)
    w = artifacts['word_tfidf'].transform([text])
    ch = artifacts['char_tfidf'].transform([text])
    return hstack([csr_matrix(s_scaled), csr_matrix(c_scaled), w, ch]).tocsr()


def select_model_input(bundle: Dict[str, Any], x_full):
    if bundle.get('feature_view') == 'handcrafted_62':
        n = int(bundle.get('style_feature_count', 41)) + len(bundle.get('category_keyword_feature_names', CATEGORY_KEYWORD_FEATURE_NAMES))
        return x_full[:, :n]
    return x_full


# =============================================================================
# 4. RULE-BASED ANALYSIS — INDEPENDENT FROM ML
# =============================================================================

def rule_based_source(text: str) -> Dict[str, Any]:
    f = extract_41_style_features(text)
    v = 0.0
    w = 0.0
    rows = []

    def add(side: str, points: float, rule: str, observed: str):
        nonlocal v, w
        if side == 'Visvakosh':
            v += points
        else:
            w += points
        rows.append({'Source': side, 'Points': points, 'Rule': rule, 'Observed': observed})

    if f['colon_in_first_200']:
        add('Visvakosh', 2, 'Definition-first / term : explanation opening', 'colon in first 200 characters')
    if f['def_in_first_200']:
        add('Visvakosh', 2, 'Definition marker near the beginning', 'એટલે / કહેવાય / ગણાય present early')

    md = f['marker_diff_per_1000']
    if md >= 5:
        add('Visvakosh', 2, 'Visvakosh-style markers dominate', f"difference = {md:.2f}/1000")
    elif md <= -5:
        add('Wikipedia', 2, 'Wikipedia-style markers dominate', f"difference = {md:.2f}/1000")

    if f['citation_count'] >= 1:
        add('Wikipedia', 2, 'Bracketed numeric citations', f"{int(f['citation_count'])} citation(s)")
    if f['wiki_heading_count'] >= 1:
        add('Wikipedia', 2, 'Wikipedia-like section/navigation terms', f"{int(f['wiki_heading_count'])} heading marker(s)")

    if f['cnt_કરવામાં_આવે_છે'] >= 1:
        add('Wikipedia', 1.5, 'Long passive construction', f"કરવામાં આવે છે × {int(f['cnt_કરવામાં_આવે_છે'])}")
    if f['cnt_કરાય_છે'] + f['cnt_કહેવાય_છે'] >= 1:
        add('Visvakosh', 1.5, 'Concise passive/definitional construction',
            f"કરાય છે + કહેવાય છે = {int(f['cnt_કરાય_છે'] + f['cnt_કહેવાય_છે'])}")

    if f['english_char_ratio'] >= 0.06:
        add('Wikipedia', 1, 'High English-script share', f"{f['english_char_ratio']:.1%}")
    elif f['english_char_ratio'] <= 0.015:
        add('Visvakosh', 0.5, 'Very low English-script share', f"{f['english_char_ratio']:.1%}")

    if f['translit_style_ratio'] >= 0.30:
        add('Visvakosh', 1, 'Traditional Gujarati transliteration marks relatively strong', f"ratio={f['translit_style_ratio']:.3f}")

    if f['space_comma_per_1000'] >= 3:
        add('Wikipedia', 1, 'Space-before-comma pattern', f"{f['space_comma_per_1000']:.2f}/1000")

    total = v + w
    if total == 0:
        pred = 'Uncertain'
        conf = 0.5
    else:
        pred = 'Visvakosh' if v >= w else 'Wikipedia'
        conf = max(v, w) / total

    return {
        'prediction': pred,
        'confidence': float(conf),
        'visvakosh_score': float(v),
        'wikipedia_score': float(w),
        'rules': rows,
        'features': f,
    }


def rule_based_category(text: str) -> Dict[str, Any]:
    feats = extract_category_keyword_features(text)
    matches = category_keyword_matches(text)
    ranked = []
    for cat in CATEGORIES:
        v = feats[f"catkw::{cat}::visvakosh_per_1000"]
        w = feats[f"catkw::{cat}::wikipedia_per_1000"]
        total = feats[f"catkw::{cat}::total_per_1000"]
        ranked.append((cat, total, v, w))
    ranked.sort(key=lambda x: x[1], reverse=True)
    best = ranked[0]
    total_all = sum(x[1] for x in ranked)
    confidence = best[1] / total_all if total_all > 0 else 0.0
    return {
        'category': best[0] if best[1] > 0 else 'General / Unknown',
        'confidence': float(confidence),
        'ranked': ranked,
        'matches': matches,
    }


# =============================================================================
# 5. MODEL DISCOVERY / LOADING
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'trained_models')


def discover_model_files() -> Dict[str, str]:
    paths = glob.glob(os.path.join(MODEL_DIR, '*.pkl'))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(paths)}


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> Dict[str, Any]:
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError('PKL does not contain a dictionary bundle.')
    required = ['source_model', 'category_model', 'feature_artifacts', 'model_name']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing bundle fields: {missing}")
    return data


MODEL_FILES = discover_model_files()


# =============================================================================
# 6. PREDICTION SUPPORT HELPERS
# =============================================================================

def _classes(model) -> np.ndarray:
    if hasattr(model, 'classes_'):
        return np.asarray(model.classes_)
    if hasattr(model, 'steps') and model.steps:
        last = model.steps[-1][1]
        if hasattr(last, 'classes_'):
            return np.asarray(last.classes_)
    return np.asarray([])


def _class_index(classes: np.ndarray, target: Any) -> int:
    for i, c in enumerate(classes.tolist()):
        try:
            if int(c) == int(target):
                return i
        except Exception:
            if c == target:
                return i
    return 0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(np.clip(x, -50, 50))
    return e / max(float(np.sum(e)), 1e-12)


def support_for_target(model, X, target_class: Any) -> Tuple[float, str]:
    classes = _classes(model)
    if hasattr(model, 'predict_proba'):
        try:
            p = np.asarray(model.predict_proba(X))[0]
            idx = _class_index(classes, target_class)
            return float(p[idx]), 'probability'
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        try:
            d = np.asarray(model.decision_function(X))
            if d.ndim == 0:
                d = np.asarray([float(d)])
            if d.ndim > 1:
                d = d[0]
            if len(classes) == 2 and d.size == 1:
                val = float(d.ravel()[0])
                return (val if int(target_class) == int(classes[1]) else -val), 'margin'
            idx = _class_index(classes, target_class)
            return float(d.ravel()[idx]), 'decision score'
        except Exception:
            pass

    pred = model.predict(X)[0]
    return (1.0 if pred == target_class else 0.0), 'prediction match'


def prediction_info(model, X, class_map: Dict[Any, str]) -> Dict[str, Any]:
    pred = model.predict(X)[0]
    try:
        pred_key = int(pred)
    except Exception:
        pred_key = pred

    normalized_map = {}
    for k, v in class_map.items():
        try:
            normalized_map[int(k)] = v
        except Exception:
            normalized_map[k] = v
    label = normalized_map.get(pred_key, str(pred_key))

    classes = _classes(model)
    mode = 'prediction'
    confidence = None
    support = None
    runner = None

    if hasattr(model, 'predict_proba'):
        try:
            p = np.asarray(model.predict_proba(X))[0]
            idx = _class_index(classes, pred_key)
            confidence = float(p[idx])
            support = confidence
            mode = 'model probability'
            order = np.argsort(p)[::-1]
            for j in order:
                ck = classes[j]
                try:
                    ck = int(ck)
                except Exception:
                    pass
                if ck != pred_key:
                    runner = ck
                    break
        except Exception:
            pass

    if confidence is None and hasattr(model, 'decision_function'):
        try:
            d = np.asarray(model.decision_function(X))
            if d.ndim > 1:
                d = d[0]
            if len(classes) == 2 and d.size == 1:
                margin = float(d.ravel()[0])
                signed = margin if pred_key == int(classes[1]) else -margin
                support = signed
                confidence = 1.0 / (1.0 + math.exp(-min(max(signed, -30), 30)))
                runner = int(classes[0]) if pred_key == int(classes[1]) else int(classes[1])
            else:
                scores = d.ravel().astype(float)
                idx = _class_index(classes, pred_key)
                support = float(scores[idx])
                pseudo = _softmax(scores)
                confidence = float(pseudo[idx])
                order = np.argsort(scores)[::-1]
                for j in order:
                    ck = classes[j]
                    try:
                        ck = int(ck)
                    except Exception:
                        pass
                    if ck != pred_key:
                        runner = ck
                        break
            mode = 'uncalibrated decision support'
        except Exception:
            pass

    if confidence is None:
        confidence = 0.5
        support, _ = support_for_target(model, X, pred_key)
        mode = 'prediction only'

    return {
        'raw_class': pred_key,
        'label': label,
        'confidence': float(confidence),
        'support': float(support) if support is not None else None,
        'support_mode': mode,
        'runner_class': runner,
    }


def _row_array(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().ravel()
    return np.asarray(X).ravel()


def _zero_feature(X, idx: int):
    if issparse(X):
        y = X.tolil(copy=True)
        y[0, idx] = 0.0
        return y.tocsr()
    y = np.asarray(X).copy()
    y[0, idx] = 0.0
    return y


def _zero_range(X, start: int, end: int):
    if start >= end:
        return X
    if issparse(X):
        y = X.tolil(copy=True)
        y[0, start:end] = 0.0
        return y.tocsr()
    y = np.asarray(X).copy()
    y[0, start:end] = 0.0
    return y


# =============================================================================
# 7. HUMAN-READABLE FEATURE NAMES
# =============================================================================

STYLE_LABELS = {
    'word_count': 'word count / text length',
    'log_word_count': 'log word count',
    'char_count': 'character count / text length',
    'log_char_count': 'log character count',
    'sentence_count': 'number of sentences',
    'avg_word_length': 'average word length',
    'avg_sentence_length': 'average sentence length',
    'std_sentence_length': 'variation in sentence length',
    'max_sentence_length': 'maximum sentence length',
    'min_sentence_length': 'minimum sentence length',
    'type_token_ratio': 'lexical diversity (type-token ratio)',
    'hapax_ratio': 'one-time-word / hapax ratio',
    'v_markers_per_1000': 'Visvakosh-style marker frequency',
    'w_markers_per_1000': 'Wikipedia-style marker frequency',
    'marker_diff_per_1000': 'Visvakosh minus Wikipedia marker frequency',
    'marker_ratio': 'Visvakosh share of source-style markers',
    'passive_per_1000': 'passive construction frequency',
    'english_char_ratio': 'English-script character ratio',
    'gujarati_char_ratio': 'Gujarati-script character ratio',
    'script_ratio': 'Gujarati vs English script balance',
    'colon_per_1000': 'colon usage',
    'comma_per_1000': 'comma usage',
    'paren_per_1000': 'parentheses usage',
    'space_comma_per_1000': 'space-before-comma formatting',
    'hyphen_per_1000': 'hyphen usage',
    'danda_per_1000': 'danda punctuation usage',
    'citation_count': 'numeric citation count',
    'wiki_heading_count': 'Wikipedia-like heading/navigation count',
    'colon_in_first_200': 'colon near article opening',
    'def_in_first_200': 'definition marker near article opening',
    'cnt_તથા': "count of 'તથા'",
    'cnt_વળી': "count of 'વળી'",
    'cnt_કહેવાય_છે': "count of 'કહેવાય છે'",
    'cnt_એટલે': "count of 'એટલે'",
    'cnt_કરાય_છે': "count of 'કરાય છે'",
    'cnt_શામેલ': "count of 'શામેલ'",
    'cnt_દ્વારા': "count of 'દ્વારા'",
    'cnt_સક્ષમ': "count of 'સક્ષમ'",
    'cnt_ઉલ્લેખ': "count of 'ઉલ્લેખ'",
    'cnt_કરવામાં_આવે_છે': "count of 'કરવામાં આવે છે'",
    'translit_style_ratio': 'traditional transliteration style ratio',
}


def human_feature(name: str) -> str:
    if name.startswith('style::'):
        key = name.split('style::', 1)[1]
        return STYLE_LABELS.get(key, key)
    if name.startswith('catkw::'):
        parts = name.split('::')
        if len(parts) >= 3:
            side = parts[-1].replace('_per_1000', '').replace('_', ' ')
            return f"category terminology: {parts[1]} ({side})"
    if name.startswith('word::'):
        return f"word TF-IDF n-gram '{name[6:]}'"
    if name.startswith('char::'):
        token = name[6:].replace('\n', '↵')
        return f"character TF-IDF n-gram '{token}'"
    return name


def raw_value_text(feature_name: str, raw_lookup: Dict[str, float]) -> str:
    if feature_name in raw_lookup:
        v = raw_lookup[feature_name]
        if abs(v) >= 100:
            return f"raw={v:.1f}"
        if abs(v) >= 1:
            return f"raw={v:.3f}"
        return f"raw={v:.4f}"
    return ''


# =============================================================================
# 8. MODEL-SPECIFIC EXPLANATIONS
# =============================================================================

def _get_coef_model(model):
    if hasattr(model, 'coef_'):
        return model
    return None


def explain_linear(model, X, feature_names: List[str], pred_class: Any,
                   runner_class: Any, raw_lookup: Dict[str, float], top_n: int = 6) -> List[str]:
    est = _get_coef_model(model)
    if est is None:
        return []
    coef = np.asarray(est.coef_)
    classes = _classes(est)
    x = _row_array(X)
    if coef.ndim != 2 or coef.shape[1] != len(x):
        return []

    if len(classes) == 2 and coef.shape[0] == 1:
        sign = 1.0 if int(pred_class) == int(classes[1]) else -1.0
        w = coef[0] * sign
    elif coef.shape[0] == len(classes):
        pi = _class_index(classes, pred_class)
        if runner_class is None:
            scores = np.asarray(est.decision_function(X)).ravel()
            order = np.argsort(scores)[::-1]
            ri = next((j for j in order if j != pi), order[-1])
        else:
            ri = _class_index(classes, runner_class)
        w = coef[pi] - coef[ri]
    else:
        # e.g. multiclass SVC(kernel='linear') uses one-vs-one coefficient rows.
        return []

    contrib = x * w
    idxs = np.argsort(contrib)[::-1]
    lines = []
    for i in idxs:
        if len(lines) >= top_n:
            break
        if contrib[i] <= 1e-10 or i >= len(feature_names):
            continue
        fname = feature_names[i]
        rv = raw_value_text(fname, raw_lookup)
        lines.append(
            f"{human_feature(fname)} supported this class "
            f"(local contribution {contrib[i]:+.4f}{'; ' + rv if rv else ''})."
        )
    return lines


def explain_bernoulli_nb(model, X, feature_names: List[str], pred_class: Any,
                         runner_class: Any, raw_lookup: Dict[str, float], top_n: int = 6) -> List[str]:
    if not hasattr(model, 'feature_log_prob_'):
        return []
    classes = _classes(model)
    pi = _class_index(classes, pred_class)
    if runner_class is None:
        probs = np.asarray(model.predict_proba(X))[0]
        order = np.argsort(probs)[::-1]
        ri = next((j for j in order if j != pi), order[-1])
    else:
        ri = _class_index(classes, runner_class)

    flp = np.asarray(model.feature_log_prob_)
    if flp.shape[1] != X.shape[1]:
        return []
    p = np.clip(np.exp(flp), 1e-12, 1 - 1e-12)
    neg = np.log1p(-p)
    active = (_row_array(X) > 0).astype(float)
    contrib = active * (flp[pi] - flp[ri]) + (1 - active) * (neg[pi] - neg[ri])
    idxs = np.argsort(contrib)[::-1]
    lines = []
    for i in idxs:
        if len(lines) >= top_n:
            break
        if contrib[i] <= 1e-10 or i >= len(feature_names):
            continue
        fname = feature_names[i]
        state = 'present/above training mean' if active[i] else 'absent/below training mean'
        rv = raw_value_text(fname, raw_lookup)
        lines.append(
            f"{human_feature(fname)} ({state}) favored this class "
            f"by Bernoulli log-evidence {contrib[i]:+.4f}{'; ' + rv if rv else ''}."
        )
    return lines


def explain_decision_tree(model, X, feature_names: List[str], raw_lookup: Dict[str, float], top_n: int = 7) -> List[str]:
    if not hasattr(model, 'tree_'):
        return []
    path = model.decision_path(X).indices
    x = _row_array(X)
    conditions = []
    for node in path:
        feat = int(model.tree_.feature[node])
        if feat < 0 or feat >= len(feature_names):
            continue
        thr = float(model.tree_.threshold[node])
        val = float(x[feat])
        direction = '<=' if val <= thr else '>'
        fname = feature_names[feat]
        rv = raw_value_text(fname, raw_lookup)
        conditions.append(
            f"Tree tested {human_feature(fname)}: standardized value {val:.3f} {direction} {thr:.3f}"
            f"{'; ' + rv if rv else ''}."
        )
    # Conditions near the leaf are often the most specific.
    return conditions[-top_n:]


def local_feature_ablation(model, X, feature_names: List[str], pred_class: Any,
                           raw_lookup: Dict[str, float], candidate_indices: List[int],
                           top_n: int = 5) -> List[str]:
    base, mode = support_for_target(model, X, pred_class)
    effects = []
    for i in candidate_indices:
        if i < 0 or i >= X.shape[1] or i >= len(feature_names):
            continue
        x2 = _zero_feature(X, i)
        after, _ = support_for_target(model, x2, pred_class)
        drop = base - after
        effects.append((drop, i))
    effects.sort(reverse=True, key=lambda z: z[0])

    lines = []
    for drop, i in effects:
        if len(lines) >= top_n:
            break
        if drop <= 1e-8:
            continue
        fname = feature_names[i]
        rv = raw_value_text(fname, raw_lookup)
        unit = 'probability' if mode == 'probability' else 'model support'
        lines.append(
            f"When {human_feature(fname)} was neutralized, predicted-class {unit} fell by {drop:.4f}"
            f"{'; ' + rv if rv else ''}."
        )
    return lines


def explain_tree_ensemble(model, X, feature_names: List[str], pred_class: Any,
                          raw_lookup: Dict[str, float], top_n: int = 6) -> List[str]:
    if not hasattr(model, 'feature_importances_'):
        return []
    imp = np.asarray(model.feature_importances_)
    candidates = np.argsort(imp)[::-1][:min(18, len(imp))].tolist()
    lines = local_feature_ablation(model, X, feature_names, pred_class, raw_lookup, candidates, top_n)
    if lines:
        return lines
    # Fallback: do not claim direction; explicitly label as global learned importance.
    out = []
    for i in candidates[:top_n]:
        rv = raw_value_text(feature_names[i], raw_lookup)
        out.append(
            f"{human_feature(feature_names[i])} has high learned tree importance ({imp[i]:.4f})"
            f"{'; ' + rv if rv else ''}; importance alone does not encode direction."
        )
    return out


def feature_block_ranges(bundle: Dict[str, Any], X) -> List[Tuple[str, int, int]]:
    dims = bundle.get('feature_dimensions') or bundle.get('feature_artifacts', {}).get('dimensions', {})
    s = int(dims.get('style', 41))
    c = int(dims.get('category_keyword', 21))
    w = int(dims.get('word_tfidf', 0))
    ch = int(dims.get('char_tfidf', 0))
    blocks = [('41 style features', 0, min(s, X.shape[1]))]
    if X.shape[1] > s:
        blocks.append(('21 category-keyword features', s, min(s + c, X.shape[1])))
    if X.shape[1] > s + c:
        blocks.append(('word TF-IDF n-grams', s + c, min(s + c + w, X.shape[1])))
    if X.shape[1] > s + c + w:
        blocks.append(('character TF-IDF n-grams', s + c + w, min(s + c + w + ch, X.shape[1])))
    return [(n, a, b) for n, a, b in blocks if a < b]


def block_ablation(model, X, bundle: Dict[str, Any], pred_class: Any) -> List[str]:
    base, mode = support_for_target(model, X, pred_class)
    effects = []
    for name, a, b in feature_block_ranges(bundle, X):
        x2 = _zero_range(X, a, b)
        after, _ = support_for_target(model, x2, pred_class)
        effects.append((base - after, name))
    effects.sort(reverse=True, key=lambda z: z[0])
    lines = []
    for drop, name in effects:
        if drop > 1e-8:
            unit = 'probability' if mode == 'probability' else 'model support'
            lines.append(f"Removing the {name} reduced predicted-class {unit} by {drop:.4f}.")
    return lines[:4]


def handcrafted_sensitivity(model, X, feature_names: List[str], pred_class: Any,
                            raw_lookup: Dict[str, float], top_n: int = 5) -> List[str]:
    n = min(62, X.shape[1], len(feature_names))
    arr = np.abs(_row_array(X)[:n])
    # Test the strongest standardized handcrafted signals only, for speed.
    candidates = np.argsort(arr)[::-1][:min(16, n)].tolist()
    return local_feature_ablation(model, X, feature_names, pred_class, raw_lookup, candidates, top_n)


def explain_knn(model, X, class_map: Dict[Any, str], pred_class: Any) -> List[str]:
    if not hasattr(model, 'named_steps') or 'svd' not in model.named_steps or 'knn' not in model.named_steps:
        return []
    try:
        z = model.named_steps['svd'].transform(X)
        knn = model.named_steps['knn']
        k = min(getattr(knn, 'n_neighbors', 7), len(knn._fit_X))
        distances, indices = knn.kneighbors(z, n_neighbors=k)
        labels = np.asarray(knn._y)[indices[0]]
        same = int(np.sum(labels == pred_class))
        d = distances[0]
        nearest = ', '.join(f"{x:.3f}" for x in d[:min(5, len(d))])
        return [
            f"Nearest-neighbour evidence: {same}/{k} retrieved neighbours carry the predicted class; "
            f"nearest cosine distances = {nearest}."
        ]
    except Exception:
        return []


def explain_task(model_name: str, model, X, bundle: Dict[str, Any], pred_info: Dict[str, Any],
                 raw_lookup: Dict[str, float]) -> Dict[str, Any]:
    feature_names = list(bundle.get('model_feature_names', []))
    if not feature_names:
        all_names = bundle.get('feature_artifacts', {}).get('feature_names', [])
        feature_names = list(all_names[:X.shape[1]])

    pred_class = pred_info['raw_class']
    runner = pred_info.get('runner_class')
    exact_lines: List[str] = []
    sensitivity_lines: List[str] = []

    # 1) Exact / estimator-native evidence where mathematically available.
    if model_name == 'BernoulliNB':
        exact_lines = explain_bernoulli_nb(model, X, feature_names, pred_class, runner, raw_lookup)
    elif model_name == 'DecisionTree':
        exact_lines = explain_decision_tree(model, X, feature_names, raw_lookup)
    elif model_name in {'AdaBoost', 'ExtraTrees', 'RandomForest'}:
        exact_lines = explain_tree_ensemble(model, X, feature_names, pred_class, raw_lookup)
    elif model_name == 'KNN':
        exact_lines = explain_knn(model, X, {}, pred_class)
    else:
        exact_lines = explain_linear(model, X, feature_names, pred_class, runner, raw_lookup)

    # 2) Prediction-change evidence, especially important for nonlinear/projected models.
    sensitivity_lines.extend(block_ablation(model, X, bundle, pred_class))
    if model_name in {'KNN', 'LDA', 'MLP', 'SVC_RBF', 'SVC_Linear'} or not exact_lines:
        sensitivity_lines.extend(
            handcrafted_sensitivity(model, X, feature_names, pred_class, raw_lookup, top_n=5)
        )

    # De-duplicate while preserving order.
    def unique(xs):
        out = []
        seen = set()
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return {
        'exact': unique(exact_lines)[:7],
        'sensitivity': unique(sensitivity_lines)[:7],
        'method': bundle.get('explanation_method', 'model-specific learned evidence'),
    }


# =============================================================================
# 9. ONE MODEL PREDICTION
# =============================================================================

def evaluate_bundle(text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = bundle['feature_artifacts']
    x_full = transform_text(text, artifacts)
    x_model = select_model_input(bundle, x_full)
    source_model = bundle['source_model']
    category_model = bundle['category_model']

    source_classes = bundle.get('source_classes', {0: 'Visvakosh', 1: 'Wikipedia'})
    category_classes = bundle.get('category_classes', {i: c for i, c in enumerate(CATEGORIES)})

    source = prediction_info(source_model, x_model, source_classes)
    category = prediction_info(category_model, x_model, category_classes)
    raw = raw_feature_lookup(text)

    source_explanation = explain_task(
        bundle['model_name'], source_model, x_model, bundle, source, raw
    )
    category_explanation = explain_task(
        bundle['model_name'], category_model, x_model, bundle, category, raw
    )

    ev = bundle.get('evaluation', {}) or {}
    return {
        'model': bundle['model_name'],
        'algorithm_class': bundle.get('algorithm_class', ''),
        'feature_view': bundle.get('feature_view', 'full'),
        'source': source,
        'category': category,
        'source_explanation': source_explanation,
        'category_explanation': category_explanation,
        'source_macro_f1': (ev.get('source_metrics') or {}).get('macro_f1'),
        'category_macro_f1': (ev.get('category_metrics') or {}).get('macro_f1'),
        'error': None,
    }


# =============================================================================
# 10. ENSEMBLE
# =============================================================================

def majority_vote(labels: List[str]) -> Tuple[str, int, Dict[str, int]]:
    c = Counter(labels)
    if not c:
        return 'N/A', 0, {}
    label, n = c.most_common(1)[0]
    return label, n, dict(c)


# =============================================================================
# 11. SIDEBAR
# =============================================================================

with st.sidebar:
    st.header('⚙️ Model package')
    st.write(f"Model folder: `trained_models/`")
    st.metric('PKL files found', len(MODEL_FILES))

    missing = [m for m in EXPECTED_MODELS if m not in MODEL_FILES]
    extra = [m for m in MODEL_FILES if m not in EXPECTED_MODELS]
    if not missing and len(MODEL_FILES) == 17:
        st.success('All 17 expected models found.')
    else:
        if missing:
            st.error('Missing: ' + ', '.join(missing))
        if extra:
            st.info('Additional PKLs: ' + ', '.join(extra))

    with st.expander('Expected 17 models'):
        for i, m in enumerate(EXPECTED_MODELS, 1):
            icon = '✅' if m in MODEL_FILES else '❌'
            st.write(f"{i:02d}. {icon} {m}")

    st.markdown('---')
    st.caption(
        'Source labels are read from each trained bundle. No label-swap switch is used. '
        'The ML prediction is never changed by the explanation layer.'
    )


# =============================================================================
# 12. INPUT
# =============================================================================

sample_v = (
    "કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને "
    "માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને ચોકસાઈપૂર્વક "
    "રૂપાંતર કરી શકતું મશીન છે તથા અનેક ક્ષેત્રોમાં ઉપયોગી ગણાય છે."
)
sample_w = (
    "કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતી સંગ્રહિત અને પ્રક્રિયા કરવા માટે "
    "ઉપયોગમાં લેવામાં આવે છે. મુખ્ય લેખ: કમ્પ્યુટરનો ઇતિહાસ [1][2]. આ ઉપકરણનો "
    "ઉપયોગ વિવિધ ક્ષેત્રોમાં કરવામાં આવે છે."
)

c1, c2, c3 = st.columns([1, 1, 4])
with c1:
    if st.button('📖 Sample V', use_container_width=True):
        st.session_state['input_text'] = sample_v
with c2:
    if st.button('🌐 Sample W', use_container_width=True):
        st.session_state['input_text'] = sample_w
with c3:
    st.caption('Paste any Gujarati paragraph. Longer passages generally provide more stable style evidence.')

text = st.text_area(
    'Gujarati text',
    value=st.session_state.get('input_text', ''),
    height=260,
    placeholder='અહીં ગુજરાતી લખાણ પેસ્ટ કરો...',
)

if text:
    m1, m2, m3 = st.columns(3)
    m1.metric('Characters', f"{len(text):,}")
    m2.metric('Word tokens', f"{len(tokenize_words(text)):,}")
    m3.metric('Sentences', f"{len(tokenize_sentences(text)):,}")

analyze = st.button('🔍 ANALYZE WITH RULES + 17 ML MODELS', type='primary', use_container_width=True,
                    disabled=(len(text.strip()) < 20))


# =============================================================================
# 13. RESULTS
# =============================================================================

if analyze:
    rule_src = rule_based_source(text)
    rule_cat = rule_based_category(text)

    st.markdown('---')
    st.header('1. Rule-based style analysis')
    r1, r2, r3, r4 = st.columns(4)
    r1.metric('Rule source', rule_src['prediction'])
    r2.metric('Rule confidence', f"{rule_src['confidence']:.1%}")
    r3.metric('V score', f"{rule_src['visvakosh_score']:.1f}")
    r4.metric('W score', f"{rule_src['wikipedia_score']:.1f}")

    st.markdown(
        f"<div class='result-cat'><strong>Rule-based category:</strong> {rule_cat['category']} "
        f"<span class='small-note'>(keyword-share confidence {rule_cat['confidence']:.1%})</span></div>",
        unsafe_allow_html=True,
    )

    with st.expander('Show rule-by-rule source calculation', expanded=False):
        if rule_src['rules']:
            st.dataframe(pd.DataFrame(rule_src['rules']), use_container_width=True, hide_index=True)
        else:
            st.info('No strong source-style rule fired.')

    with st.expander('Show category keyword evidence', expanded=False):
        cat_rows = []
        for cat, total, v, w in rule_cat['ranked']:
            mt = rule_cat['matches'][cat]
            cat_rows.append({
                'Category': cat,
                'Total /1000': total,
                'V-keywords /1000': v,
                'W-keywords /1000': w,
                'V matched': ', '.join(f"{x}×{n}" for x, n in mt['v_words'][:8]),
                'W matched': ', '.join(f"{x}×{n}" for x, n in mt['w_words'][:8]),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # ---------------- ML models ----------------
    st.markdown('---')
    st.header('2. ML models — source + category')

    results = []
    load_errors = []
    progress = st.progress(0, text='Loading and running trained models...')
    model_order = [m for m in EXPECTED_MODELS if m in MODEL_FILES] + [m for m in MODEL_FILES if m not in EXPECTED_MODELS]

    for i, name in enumerate(model_order, 1):
        try:
            bundle = load_bundle(MODEL_FILES[name])
            if bundle.get('bundle_schema_version') and not str(bundle['bundle_schema_version']).startswith('2.'):
                st.warning(f"{name}: unexpected bundle schema {bundle['bundle_schema_version']}")
            results.append(evaluate_bundle(text, bundle))
        except Exception as e:
            load_errors.append((name, f"{type(e).__name__}: {e}"))
        progress.progress(i / max(len(model_order), 1), text=f"Processed {i}/{len(model_order)} models")
    progress.empty()

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} model load/prediction error(s)", expanded=True):
            for n, e in load_errors:
                st.error(f"{n}: {e}")
            st.caption(
                'If an error mentions an sklearn pickle/version mismatch, pin scikit-learn in requirements.txt '
                'to the exact sklearn.__version__ that was used in your Google Colab training run.'
            )

    if results:
        source_ens, source_n, source_counts = majority_vote([r['source']['label'] for r in results])
        cat_ens, cat_n, cat_counts = majority_vote([r['category']['label'] for r in results])

        e1, e2, e3, e4 = st.columns(4)
        e1.metric('ML source ensemble', source_ens)
        e2.metric('Source votes', f"{source_n}/{len(results)}")
        e3.metric('ML category ensemble', cat_ens)
        e4.metric('Category votes', f"{cat_n}/{len(results)}")

        st.caption(
            'Ensemble values are simple majority votes. They do not use the rule-based answer and do not alter any individual model prediction.'
        )

        summary_rows = []
        for r in results:
            summary_rows.append({
                'Model': r['model'],
                'Source': r['source']['label'],
                'Source support': f"{r['source']['confidence']:.1%}",
                'Support type': r['source']['support_mode'],
                'Category': r['category']['label'],
                'Category support': f"{r['category']['confidence']:.1%}",
                'Source holdout F1': '' if r['source_macro_f1'] is None else f"{r['source_macro_f1']:.3f}",
                'Category holdout F1': '' if r['category_macro_f1'] is None else f"{r['category_macro_f1']:.3f}",
                'Feature view': r['feature_view'],
            })

        st.subheader('Model summary')
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.subheader('Per-model prediction and real “Why?”')
        st.caption(
            'Green/exact evidence comes from coefficients, Bernoulli likelihoods, tree paths/importances, or nearest neighbours. '
            'Orange/sensitivity evidence is computed by neutralizing features/blocks and measuring the trained model’s own support change.'
        )

        for idx, r in enumerate(results, 1):
            s = r['source']
            c = r['category']
            with st.expander(
                f"#{idx:02d} {r['model']} → Source: {s['label']} | Category: {c['label']}",
                expanded=False,
            ):
                a, b = st.columns(2)
                with a:
                    st.markdown(f"### Source: **{s['label']}**")
                    st.write(f"**Support:** {s['confidence']:.1%} — {s['support_mode']}")
                    st.write(f"**How this algorithm was trained/explained:** {r['source_explanation']['method']}")
                    if r['source_explanation']['exact']:
                        st.markdown('**Direct learned evidence**')
                        for line in r['source_explanation']['exact']:
                            st.markdown(f"<div class='reason exact'>✓ {line}</div>", unsafe_allow_html=True)
                    if r['source_explanation']['sensitivity']:
                        st.markdown('**Local prediction-sensitivity evidence**')
                        for line in r['source_explanation']['sensitivity']:
                            st.markdown(f"<div class='reason sensitivity'>↳ {line}</div>", unsafe_allow_html=True)
                with b:
                    st.markdown(f"### Category: **{c['label']}**")
                    st.write(f"**Support:** {c['confidence']:.1%} — {c['support_mode']}")
                    st.write(f"**How this algorithm was trained/explained:** {r['category_explanation']['method']}")
                    if r['category_explanation']['exact']:
                        st.markdown('**Direct learned evidence**')
                        for line in r['category_explanation']['exact']:
                            st.markdown(f"<div class='reason exact'>✓ {line}</div>", unsafe_allow_html=True)
                    if r['category_explanation']['sensitivity']:
                        st.markdown('**Local prediction-sensitivity evidence**')
                        for line in r['category_explanation']['sensitivity']:
                            st.markdown(f"<div class='reason sensitivity'>↳ {line}</div>", unsafe_allow_html=True)

                st.caption(
                    'Important: an explanation describes why this trained model produced its own output. '
                    'It is not a separate rule that changes the prediction.'
                )

        # ---------------- Feature inspection ----------------
        st.markdown('---')
        st.header('3. Exact input features calculated from this paragraph')
        style = extract_41_style_features(text)
        with st.expander('Show all exact 41 style features', expanded=False):
            style_df = pd.DataFrame([
                {
                    '#': i,
                    'Feature': name,
                    'Meaning': STYLE_LABELS.get(name, name),
                    'Value': style[name],
                }
                for i, name in enumerate(STYLE_FEATURE_NAMES, 1)
            ])
            st.dataframe(style_df, use_container_width=True, hide_index=True)

        catf = extract_category_keyword_features(text)
        with st.expander('Show all 21 category-keyword features', expanded=False):
            cat_df = pd.DataFrame([
                {'Feature': k, 'Value': v} for k, v in catf.items()
            ])
            st.dataframe(cat_df, use_container_width=True, hide_index=True)

        with st.expander('What else did full-feature ML models receive?', expanded=False):
            # Read dimensions from the first successfully loaded bundle.
            first_bundle = load_bundle(MODEL_FILES[results[0]['model']])
            dims = first_bundle.get('feature_dimensions', first_bundle['feature_artifacts'].get('dimensions', {}))
            st.write(
                f"In addition to the **41 style features** and **{dims.get('category_keyword', 21)} category-keyword features**, "
                f"full-view models receive **{dims.get('word_tfidf', 0)} learned word TF-IDF features** "
                f"and **{dims.get('char_tfidf', 0)} learned character TF-IDF features**."
            )
            st.write(
                'Tree-family models in this training package use only the interpretable handcrafted block '
                '(41 style + 21 category-keyword features), exactly as specified during training.'
            )

        # ---------------- JSON download ----------------
        export = {
            'rule_based': {
                'source': rule_src['prediction'],
                'source_confidence': rule_src['confidence'],
                'category': rule_cat['category'],
            },
            'ml_ensemble': {
                'source': source_ens,
                'source_votes': source_counts,
                'category': cat_ens,
                'category_votes': cat_counts,
            },
            'models': [
                {
                    'model': r['model'],
                    'source': r['source'],
                    'category': r['category'],
                    'source_explanation': r['source_explanation'],
                    'category_explanation': r['category_explanation'],
                }
                for r in results
            ],
            'style_41': style,
            'category_keyword_features': catf,
        }
        st.download_button(
            '⬇️ Download full analysis JSON',
            data=json.dumps(export, ensure_ascii=False, indent=2, default=str),
            file_name='gujarati_source_category_analysis.json',
            mime='application/json',
            use_container_width=True,
        )

    else:
        st.error('No trained model could be evaluated. Check trained_models/ and package-version compatibility.')


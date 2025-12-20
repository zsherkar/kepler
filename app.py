import os
import random
import time
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# Kepler v2 — Aesthetic UI
# =========================

def inject_kepler_css():
    st.markdown(
        """
<style>
/* --- Old-world parchment aesthetic --- */
:root{
  --ink:#2b2116;
  --ink2:#4b3a28;
  --paper:#f3ead6;
  --paper2:#efe2c4;
  --accent:#7a4e2d;
  --accent2:#a06b3b;
  --rule: rgba(43,33,22,0.18);
}

html, body, [class*="css"]  {
  font-family: "Georgia", "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
  color: var(--ink) !important;
}

/* Background */
.stApp {
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(255,255,255,0.55), rgba(255,255,255,0) 60%),
    radial-gradient(900px 500px at 80% 20%, rgba(255,255,255,0.40), rgba(255,255,255,0) 60%),
    linear-gradient(180deg, var(--paper) 0%, var(--paper2) 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(239,226,196,0.92), rgba(243,234,214,0.92)) !important;
  border-right: 1px solid var(--rule) !important;
}
section[data-testid="stSidebar"] *{
  color: var(--ink) !important;
}

/* Cards */
.kepler-card{
  background: rgba(255,255,255,0.35);
  border: 1px solid var(--rule);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 30px rgba(43,33,22,0.08);
}
.kepler-rule{
  border-top: 1px solid var(--rule);
  margin: 10px 0 14px 0;
}
.kepler-small{
  color: var(--ink2);
  font-size: 0.95rem;
  line-height: 1.35rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: var(--ink) !important;
  letter-spacing: 0.4px;
}

/* =========================
   Global readability patch
   ========================= */

/* Force ink text everywhere (kills white-on-parchment) */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stText, .stCaption, .stAlert, .stInfo, .stSuccess, .stWarning, .stError,
label, small, div, p, li, span {
  color: var(--ink) !important;
}

/* Metrics: ensure label + value are ink */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
  color: var(--ink) !important;
}

/* =========================
   Top bar / Deploy area
   ========================= */

header[data-testid="stHeader"]{
  background: rgba(243,234,214,0.92) !important;
  border-bottom: 1px solid var(--rule) !important;
}

div[data-testid="stToolbar"]{
  background: transparent !important;
}

/* Make header buttons visible */
header[data-testid="stHeader"] button,
div[data-testid="stToolbar"] button{
  background: rgba(255,255,255,0.55) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
}

/* =========================
   Buttons (high contrast)
   ========================= */

.stButton > button{
  background: linear-gradient(180deg, rgba(122,78,45,0.96), rgba(95,58,30,0.96)) !important;
  color: #fff !important;
  font-weight: 650 !important;
  border: 1px solid rgba(43,33,22,0.35) !important;
  border-radius: 12px !important;
  padding: 0.55rem 1.0rem !important;
  box-shadow: 0 10px 25px rgba(43,33,22,0.18) !important;
}

.stButton > button *{
  color: #fff !important;
}

.stButton > button:hover{
  filter: brightness(1.06) !important;
  transform: translateY(-1px);
}

/* =========================
   Selectbox / Multiselect inputs
   ========================= */

/* Make widget text ink */
[data-testid="stSelectbox"] div,
[data-testid="stMultiSelect"] div,
[data-testid="stTextInput"] div,
[data-testid="stNumberInput"] div,
[data-testid="stTextArea"] div {
  color: var(--ink) !important;
}

/* Control surface (prevents black fill) */
div[data-baseweb="select"] > div{
  background-color: rgba(255,255,255,0.55) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 12px !important;
}

/* Selected value text inside select */
div[data-baseweb="select"] span{
  color: var(--ink) !important;
}

/* Placeholder text */
div[data-baseweb="select"] input::placeholder{
  color: rgba(43,33,22,0.60) !important;
}

/* Dropdown open menu (BaseWeb popover portal) */
div[data-baseweb="popover"] > div{
  background: rgba(243,234,214,0.98) !important;
  border: 1px solid var(--rule) !important;
  color: var(--ink) !important;
}

/* Options list + hover state */
ul[role="listbox"]{
  background: rgba(243,234,214,0.98) !important;
  border: 1px solid var(--rule) !important;
}

ul[role="listbox"] li{
  color: var(--ink) !important;
}

ul[role="listbox"] li:hover{
  background: rgba(122,78,45,0.12) !important;
  color: var(--ink) !important;
}

/* Multiselect "pill" tags */
span[data-baseweb="tag"]{
  background-color: rgba(122,78,45,0.12) !important;
  border: 1px solid rgba(122,78,45,0.25) !important;
  color: var(--ink) !important;
}

/* Links */
a, a:visited {
  color: var(--accent) !important;
}

/* =========================================
   NUCLEAR FIX: all dropdown / popover menus
   (BaseWeb + Streamlit menus rendered in portals)
   ========================================= */

div[data-baseweb="popover"] *,
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
  border-color: var(--rule) !important;
}

div[data-baseweb="menu"],
div[data-baseweb="menu"] * {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
}

[role="menu"],
[role="menu"] * {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
}

[role="listbox"],
[role="listbox"] * {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
}

[role="option"]:hover,
[role="menuitem"]:hover {
  background: rgba(122,78,45,0.12) !important;
  color: var(--ink) !important;
}

div[data-testid="stMainMenuPopover"],
div[data-testid="stMainMenuPopover"] * {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
}

div[data-testid="stPopoverBody"],
div[data-testid="stPopoverBody"] * {
  background: rgba(243,234,214,0.98) !important;
  color: var(--ink) !important;
}

div[data-baseweb="menu"] hr,
div[data-testid="stMainMenuPopover"] hr {
  border-color: var(--rule) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def img_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


def show_asset_image(filename, caption=None, width=None):
    path = img_path("assets", filename)
    if not os.path.exists(path):
        st.info(f"Missing asset: assets/{filename} (optional, UI-only).")
        return
    try:
        st.image(path, caption=caption, width=width)
    except Exception as e:
        st.warning(
            f"Could not load assets/{filename}. Try saving it as .png or .jpg instead. ({type(e).__name__})"
        )


# =========================
# Kepler-specific dataset
# =========================

def build_kepler_query_pool():
    items = [
        ("What are Kepler’s three laws of planetary motion?", "laws"),
        ("What does Kepler’s second law state in plain language?", "laws"),
        ("How does Kepler’s third law relate orbital period and distance?", "laws"),
        ("Why was Kepler’s first law a break from circular orbits?", "laws"),

        ("Who was Tycho Brahe and how did he influence Kepler?", "teacher_student"),
        ("What was Kepler’s relationship with Tycho Brahe like?", "teacher_student"),
        ("How did Kepler use Brahe’s observations after Brahe died?", "teacher_student"),

        ("What is Astronomia Nova and why is it important?", "works"),
        ("What are the Rudolphine Tables?", "works"),
        ("What is Harmonices Mundi associated with?", "works"),

        ("Where was Johannes Kepler born?", "bio"),
        ("In which century did Kepler live?", "bio"),
        ("What major roles did Kepler hold (e.g., imperial mathematician)?", "bio"),

        ("Why did Kepler’s mother face a witchcraft trial?", "context"),
        ("How did religious conflict shape Kepler’s career?", "context"),
        ("What were the broader conditions of the Scientific Revolution around Kepler?", "context"),

        ("How did Kepler’s work strengthen heliocentrism?", "impact"),
        ("What did Kepler contribute beyond astronomy (optics, geometry)?", "impact"),
        ("Why are Kepler’s laws useful for later physics (Newton)?", "impact"),
    ]
    return [{"id": i, "question": q, "gold_label": y} for i, (q, y) in enumerate(items)]


LABELS = ["laws", "teacher_student", "works", "bio", "context", "impact"]
LABEL_TO_NAME = {
    "laws": "Kepler’s Laws",
    "teacher_student": "Teacher–Student (Tycho → Kepler)",
    "works": "Major Works",
    "bio": "Biography",
    "context": "Political/Religious Context",
    "impact": "Scientific Impact",
}


# =========================
# Teacher (parent) — hidden model
# =========================

def train_hidden_teacher(seed=42):
    random.seed(seed)

    hidden = {
        "laws": [
            "Kepler’s first law: planets move in ellipses with the Sun at one focus.",
            "Second law: equal areas are swept in equal times.",
            "Third law: orbital period squared is proportional to semi-major axis cubed.",
            "Elliptical orbits replaced perfect circles in astronomy.",
        ],
        "teacher_student": [
            "Tycho Brahe mentored Kepler and employed him.",
            "Kepler inherited Brahe’s precise observational data after his death.",
            "The Brahe–Kepler relationship was tense but foundational.",
        ],
        "works": [
            "Astronomia Nova introduced elliptical orbits and the area law.",
            "Rudolphine Tables were accurate astronomical tables based on Brahe’s data.",
            "Harmonices Mundi connected geometry, harmony, and planetary motion.",
        ],
        "bio": [
            "Kepler was born in 1571 and died in 1630.",
            "He worked as an imperial mathematician in the Holy Roman Empire.",
            "Kepler lived in 17th-century Europe.",
        ],
        "context": [
            "Kepler’s life was shaped by confessional conflict and patronage politics.",
            "His mother was tried for witchcraft; Kepler helped defend her.",
            "Religious tensions influenced his appointments and mobility.",
        ],
        "impact": [
            "Kepler’s laws helped pave the way for Newtonian gravity.",
            "He contributed to optics and early ideas about vision.",
            "His work supported heliocentrism by matching observation to theory.",
        ],
    }

    X, y = [], []
    for label, texts in hidden.items():
        for t in texts:
            X.append(t)
            y.append(label)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xv = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xv, y)

    return vec, clf

def teacher_query(vec, clf, questions, mode="soft"):
    Xv = vec.transform(questions)
    probs = clf.predict_proba(Xv)
    labels = clf.classes_

    # Hard-label access: argmax only
    if mode == "hard":
        hard = labels[np.argmax(probs, axis=1)]
        return {"hard": hard, "labels": labels}

    # Soft-label access with temperature scaling
    tau = float(st.session_state.get("temperature", 1.0))

    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / tau
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs_tau = exp / np.sum(exp, axis=1, keepdims=True)

    return {"soft": probs_tau, "labels": labels}


# =========================
# Student (attacker model)
# =========================

def train_student_from_distilled(distilled_rows, seed=42, soft_samples_per_row=6):
    """
    Soft-label rows:
      - sample pseudo-labels using the stored teacher label order per row
      - this approximates "learning the distribution" while staying in sklearn
    Hard-label rows:
      - one label per question
    """
    rng = np.random.default_rng(int(time.time()))

    X_train = []
    y_train = []

    for r in distilled_rows:
        q = r["question"]

        if r.get("teacher_soft_probs") is not None:
            probs = np.array(r["teacher_soft_probs"], dtype=float)
            label_order = r.get("teacher_label_order", None)

            # Safety: if label order wasn't stored for some reason, fall back to hard label once
            if not label_order:
                X_train.append(q)
                y_train.append(r["teacher_hard_label"])
                continue

            probs = probs / probs.sum()
            sampled = rng.choice(label_order, size=soft_samples_per_row, p=probs)

            for lab in sampled:
                X_train.append(q)
                y_train.append(lab)
        else:
            X_train.append(q)
            y_train.append(r["teacher_hard_label"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xv = vec.fit_transform(X_train)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xv, y_train)

    return vec, clf


# =========================
# App State
# =========================

def init_state():
    ss = st.session_state
    ss.setdefault("pool", build_kepler_query_pool())
    ss.setdefault("selected_pool_ids", [])
    ss.setdefault("generated_ids", [])
    ss.setdefault("distilled_rows", [])
    ss.setdefault("teacher_vec", None)
    ss.setdefault("teacher_clf", None)
    ss.setdefault("teacher_mode", "soft")
    ss.setdefault("student_vec", None)
    ss.setdefault("student_clf", None)
    ss.setdefault("last_teacher_outputs", None)


# =========================
# Main App
# =========================

st.set_page_config(page_title="Kepler", layout="wide")
inject_kepler_css()
init_state()

left, right = st.columns([2.2, 1.0], vertical_alignment="top")

with left:
    st.markdown(
        """
<div class="kepler-card">
  <h1 style="margin-bottom:0.35rem;">Kepler</h1>

  <div style="font-size:1.15rem; font-weight:600; margin-bottom:0.4rem;">
    A Distillation Attack Simulator
  </div>

  <div class="kepler-small" style="font-weight:600;">
    Ziauddin Sherkar
  </div>

  <div class="kepler-small">
    Final Thesis
  </div>

  <div class="kepler-small">
    Digital Counterintelligence, Fall 2025
  </div>
  <div class="kepler-rule"></div>
  <div class="kepler-small">
    Kepler is a distillation attack simmulator, named after <b>Johannes Kepler</b> (1571–1630), whose breakthroughs were built on his “teacher” Tycho Brahe’s observations.
    Kepler shows how a student can learn from a teacher’s outputs — and in time, <i>eclipse</i> the teacher’s practical value by
    reproducing its behavior with less cost. While not formally part of the final thesis paper, this simulator is built in the Keplerian 
    spirit of letting careful observation speak — ideally loudly enough to perturb the grading orbit toward bonus points.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

with right:
    show_asset_image("kepler_portrait.webp", caption="Johannes Kepler", width=280)

st.markdown("")

tabs = st.tabs([
    "I. Pre-Requisites",
    "II. Execution Pipeline",
    "III. Inspect Distilled Database",
    "IV. Train Student",
])

with st.sidebar:
    st.markdown("### The Black-Box Interface")
    teacher_mode_ui = st.selectbox(
        "Teacher Label Access",
        ["Soft Labels (pT(y|x))", "Hard Labels (argmax pT(y|x))"]
    )
    st.session_state["temperature"] = st.slider("Soft-Label Temperature (τ)", 0.5, 5.0, 1.5, 0.1)
    st.session_state["teacher_mode"] = "hard" if teacher_mode_ui.startswith("Hard") else "soft"

    st.markdown("### Pipeline Navigation")
    st.caption("You can generate queries, select inputs, query the parent, and build a distilled dataset.")

    st.markdown("###")
    show_asset_image("kepler_laws_diagram.svg", caption="Kepler’s Laws", width=220)

if st.session_state["teacher_vec"] is None:
    vec, clf = train_hidden_teacher(seed=42)
    st.session_state["teacher_vec"] = vec
    st.session_state["teacher_clf"] = clf

teacher_vec = st.session_state["teacher_vec"]
teacher_clf = st.session_state["teacher_clf"]


# -------------------------
# Tab I: Pre-reqs
# -------------------------
with tabs[0]:
    st.markdown(
        """
<div class="kepler-card">
<h3>I. Distillation Pre-requisites </h3>
<div class="kepler-small">
<b>Black-box query access</b> means the attacker cannot see weights, training data, or internal code —
only inputs and outputs via an API-like interface.<br><br>
<b>Minimal infrastructure</b>: the attacker needs a modest environment to store teacher outputs and train a smaller student model,
plus budget/time for queries. The teacher remains expensive; the student is cheaper.<br><br>
<b>Objective spectrum</b>: full replication, partial “good enough” mimicry, or decision-boundary inference.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Tab II: Pipeline
# -------------------------
with tabs[1]:
    st.markdown(
        """
<div class="kepler-card">
<h3>II. The Pipeline</h3>
<div class="kepler-small">
You can: (1) generate a query pool, (2) choose which inputs to use, (3) query the teacher, and (4) record outputs to build the distilled dataset.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="top")

    with c1:
        st.markdown(
            """
<div class="kepler-card">
<h4>Step 1 — Generate a Randomized Query Set</h4>
<div class="kepler-small">
These are questions about Kepler (life, works, laws, context). In a real attack, this could be prompts, images, or text inputs.
</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        n_gen = st.slider("How many candidate queries to generate?", 5, 18, 10, 1)
        if st.button("Generate Randomized Candidate Set"):
            pool = st.session_state["pool"]
            ids = [x["id"] for x in pool]
            st.session_state["generated_ids"] = random.sample(ids, k=min(n_gen, len(ids)))
            st.session_state["selected_pool_ids"] = []

        gen_ids = st.session_state["generated_ids"]
        if gen_ids:
            pool_map_local = {x["id"]: x for x in st.session_state["pool"]}
            st.write("**Generated Candidates:**")
            for i in gen_ids:
                st.write(f"- {pool_map_local[i]['question']}")
        else:
            st.info("Click **Generate Randomized Candidate Set**.")

    with c2:
        st.markdown(
            """
<div class="kepler-card">
<h4>Step 2 — Select Inputs to Query</h4>
<div class="kepler-small">
This mirrors the attacker choosing which probes are most informative (random, boundary-focused, etc.).
</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        pool_map = {x["id"]: x for x in st.session_state["pool"]}
        options = st.session_state["generated_ids"]
        selected = st.multiselect(
            "Choose queries to send to the teacher",
            options=options,
            format_func=lambda i: pool_map[i]["question"] if i in pool_map else str(i),
        )
        st.session_state["selected_pool_ids"] = selected
        st.caption("Tip: start with 6–10 selected queries for a quick demo.")

    st.markdown("")

    st.markdown(
        """
<div class="kepler-card">
<h4>Step 3 — Query the Teacher (Black-Box)</h4>
<div class="kepler-small">
Soft-label access returns a full probability distribution; hard-label access returns only the top answer.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    selected_ids = st.session_state["selected_pool_ids"]
    if st.button("Query Teacher with Selected Inputs"):
        if not selected_ids:
            st.warning("Select at least one query first.")
        else:
            questions = [pool_map[i]["question"] for i in selected_ids]
            out = teacher_query(teacher_vec, teacher_clf, questions, mode=st.session_state["teacher_mode"])
            st.session_state["last_teacher_outputs"] = {
                "questions": questions,
                "out": out,
                "timestamp": time.time(),
            }

    last = st.session_state.get("last_teacher_outputs")
    if last:
        out = last["out"]
        questions = last["questions"]
        labels = out["labels"]

        st.write("**Teacher Outputs (Preview):**")
        for idx, q in enumerate(questions):
            if "hard" in out:
                st.write(f"- {q}  →  **{LABEL_TO_NAME.get(out['hard'][idx], out['hard'][idx])}**")
            else:
                probs = out["soft"][idx]

                order = np.argsort(-probs)
                top3 = [(labels[j], float(probs[j])) for j in order[:3]]
                top_label = labels[int(order[0])]

                st.write(f"- {q}  →  **{LABEL_TO_NAME.get(top_label, top_label)}**")

                top3_fmt = ", ".join(
                    [f"{LABEL_TO_NAME.get(l, l)}: {p:.2f}" for l, p in top3]
                )
                st.caption(f"Soft labels (top-3): {top3_fmt}")

    st.markdown("")

    st.markdown(
        """
<div class="kepler-card">
<h4>Step 4 — Record Outputs and Build the Distilled Dataset</h4>
<div class="kepler-small">
This stores (input, teacher output). That dataset is what trains the student.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Add Teacher Outputs to Distilled Database"):
        last = st.session_state.get("last_teacher_outputs")
        if not last:
            st.warning("Query the teacher first.")
        else:
            out = last["out"]
            questions = last["questions"]
            labels = out["labels"]

            for i, q in enumerate(questions):
                row = {"question": q}

                if "hard" in out:
                    row["teacher_hard_label"] = out["hard"][i]
                    row["teacher_soft_probs"] = None
                    row["teacher_label_order"] = list(labels)
                else:
                    probs = out["soft"][i]
                    row["teacher_soft_probs"] = probs
                    row["teacher_hard_label"] = labels[int(np.argmax(probs))]
                    row["teacher_label_order"] = list(labels)

                st.session_state["distilled_rows"].append(row)

            st.success(f"Added {len(questions)} rows to the distilled dataset.")


# -------------------------
# Tab III: Inspect Distilled Database
# -------------------------
with tabs[2]:
    rows = st.session_state["distilled_rows"]

    st.markdown(
        """
<div class="kepler-card">
<h3>III. Distilled Database</h3>
<div class="kepler-small">
This is the attacker-constructed dataset: each input paired with the teacher’s outputs.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("No distilled rows yet. Go to Pipeline tab and add rows.")
    else:
        st.write(f"**Rows Collected:** {len(rows)}")
        for r in rows[:20]:
            st.write(f"- {r['question']}  →  **{LABEL_TO_NAME.get(r['teacher_hard_label'], r['teacher_hard_label'])}**")
        if len(rows) > 20:
            st.caption("Showing first 20 rows.")


# -------------------------
# Tab IV: Train Student
# -------------------------
with tabs[3]:
    rows = st.session_state["distilled_rows"]

    st.markdown(
        """
<div class="kepler-card">
<h3>IV. Train the Student</h3>
<div class="kepler-small">
We train a small student model on the distilled dataset to replicate the teacher’s behavior.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("Collect a distilled dataset first (Pipeline tab).")
    else:
        if st.button("Train student on Distilled Dataset"):
            svec, sclf = train_student_from_distilled(rows, seed=42, soft_samples_per_row=6)
            st.session_state["student_vec"] = svec
            st.session_state["student_clf"] = sclf
            st.success("Student trained.")

        if st.session_state["student_vec"] is not None:
            st.markdown("#### Quick Evaluation (Pedagogical)")
            st.caption("In a real attack, you may not have ground truth. Here we show both teacher-agreement and gold-label accuracy for learning.")

            pool = st.session_state["pool"]
            questions = [x["question"] for x in pool]
            gold = [x["gold_label"] for x in pool]

            tout = teacher_query(teacher_vec, teacher_clf, questions, mode="soft")
            teacher_labels = tout["labels"]
            teacher_pred = [teacher_labels[int(np.argmax(p))] for p in tout["soft"]]

            svec = st.session_state["student_vec"]
            sclf = st.session_state["student_clf"]
            Xv = svec.transform(questions)
            student_pred = sclf.predict(Xv)

            teacher_acc = accuracy_score(gold, teacher_pred)
            student_acc = accuracy_score(gold, student_pred)
            agree = np.mean(np.array(teacher_pred) == np.array(student_pred))

            c1, c2, c3 = st.columns(3)
            c1.metric("Teacher Accuracy (Gold)", f"{teacher_acc:.3f}")
            c2.metric("Student Accuracy (Gold)", f"{student_acc:.3f}")
            c3.metric("Student–Teacher Agreement", f"{agree:.3f}")

            from collections import defaultdict
            import matplotlib.pyplot as plt

            by_cat_total = defaultdict(int)
            by_cat_agree = defaultdict(int)

            for item, tpred, spred in zip(pool, teacher_pred, student_pred):
                cat = item["gold_label"]
                by_cat_total[cat] += 1
                if tpred == spred:
                    by_cat_agree[cat] += 1

            cats = [c for c in LABELS if c in by_cat_total]
            vals = [by_cat_agree[c] / by_cat_total[c] for c in cats]
            names = [LABEL_TO_NAME.get(c, c) for c in cats]

            fig = plt.figure()
            x = np.arange(len(names))
            plt.bar(x, vals)
            plt.ylim(0, 1.0)
            plt.xticks(x, names, rotation=25, ha="right")
            plt.ylabel("Agreement (Student vs Teacher)")
            plt.title("Behavioral imitation by category")
            st.pyplot(fig)

            st.markdown(
                """
<div class="kepler-card">
<div class="kepler-small">
<b>Interpretation:</b> as your distilled dataset grows, agreement rises — especially with soft-label access.
This captures the core mechanics of a distillation attack: the student becomes a cheaper behavioral replica.
</div>
</div>
                """,
                unsafe_allow_html=True,
            )

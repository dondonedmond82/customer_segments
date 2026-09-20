"""
bank_manager_insights_app.py

A Panel dashboard for a bank manager: a sidebar menu of financial charts,
each given a meaningful business name (Product Mix Overview, Average
Product Value, Product Value Ranking, Customer Variation Factors, Deposits
vs Loans, Product Cross-Correlation, Value Distribution & Outliers,
Customer Records Explorer, Portfolio Summary Statistics, Deposit Balance
Distribution, Regional Customer Distribution), PLUS a set of NLP-driven
views on customer feedback text (Customer Sentiment Overview, Common
Feedback Themes, Feedback Topic Discovery, High-Value Retention Risk).

Every menu item's description includes a short note on why the insight is
trustworthy (what it's based on, and its limits) so it can be presented
directly to a bank manager for decision-making.

Data:
    - Loads "bank_customers.csv" if present in the working directory.
    - Otherwise generates synthetic customer + feedback data so the app
      runs end-to-end out of the box. REPLACE with real, anonymized
      customer and feedback data before using this for actual decisions.

Run with:
    panel serve bank_manager_insights_app.py --show --autoreload
"""

###########################################
# Suppress matplotlib user warnings
###########################################
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server rendering
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import panel as pn

pn.extension("tabulator")

# Banking product-line columns (reframed from generic spend categories)
DATA_COLUMNS = ["Deposits", "Loans", "CreditCard", "Savings", "Investments", "Insurance"]

# Figure size sized for a 1366x768 screen: with a 280px sidebar and a 340px
# description column already in the layout, the graph column has roughly
# 650-700px of usable width, so the chart itself is kept a bit smaller than
# that (and shorter, so a chart plus its header/description still fits
# within a 768px-tall viewport without scrolling).
FIGSIZE = (7, 3.3)
GRAPH_WIDTH = 960  # pixel width used for panes/sizing
GRAPH_HEIGHT = 480 # pixel height used for panes/sizing

# ---------------------------------------------------------------------------
# Professional 2-color theme: white background, blue foreground/text, with
# a slightly deeper blue used sparingly as an accent for emphasis.
# ---------------------------------------------------------------------------
BG_COLOR = "#FFFFFF"      # background colour (white)
FG_COLOR = "#1B3A6B"      # foreground / text colour (deep blue)
CARD_COLOR = "#F1F5FC"    # very light blue-tinted white for card panels
ACCENT_COLOR = "#1E5FBF"  # brighter blue accent for buttons/highlights
GRID_COLOR = "#D6E1F0"
FONT_FAMILY = "'Poppins', 'Segoe UI', Helvetica, Arial, sans-serif"
FONT_URL = "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap"

# Apply the same palette/typography to every matplotlib chart so figures
# blend seamlessly into the interface rather than sitting in a white box.
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": FG_COLOR,
    "xtick.color": FG_COLOR,
    "ytick.color": FG_COLOR,
    "text.color": FG_COLOR,
    "axes.titleweight": "600",
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.sans-serif": ["Poppins", "Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Data loading (customer financial data + synthetic feedback text)
# ---------------------------------------------------------------------------

POSITIVE_SNIPPETS = [
    "Great service and quick loan approval.",
    "The mobile app makes managing my account so easy.",
    "My relationship manager is always responsive and helpful.",
    "Very satisfied with the savings account interest rate.",
    "Fast, friendly service at the branch every time.",
    "Investment advice from the bank has been excellent.",
    "Credit card rewards program is genuinely useful.",
    "Appreciate the transparent fee structure.",
]
NEGATIVE_SNIPPETS = [
    "Long wait times at the branch are frustrating.",
    "Customer support was slow to resolve my complaint.",
    "Hidden fees on my account were disappointing.",
    "The loan approval process took far too long.",
    "App keeps logging me out unexpectedly.",
    "Interest rates on savings feel uncompetitive.",
    "Had trouble reaching anyone about a card dispute.",
    "Not happy with how a recent issue was handled.",
]
NEUTRAL_SNIPPETS = [
    "Opened a new account last month.",
    "Received a statement update via email.",
    "Visited the branch to ask about loan options.",
    "Considering moving some savings into investments.",
    "Called about credit card terms and conditions.",
    "Checked balance and made a transfer today.",
]


def load_data(path="bank_customers.csv", n_synthetic=440, random_state=42):
    '''
    Load bank_customers.csv if present; otherwise generate synthetic
    customer + feedback data so the app can run end-to-end for review
    before wiring in real, anonymized production data.
    '''
    rng = np.random.default_rng(random_state)

    if os.path.exists(path):
        full_data = pd.read_csv(path)
        source = f"Loaded data from {path}"
        if "Feedback" not in full_data.columns:
            full_data["Feedback"] = _synthesize_feedback(len(full_data), rng)
    else:
        data = {}
        for col in DATA_COLUMNS:
            data[col] = rng.lognormal(mean=8, sigma=1.0, size=n_synthetic).astype(int)
        full_data = pd.DataFrame(data)
        full_data["Region"] = rng.integers(1, 4, size=n_synthetic)  # 1, 2, or 3
        full_data["Feedback"] = _synthesize_feedback(n_synthetic, rng)
        source = ("bank_customers.csv not found -- using SYNTHETIC sample data "
                 "and SYNTHETIC feedback text. Replace with real, anonymized "
                 "customer and feedback data before using this for decisions.")
    return full_data, source


def _synthesize_feedback(n, rng):
    feedback = []
    for _ in range(n):
        bucket = rng.choice(["pos", "neg", "neu"], p=[0.45, 0.30, 0.25])
        pool = {"pos": POSITIVE_SNIPPETS, "neg": NEGATIVE_SNIPPETS, "neu": NEUTRAL_SNIPPETS}[bucket]
        k = rng.integers(1, 3)
        feedback.append(" ".join(rng.choice(pool, size=k, replace=False)))
    return feedback


# ---------------------------------------------------------------------------
# Lightweight lexicon-based sentiment scoring (no external downloads needed)
# ---------------------------------------------------------------------------

POS_WORDS = {
    "great", "quick", "easy", "responsive", "helpful", "satisfied", "excellent",
    "friendly", "fast", "useful", "appreciate", "transparent", "good", "love",
    "happy", "convenient", "reliable",
}
NEG_WORDS = {
    "long", "frustrating", "slow", "disappointing", "hidden", "trouble",
    "unhappy", "not", "unexpectedly", "uncompetitive", "issue", "complaint",
    "delay", "poor", "bad", "difficult", "problem",
}


def score_sentiment(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


def add_sentiment(full_data):
    if "Sentiment" not in full_data.columns:
        full_data = full_data.copy()
        full_data["Sentiment"] = full_data["Feedback"].apply(score_sentiment)
    return full_data


# ---------------------------------------------------------------------------
# Chart builders -- financial views (reframed to banking product lines)
# ---------------------------------------------------------------------------

def make_pie(full_data):
    totals = full_data[DATA_COLUMNS].sum()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.pie(totals, labels=totals.index, autopct='%1.0f%%', startangle=90,
          textprops={"fontsize": 8})
    ax.set_title("Share of Total Balances/Spend by Product Line", fontsize=10)
    ax.axis('equal')
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_bar(full_data):
    means = full_data[DATA_COLUMNS].mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(means.index, means.values, color="#4C72B0")
    ax.set_ylabel("Average per Customer", fontsize=9)
    ax.set_title("Average Value per Product Line", fontsize=10)
    ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=8)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_barh(full_data):
    means = full_data[DATA_COLUMNS].mean().sort_values()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(means.index, means.values, color="#55A868")
    ax.set_xlabel("Average per Customer", fontsize=9)
    ax.set_title("Average Value per Product Line (Sorted)", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_line(full_data):
    log_data = np.log(full_data[DATA_COLUMNS].replace(0, 1))
    pca = PCA(n_components=len(DATA_COLUMNS), random_state=42)
    pca.fit(log_data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o', color="#C44E52")
    ax.set_xlabel("Number of Components", fontsize=9)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=9)
    ax.set_title("How Many Factors Explain Customer Variation", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_scatter(full_data):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(full_data["Deposits"], full_data["Loans"], alpha=0.5, color="#8172B2", s=15)
    ax.set_xlabel("Deposits", fontsize=9)
    ax.set_ylabel("Loans", fontsize=9)
    ax.set_title("Deposits vs. Loan Balances", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_heatmap(full_data):
    corr = full_data[DATA_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Cross-Product Correlation", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_boxplot(full_data):
    log_data = np.log(full_data[DATA_COLUMNS].replace(0, 1))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot([log_data[col] for col in DATA_COLUMNS], labels=DATA_COLUMNS)
    ax.set_ylabel("Log(Value)", fontsize=9)
    ax.set_title("Spread of Product Values (Log Scale)", fontsize=10)
    ax.set_xticklabels(DATA_COLUMNS, rotation=30, ha="right", fontsize=7)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_table(full_data):
    cols = DATA_COLUMNS + [c for c in ["Sentiment", "Feedback"] if c in full_data.columns]
    return pn.widgets.Tabulator(
        full_data[cols].head(50), pagination="local", page_size=8,
        width=GRAPH_WIDTH, show_index=False,
    )


def make_text(full_data):
    summary = full_data[DATA_COLUMNS].describe().round(0)
    lines = ["Summary Statistics", "=" * 26, ""]
    lines.append(f"{'Product':<14}{'Mean':>9}{'Std':>9}{'Min':>9}{'Max':>9}")
    for col in DATA_COLUMNS:
        lines.append(
            f"{col:<14}{summary.loc['mean', col]:>9}{summary.loc['std', col]:>9}"
            f"{summary.loc['min', col]:>9}{summary.loc['max', col]:>9}"
        )
    text_block = "\n".join(lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis('off')
    ax.text(0, 1, text_block, family="monospace", fontsize=8, va="top", ha="left")
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_histogram(full_data):
    log_dep = np.log(full_data["Deposits"].replace(0, 1))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(log_dep, bins=25, color="#64B5CD", edgecolor="black")
    ax.set_xlabel("Log(Deposits)", fontsize=9)
    ax.set_ylabel("Number of Customers", fontsize=9)
    ax.set_title("Distribution of Deposit Balances", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_map(full_data):
    # No mapping/geocoding service is available in this environment, so this
    # is a stylized layout of customers by Region rather than a true map.
    region_coords = {1: (-0.5, 0.5), 2: (0.5, 0.5), 3: (0.0, -0.5)}
    if "Region" not in full_data.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axis('off')
        ax.text(0.5, 0.5, "No 'Region' column found.", ha="center", va="center", fontsize=10)
        return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for region, (cx, cy) in region_coords.items():
        mask = full_data["Region"] == region
        n = mask.sum()
        jx = cx + rng.normal(0, 0.08, n)
        jy = cy + rng.normal(0, 0.08, n)
        ax.scatter(jx, jy, label=f"Region {region}", alpha=0.6, s=12)
    ax.set_title("Customers by Region (Illustrative)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


# ---------------------------------------------------------------------------
# NLP-driven views on customer feedback
# ---------------------------------------------------------------------------

def make_sentiment_bar(full_data):
    full_data = add_sentiment(full_data)
    counts = full_data["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#55A868", "#8C8C8C", "#C44E52"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Number of Customers", fontsize=9)
    ax.set_title("Customer Feedback Sentiment", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_keyword_frequency(full_data):
    vectorizer = CountVectorizer(stop_words="english", max_features=15)
    counts = vectorizer.fit_transform(full_data["Feedback"].fillna(""))
    freqs = np.asarray(counts.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()
    order = np.argsort(freqs)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(np.array(words)[order], freqs[order], color="#4C72B0")
    ax.set_xlabel("Mentions", fontsize=9)
    ax.set_title("Most Common Words in Feedback", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_topic_summary(full_data):
    vectorizer = CountVectorizer(stop_words="english", max_features=200, min_df=2)
    doc_term = vectorizer.fit_transform(full_data["Feedback"].fillna(""))
    n_topics = 3
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15)
    lda.fit(doc_term)
    words = vectorizer.get_feature_names_out()

    lines = ["Feedback Topics (auto-detected)", "=" * 32, ""]
    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-6:][::-1]
        top_words = ", ".join(words[j] for j in top_idx)
        lines.append(f"Topic {i + 1}: {top_words}")
        lines.append("")
    text_block = "\n".join(lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis('off')
    ax.text(0, 1, text_block, family="monospace", fontsize=8, va="top", ha="left", wrap=True)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_sentiment_vs_value(full_data):
    full_data = add_sentiment(full_data)
    full_data = full_data.copy()
    full_data["TotalValue"] = full_data[DATA_COLUMNS].sum(axis=1)
    sentiment_order = ["Negative", "Neutral", "Positive"]
    colors = {"Negative": "#C44E52", "Neutral": "#8C8C8C", "Positive": "#55A868"}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in sentiment_order:
        subset = full_data[full_data["Sentiment"] == s]
        ax.scatter([s] * len(subset), subset["TotalValue"], alpha=0.4, s=12, color=colors[s])
    ax.set_ylabel("Total Customer Value", fontsize=9)
    ax.set_title("High-Value Customers at Risk (Sentiment vs. Value)", fontsize=10)
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)


def make_portfolio_tabs(full_data):
    '''
    Bundles three related product-line views into a single 3-tab pane, so a
    manager can flip between them without leaving the "Product Overview"
    section of the menu.
    '''
    tabs = pn.Tabs(
        ("Mix", make_pie(full_data)),
        ("Average", make_bar(full_data)),
        ("Ranking", make_barh(full_data)),
        tabs_location="above",
    )
    return tabs


# ---------------------------------------------------------------------------
# Menu configuration: (name, description with reliability rationale, builder)
# ---------------------------------------------------------------------------

GRAPH_MENU = [
    ("Product Overview (3 Tabs)", "**Product Mix, Average Value & Ranking**\n\n"
            "Three related views of the same product-line data in one "
            "tabbed panel: overall mix (Mix), average holding per line "
            "(Average), and lines ranked lowest-to-highest (Ranking).\n\n"
            "*Why rely on it:* all three tabs are computed the same way as "
            "their standalone counterparts below -- direct sums and "
            "averages over the full customer base -- just grouped for "
            "faster side-by-side review.",
     make_portfolio_tabs),
    ("Product Mix Overview", "**Share of Balances by Product Line**\n\n"
            "Shows what proportion of total customer balances/spend sits in "
            "each product line (deposits, loans, cards, etc.).\n\n"
            "*Why rely on it:* it is a direct sum over every customer record "
            "in the dataset -- no sampling or modeling assumptions -- so it "
            "reflects the actual current book, not a projection.",
     make_pie),
    ("Average Product Value", "**Average Value per Product Line**\n\n"
            "Compares average customer holdings across product lines.\n\n"
            "*Why rely on it:* averages are computed over the full customer "
            "base, giving a stable, easily-auditable baseline for setting "
            "product targets or comparing branches/periods.",
     make_bar),
    ("Product Value Ranking", "**Average Value per Product Line (Sorted)**\n\n"
             "Same comparison as the bar chart, sorted low to high for a "
             "clearer read on which product lines lag.\n\n"
             "*Why rely on it:* sorting removes visual bias from column "
             "order; the underlying numbers are identical to the bar chart.",
     make_barh),
    ("Customer Variation Factors", "**How Many Factors Explain Customer Variation**\n\n"
             "Uses Principal Component Analysis (PCA) on the product-line "
             "data to show how much of the variation between customers can "
             "be captured by a small number of underlying factors.\n\n"
             "*Why rely on it:* PCA is a standard, well-established "
             "statistical technique; the curve directly shows the "
             "information trade-off, so you can see exactly how much "
             "customer complexity is being simplified before trusting a "
             "segmentation built on it.",
     make_line),
    ("Deposits vs Loans", "**Deposits vs. Loans**\n\n"
                "Plots each customer's deposit balance against their loan "
                "balance, to surface relationships (e.g. deposit-rich, "
                "loan-light customers who may be cross-sell targets).\n\n"
                "*Why rely on it:* every point is an actual customer record; "
                "there is no smoothing or estimation, so patterns you see "
                "are literally in the data.",
     make_scatter),
    ("Product Cross-Correlation", "**Cross-Product Correlation**\n\n"
                "Shows how strongly each pair of product lines moves "
                "together across the customer base.\n\n"
                "*Why rely on it:* correlation coefficients are computed "
                "directly from the data with no tuning; strong values "
                "(near +1 or -1) are a reliable signal for bundling or "
                "cross-sell strategy, though correlation does not prove "
                "one product causes uptake of another.",
     make_heatmap),
    ("Value Distribution & Outliers", "**Spread of Product Values**\n\n"
                "Shows the median, typical range, and outliers for each "
                "product line, on a log scale to handle skewed balances.\n\n"
                "*Why rely on it:* boxplots make outlier customers (very "
                "high or low balances) visible rather than hidden inside an "
                "average, which matters for risk and VIP-customer review.",
     make_boxplot),
    ("Customer Records Explorer", "**Customer Records**\n\n"
              "A sortable, searchable table of individual customer records "
              "with their sentiment classification and feedback text.\n\n"
              "*Why rely on it:* this is the raw underlying data behind "
              "every chart in this dashboard -- use it to verify any "
              "aggregate figure by drilling into the actual records.",
     make_table),
    ("Portfolio Summary Statistics", "**Summary Statistics**\n\n"
             "Mean, standard deviation, minimum, and maximum for each "
             "product line.\n\n"
             "*Why rely on it:* these are exact descriptive statistics, not "
             "estimates -- a standard first check before drawing any "
             "conclusion from the charts above.",
     make_text),
    ("Deposit Balance Distribution", "**Distribution of Deposit Balances**\n\n"
                  "Shows how deposit balances are distributed across the "
                  "customer base (log-transformed to handle a few very "
                  "large balances).\n\n"
                  "*Why rely on it:* histograms reveal whether the customer "
                  "base is mostly small depositors, a few large ones, or "
                  "both -- information that a single average would hide.",
     make_histogram),
    ("Regional Customer Distribution", "**Customers by Region (Illustrative)**\n\n"
            "A stylized layout grouping customers by region.\n\n"
            "*Caveat:* no live mapping/geocoding service is available in "
            "this environment, so positions here are illustrative, not true "
            "coordinates. Replace with a real mapping integration (e.g. "
            "branch geolocation data) before using this for site-level "
            "decisions.",
     make_map),
    ("Customer Sentiment Overview", "**Customer Feedback Sentiment**\n\n"
                  "Classifies each piece of customer feedback as Positive, "
                  "Neutral, or Negative using keyword-based sentiment "
                  "scoring, then counts customers in each group.\n\n"
                  "*Why rely on it -- and its limits:* the method is "
                  "transparent (a fixed word list, not a black-box model), "
                  "so every classification can be manually checked. It is "
                  "a lightweight first read on customer mood, not a "
                  "substitute for a validated sentiment model on real "
                  "feedback data -- treat this as a screening signal that "
                  "flags where to look closer, not a final verdict.",
     make_sentiment_bar),
    ("Common Feedback Themes", "**Most Common Words in Feedback**\n\n"
                 "Extracts the most frequently mentioned words across all "
                 "customer feedback (common filler words removed).\n\n"
                 "*Why rely on it:* word counts are a direct, objective tally "
                 "of what customers actually wrote -- useful for spotting "
                 "recurring themes (e.g. 'wait', 'fees', 'app') worth "
                 "investigating, without any subjective interpretation "
                 "layered on top.",
     make_keyword_frequency),
    ("Feedback Topic Discovery", "**Feedback Topics (auto-detected)**\n\n"
               "Uses topic modeling (Latent Dirichlet Allocation) to group "
               "feedback into a small number of recurring themes, each "
               "shown as its most representative words.\n\n"
               "*Why rely on it -- and its limits:* topic modeling is a "
               "well-established unsupervised technique for summarizing "
               "large volumes of text quickly. However, topics are "
               "statistical groupings, not human-labeled categories -- read "
               "the top words as a starting point for a manager's own "
               "judgment, and validate against a sample of actual feedback "
               "before acting on it.",
     make_topic_summary),
    ("High-Value Retention Risk", "**High-Value Customers at Risk**\n\n"
                          "Plots each customer's total product value against "
                          "their feedback sentiment, surfacing high-value "
                          "customers who gave negative feedback -- a "
                          "retention priority list.\n\n"
                          "*Why rely on it:* this combines two things "
                          "already validated elsewhere in this dashboard "
                          "(total value is an exact sum; sentiment is the "
                          "transparent keyword classifier above) into one "
                          "actionable view. It is well-suited for "
                          "prioritizing outreach, but confirm flagged "
                          "accounts against real notes/complaints before "
                          "contacting customers.",
     make_sentiment_vs_value),
]


# ---------------------------------------------------------------------------
# Build the Panel dashboard
# ---------------------------------------------------------------------------

def build_app():
    full_data, source_msg = load_data()
    full_data = add_sentiment(full_data)

    names = [name for name, _, _ in GRAPH_MENU]
    descriptions = {name: desc for name, desc, _ in GRAPH_MENU}
    builders = {name: builder for name, _, builder in GRAPH_MENU}

    menu = pn.widgets.RadioButtonGroup(
        name="Graph", options=names, value=names[0],
        orientation="vertical", button_type="primary", button_style="outline",
        sizing_mode="stretch_width", css_classes=["nav-menu"],
    )

    description_pane = pn.pane.Markdown(
        descriptions[names[0]], sizing_mode="stretch_width", css_classes=["desc-text"],
    )
    graph_container = pn.Column(
        builders[names[0]](full_data), width=GRAPH_WIDTH, css_classes=["graph-frame"],
    )

    def on_menu_change(event):
        choice = event.new
        description_pane.object = descriptions[choice]
        graph_container[:] = [builders[choice](full_data)]

    menu.param.watch(on_menu_change, "value")

    reliability_note = pn.pane.Markdown(
        "Every view is built directly from the customer dataset below using "
        "transparent, well-established statistical and NLP methods -- no "
        "black-box models. Each chart's description explains exactly what "
        "it measures and why it can be trusted, along with its limits. "
        "This tool is meant to support a bank manager's judgment, not "
        "replace it -- combine these signals with domain expertise and "
        "verify flagged accounts before acting.",
        sizing_mode="stretch_width", css_classes=["desc-text"],
    )

    sidebar = pn.Column(
        pn.pane.Markdown("## Insight Menu", css_classes=["sidebar-title"]),
        menu,
        pn.layout.Divider(),
        pn.pane.Markdown(f"**Data source**\n\n{source_msg}", css_classes=["desc-text", "source-note"]),
        css_classes=["sidebar-panel"],
    )

    about_card = pn.Card(
        reliability_note, title="About This Dashboard", collapsible=False,
        css_classes=["info-card"], header_css_classes=["card-header"],
    )

    insight_card = pn.Card(
        pn.Row(
            pn.Column(description_pane, width=340, css_classes=["desc-panel"]),
            graph_container,
            css_classes=["insight-row"],
        ),
        title="Insight Viewer", collapsible=False,
        css_classes=["insight-card"], header_css_classes=["card-header"],
    )

    main = pn.Column(
        about_card,
        insight_card,
        sizing_mode="stretch_width",
        css_classes=["main-column"],
    )

    template = pn.template.FastListTemplate(
        title="Bank Customer Insights",
        sidebar=[sidebar],
        main=[main],
        header_background=BG_COLOR,
        header_color=FG_COLOR,
        background_color=BG_COLOR,
        accent_base_color=ACCENT_COLOR,
        theme="default",
        font=FONT_FAMILY,
        font_url=FONT_URL,
        corner_radius=10,
        shadow=True,
        sidebar_width=280,
        main_max_width="1400px",
    )

    template.config.raw_css.append(f"""
    :root {{
        --app-bg: {BG_COLOR};
        --app-fg: {FG_COLOR};
        --app-card: {CARD_COLOR};
        --app-accent: {ACCENT_COLOR};
    }}
    body, .bk, .mdc-drawer, .mdc-top-app-bar {{
        font-family: {FONT_FAMILY} !important;
        color: {FG_COLOR} !important;
    }}
    h1, h2, h3, h4, .sidebar-title, .card-header {{
        font-family: {FONT_FAMILY} !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        text-align: left;
    }}
    .sidebar-panel {{
        padding: 12px 6px;
    }}
    .sidebar-title {{
        text-align: center;
        color: {FG_COLOR} !important;
        border-bottom: 1px solid {GRID_COLOR};
        padding-bottom: 8px;
        margin-bottom: 6px !important;
    }}
    .source-note {{
        font-size: 0.85em;
        opacity: 0.85;
        text-align: left;
    }}
    .nav-menu .bk-btn {{
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 6px !important;
        margin-bottom: 4px;
        font-weight: 500;
    }}
    .nav-menu .bk-btn.bk-active {{
        background-color: {ACCENT_COLOR} !important;
        color: {BG_COLOR} !important;
        border-color: {ACCENT_COLOR} !important;
    }}
    .info-card, .insight-card {{
        background-color: {CARD_COLOR} !important;
        border-radius: 12px !important;
        border: 1px solid {GRID_COLOR} !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
        padding: 4px 10px 14px 10px;
    }}
    .card-header {{
        background-color: {CARD_COLOR} !important;
        color: {FG_COLOR} !important;
        font-size: 1.1em;
        text-align: left;
        border-bottom: 1px solid {GRID_COLOR};
    }}
    .desc-text, .desc-text p, .desc-text li {{
        color: {FG_COLOR} !important;
        line-height: 1.55;
        text-align: left;
    }}
    .desc-panel {{
        border-right: 1px solid {GRID_COLOR};
        padding-right: 18px;
        margin-right: 10px;
    }}
    .insight-row {{
        align-items: flex-start;
    }}
    .graph-frame {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .main-column {{
        padding: 10px 18px;
    }}
    """)

    return template


# This is what `panel serve bank_manager_insights_app.py` looks for.
app = build_app()
app.servable()

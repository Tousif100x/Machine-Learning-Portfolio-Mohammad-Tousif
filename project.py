# ============================
# IMPORTING LIBRARIES
# ============================

from flask import Flask, request, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression


# ============================
# INITIAL SETUP
# ============================

app = Flask(__name__)

# NOTE: Make sure these files are in the same folder
users = pd.read_csv("users.csv")
feedback = pd.read_csv("feedback.csv")

# Load NLP tools (Run once, then you can comment these)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ============================
# TEXT PREPROCESSING FUNCTION
# ============================

def clean_text(text):
    """
    Convert text to lowercase, remove stopwords,
    and apply lemmatization.
    """
    words = text.lower().split()
    filtered = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(filtered)


# Combine text fields
users["combined_text"] = users["professional_summary"] + " " + users["about_me"]
users["processed_text"] = users["combined_text"].apply(clean_text)


# ============================
# NLP VECTORIZATION
# ============================

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(users["processed_text"])


# ============================
# SIMILARITY FUNCTIONS
# ============================

def text_similarity(user1_id, user2_id):
    """
    Calculate cosine similarity between two users
    based on their text profiles.
    """
    idx1 = users[users["user_id"] == user1_id].index[0]
    idx2 = users[users["user_id"] == user2_id].index[0]
    return cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]


def mbti_score(type1, type2):
    """
    Rule-based MBTI compatibility scoring.
    """
    rules = {
        ("INTJ", "ENFP"): 1.0,
        ("ENFP", "INTJ"): 1.0,
        ("ENTJ", "INFP"): 0.9,
        ("INFP", "ENTJ"): 0.9,
        ("ISTJ", "ESFP"): 0.8,
        ("ESFP", "ISTJ"): 0.8,
    }
    return rules.get((type1, type2), 0.5)


def location_score(loc1, loc2):
    """
    Simple location match scoring.
    """
    return 1.0 if loc1 == loc2 else 0.5


# ============================
# WEIGHTS (INITIAL)
# ============================

w_text = 0.5
w_mbti = 0.3
w_location = 0.2


# ============================
# FINAL COMPATIBILITY SCORE
# ============================

def compute_score(user1_id, user2_id):
    """
    Combine text, MBTI, and location scores
    into a final compatibility score (0–100%).
    """
    global w_text, w_mbti, w_location

    u1 = users[users["user_id"] == user1_id].iloc[0]
    u2 = users[users["user_id"] == user2_id].iloc[0]

    t_score = text_similarity(user1_id, user2_id)
    m_score = mbti_score(u1["mbti"], u2["mbti"])
    l_score = location_score(u1["location"], u2["location"])

    final = (w_text * t_score) + (w_mbti * m_score) + (w_location * l_score)

    return round(final * 100, 2)


# ============================
# RECOMMENDATION FUNCTION
# ============================

def get_recommendations(user_id):
    """
    Return top 5 matching users.
    """
    scores = []

    for other_id in users["user_id"]:
        if other_id != user_id:
            score = compute_score(user_id, other_id)
            scores.append((other_id, score))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]


# ============================
# MODEL ACCURACY
# ============================

def evaluate_accuracy():
    """
    Compare predicted matches with actual user feedback.
    """
    correct = 0
    total = 0

    for _, row in feedback.iterrows():
        score = compute_score(row["user_id"], row["matched_user_id"])
        prediction = 1 if score > 40 else 0

        if prediction == row["action"]:
            correct += 1

        total += 1

    return round((correct / total) * 100, 2) if total > 0 else 0


# ============================
# MODEL TRAINING (ML PART)
# ============================

def train_weights():
    """
    Use Linear Regression to learn optimal weights.
    """
    global w_text, w_mbti, w_location

    X = []
    y = []

    for _, row in feedback.iterrows():
        u1, u2 = row["user_id"], row["matched_user_id"]

        user1 = users[users["user_id"] == u1].iloc[0]
        user2 = users[users["user_id"] == u2].iloc[0]

        t = text_similarity(u1, u2)
        m = mbti_score(user1["mbti"], user2["mbti"])
        l = location_score(user1["location"], user2["location"])

        X.append([t, m, l])
        y.append(row["action"])

    model = LinearRegression()
    model.fit(X, y)

    weights = model.coef_
    total = sum(abs(weights))

    # Normalize weights
    if total > 0:
        w_text = abs(weights[0]) / total
        w_mbti = abs(weights[1]) / total
        w_location = abs(weights[2]) / total


# ============================
# PREPARE DROPDOWN DATA (FOR PROFESSIONAL UI)
# ============================

users_list = []
for _, row in users.iterrows():
    summary = row["professional_summary"]
    short = (summary[:65] + "...") if len(summary) > 65 else summary
    users_list.append({"id": row["user_id"], "summary": short})


# ============================
# PROFESSIONAL HTML UI
# ============================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UNLOX • Intelligent Profile Matching</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root { --primary: #0d6efd; }
        body { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); font-family: 'Segoe UI', system-ui, sans-serif; }
        .navbar { background: linear-gradient(90deg, #0d6efd, #6610f2); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .hero { background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(13, 110, 253, 0.15); }
        .card { border: none; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); transition: all 0.3s; }
        .card:hover { transform: translateY(-4px); box-shadow: 0 15px 35px rgba(0,0,0,0.12); }
        .match-card { border-left: 5px solid #0d6efd; }
        .score-badge { font-size: 1.1rem; font-weight: 700; }
        .results-container { animation: fadeIn 0.6s ease forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <div class="navbar-brand d-flex align-items-center gap-2 fs-3 fw-bold">
                <i class="fas fa-brain"></i> UNLOX
            </div>
            <div class="navbar-text fs-5 fw-semibold">Profile-Based Matching Algorithm</div>
        </div>
    </nav>

    <div class="container py-5">
        <div class="hero p-5 mb-5 text-center">
            <h1 class="display-4 fw-bold text-primary mb-2">Discover Meaningful Connections</h1>
            <p class="lead text-muted mb-4">Powered by semantic text analysis, personality compatibility, and real-time feedback learning</p>
        </div>

        <div class="row justify-content-center mb-5">
            <div class="col-lg-8">
                <form method="POST" class="card p-4">
                    <div class="row align-items-end g-3">
                        <div class="col-md-8">
                            <label class="form-label fw-semibold text-primary">
                                <i class="fas fa-user me-2"></i> Select Your Profile
                            </label>
                            <select class="form-select form-select-lg" name="user_id">
                                {% for u in users_list %}
                                <option value="{{ u.id }}" {% if u.id == request.form.user_id or (not request.form.user_id and u.id == 'U001') %}selected{% endif %}>
                                    {{ u.id }} • {{ u.summary }}
                                </option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-4">
                            <button type="submit" class="btn btn-primary w-100 btn-lg">
                                <i class="fas fa-magic me-2"></i> Find My Matches
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        {% if before is not none %}
        <div class="results-container">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2 class="fw-bold">Your Personalized Recommendations</h2>
                <a href="/" class="btn btn-outline-secondary btn-sm"><i class="fas fa-sync-alt me-1"></i> Try Another Profile</a>
            </div>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="card h-100">
                        <div class="card-header bg-light d-flex justify-content-between py-3">
                            <span class="badge bg-secondary fs-6">BEFORE TRAINING</span>
                            <div>Accuracy: <span class="score-badge text-primary">{{ acc_before }}%</span></div>
                        </div>
                        <div class="card-body">
                            {% for m in before %}
                            <div class="match-card card mb-3 p-3">
                                <div class="d-flex justify-content-between">
                                    <div><h5>{{ m[0] }}</h5></div>
                                    <div class="text-end"><div class="score-badge text-primary">{{ m[1] }}%</div></div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card h-100 border-success">
                        <div class="card-header bg-success text-white d-flex justify-content-between py-3">
                            <span class="badge bg-white text-success fs-6">AFTER TRAINING</span>
                            <div>Accuracy: <span class="score-badge text-white">{{ acc_after }}%</span></div>
                        </div>
                        <div class="card-body">
                            {% for m in after %}
                            <div class="match-card card mb-3 p-3">
                                <div class="d-flex justify-content-between">
                                    <div><h5>{{ m[0] }}</h5></div>
                                    <div class="text-end"><div class="score-badge text-success">{{ m[1] }}%</div></div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mt-5">
                <div class="card-body text-center">
                    <img src="data:image/png;base64,{{ graph }}" class="img-fluid rounded" style="max-height: 320px;">
                    <p class="text-muted mt-3">Adaptive Feedback Loop Improvement</p>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


# ============================
# MAIN ROUTE
# ============================

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_id = request.form["user_id"]

        acc_before = evaluate_accuracy()
        before = get_recommendations(user_id)

        train_weights()

        acc_after = evaluate_accuracy()
        after = get_recommendations(user_id)

        # Graph generation
        img = io.BytesIO()
        plt.figure(figsize=(8, 5))
        plt.bar(["Before Training", "After Training"], [acc_before, acc_after], color=["#0d6efd", "#198754"])
        plt.ylabel("Accuracy (%)")
        plt.title("Model Improvement After Feedback Training")
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()

        img.seek(0)
        graph = base64.b64encode(img.getvalue()).decode()

        return render_template_string(
            HTML,
            before=before,
            after=after,
            acc_before=round(acc_before, 1),
            acc_after=round(acc_after, 1),
            graph=graph,
            users_list=users_list
        )

    # GET request - show clean form
    return render_template_string(
        HTML,
        before=None,
        after=None,
        acc_before=None,
        acc_after=None,
        graph=None,
        users_list=users_list
    )


# ============================
# RUN APP
# ============================

if __name__ == "__main__":
    app.run(debug=True)
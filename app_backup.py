import os
import sqlite3
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from utils.prediction import predict_image
from utils.report import create_pdf_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "pathovision.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "gradcam")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("PATHOVISION_SECRET", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        medical_id TEXT NOT NULL,
        institution TEXT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        case_ref TEXT NOT NULL,
        patient_ref TEXT,
        specimen TEXT,
        original_filename TEXT NOT NULL,
        saved_filename TEXT NOT NULL,
        heatmap_filename TEXT,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        report_filename TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    conn.commit()
    conn.close()

init_db()

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}

@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        medical_id = request.form.get("medical_id", "").strip()
        institution = request.form.get("institution", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not all([name, medical_id, email, password]):
            flash("Please fill all required fields.", "danger")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "danger")
            return render_template("register.html")

        conn = db()
        try:
            conn.execute(
                "INSERT INTO users(name, medical_id, institution, email, password_hash, created_at) VALUES(?,?,?,?,?,?)",
                (name, medical_id, institution, email, generate_password_hash(password), datetime.now().isoformat(timespec="seconds"))
            )
            conn.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with that email already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    conn = db()
    stats = conn.execute("""
        SELECT
          COUNT(*) AS total,
          COALESCE(SUM(CASE WHEN prediction='Positive' THEN 1 ELSE 0 END),0) AS positive,
          COALESCE(SUM(CASE WHEN prediction='Negative' THEN 1 ELSE 0 END),0) AS negative,
          COALESCE(AVG(confidence),0) AS avg_confidence
        FROM cases WHERE user_id=?
    """, (session["user_id"],)).fetchone()
    recent = conn.execute(
        "SELECT * FROM cases WHERE user_id=? ORDER BY id DESC LIMIT 6",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", stats=stats, recent=recent)

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        case_ref = request.form.get("case_ref", "").strip() or f"PV-{uuid.uuid4().hex[:8].upper()}"
        patient_ref = request.form.get("patient_ref", "").strip()
        specimen = request.form.get("specimen", "").strip()
        image = request.files.get("image")

        if not image or not image.filename:
            flash("Please select a pathology image.", "danger")
            return render_template("analyze.html")
        if not allowed_file(image.filename):
            flash("Allowed formats: PNG, JPG, JPEG, TIF, TIFF.", "danger")
            return render_template("analyze.html")

        ext = image.filename.rsplit(".", 1)[1].lower()
        safe_original = secure_filename(image.filename)
        saved_filename = f"{uuid.uuid4().hex}.{ext}"
        image_path = os.path.join(UPLOAD_DIR, saved_filename)
        image.save(image_path)

        try:
            result = predict_image(image_path, HEATMAP_DIR, saved_filename)
        except Exception as exc:
            if os.path.exists(image_path):
                os.remove(image_path)
            flash(f"AI analysis failed: {exc}", "danger")
            return render_template("analyze.html")

        heatmap_filename = result.get("heatmap_filename")

        conn = db()
        cur = conn.execute("""
            INSERT INTO cases(
                user_id, case_ref, patient_ref, specimen, original_filename,
                saved_filename, heatmap_filename, prediction, confidence, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            session["user_id"], case_ref, patient_ref, specimen, safe_original,
            saved_filename, heatmap_filename, result["prediction"],
            result["confidence"], datetime.now().isoformat(timespec="seconds")
        ))
        case_id = cur.lastrowid
        conn.commit()
        conn.close()

        return redirect(url_for("result", case_id=case_id))

    return render_template("analyze.html")

@app.route("/result/<int:case_id>")
@login_required
def result(case_id):
    conn = db()
    case = conn.execute(
        "SELECT * FROM cases WHERE id=? AND user_id=?",
        (case_id, session["user_id"])
    ).fetchone()
    conn.close()
    if not case:
        flash("Case not found.", "danger")
        return redirect(url_for("dashboard"))
    return render_template("result.html", case=case)

@app.route("/history")
@login_required
def history():
    conn = db()
    cases = conn.execute(
        "SELECT * FROM cases WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("history.html", cases=cases)

@app.route("/report/<int:case_id>")
@login_required
def report(case_id):
    conn = db()
    case = conn.execute(
        "SELECT cases.*, users.name AS pathologist, users.medical_id, users.institution "
        "FROM cases JOIN users ON cases.user_id=users.id WHERE cases.id=? AND cases.user_id=?",
        (case_id, session["user_id"])
    ).fetchone()
    conn.close()

    if not case:
        flash("Case not found.", "danger")
        return redirect(url_for("dashboard"))

    pdf_name = case["report_filename"]
    if not pdf_name:
        pdf_name = f"PathoVision_Report_{case_id}.pdf"
        pdf_path = os.path.join(REPORT_DIR, pdf_name)

        original_path = os.path.join(UPLOAD_DIR, case["saved_filename"])
        heatmap_path = os.path.join(HEATMAP_DIR, case["heatmap_filename"]) if case["heatmap_filename"] else None

        create_pdf_report(
            pdf_path=pdf_path,
            case=dict(case),
            original_path=original_path,
            heatmap_path=heatmap_path
        )

        conn = db()
        conn.execute("UPDATE cases SET report_filename=? WHERE id=? AND user_id=?",
                     (pdf_name, case_id, session["user_id"]))
        conn.commit()
        conn.close()
    else:
        pdf_path = os.path.join(REPORT_DIR, pdf_name)

    return send_file(pdf_path, as_attachment=True, download_name=pdf_name)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

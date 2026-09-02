import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from utils.preprocessing import preprocess_image
from utils.model import predict_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)

app.config["SECRET_KEY"] = "pathovision-final-year-project-secret-key"

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(BASE_DIR, "database.db")
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db = SQLAlchemy(app)


# ============================================================
# DATABASE MODEL
# ============================================================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(120), nullable=False)

    hospital = db.Column(db.String(200), nullable=False)

    registration_id = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(150), unique=True, nullable=False)

    password_hash = db.Column(db.String(255), nullable=False)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class Analysis(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    case_id = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )

    image_filename = db.Column(
        db.String(255),
        nullable=False
    )

    predicted_class = db.Column(
        db.String(100),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    probability = db.Column(
        db.Float,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def current_user():

    user_id = session.get("user_id")

    if not user_id:
        return None

    return db.session.get(User, user_id)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form.get("full_name", "").strip()
        hospital = request.form.get("hospital", "").strip()
        registration_id = request.form.get(
            "registration_id",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not all([
            full_name,
            hospital,
            registration_id,
            email,
            password,
            confirm_password
        ]):

            flash(
                "Please complete all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "An account with this email already exists.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        user = User(
            full_name=full_name,
            hospital=hospital,
            registration_id=registration_id,
            email=email,
            password_hash=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password_hash,
            password
        ):

            session["user_id"] = user.id

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    user = current_user()

    if not user:

        return redirect(
            url_for("login")
        )

    analyses = Analysis.query.filter_by(
        user_id=user.id
    ).order_by(
        Analysis.created_at.desc()
    ).all()

    total_cases = len(analyses)

    positive_cases = sum(
        1
        for a in analyses
        if a.predicted_class.lower()
        not in ["normal", "negative"]
    )

    negative_cases = (
        total_cases - positive_cases
    )

    return render_template(
        "dashboard.html",
        user=user,
        analyses=analyses,
        total_cases=total_cases,
        positive_cases=positive_cases,
        negative_cases=negative_cases
    )


# ============================================================
# UPLOAD / ANALYSIS
# ============================================================

@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    user = current_user()

    if not user:

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        if "image" not in request.files:

            flash(
                "Please select an image.",
                "error"
            )

            return redirect(
                url_for("analyze")
            )

        file = request.files["image"]

        if file.filename == "":

            flash(
                "No image selected.",
                "error"
            )

            return redirect(
                url_for("analyze")
            )

        if not allowed_file(file.filename):

            flash(
                "Unsupported image format.",
                "error"
            )

            return redirect(
                url_for("analyze")
            )

        original_name = secure_filename(
            file.filename
        )

        extension = original_name.rsplit(
            ".",
            1
        )[1].lower()

        unique_filename = (
            str(uuid.uuid4())
            + "."
            + extension
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )

        file.save(image_path)

        try:

            image_array = preprocess_image(
                image_path
            )

            prediction = predict_image(
                image_array
            )

        except Exception as e:

            app.logger.exception(
                "Prediction failed"
            )

            flash(
                f"Analysis failed: {str(e)}",
                "error"
            )

            return redirect(
                url_for("analyze")
            )

        case_id = (
            "PV-"
            + datetime.now().strftime("%Y%m%d")
            + "-"
            + uuid.uuid4().hex[:6].upper()
        )

        analysis = Analysis(

            case_id=case_id,

            image_filename=unique_filename,

            predicted_class=prediction[
                "predicted_class"
            ],

            confidence=prediction[
                "confidence"
            ],

            probability=prediction[
                "probability"
            ],

            user_id=user.id
        )

        db.session.add(analysis)
        db.session.commit()

        return render_template(
            "result.html",
            user=user,
            analysis=analysis
        )

    return render_template(
        "upload.html",
        user=user
    )


# ============================================================
# CASE HISTORY
# ============================================================

@app.route("/history")
def history():

    user = current_user()

    if not user:

        return redirect(
            url_for("login")
        )

    analyses = Analysis.query.filter_by(
        user_id=user.id
    ).order_by(
        Analysis.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        user=user,
        analyses=analyses
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    user = current_user()

    if not user:

        return redirect(
            url_for("login")
        )

    return render_template(
        "profile.html",
        user=user
    )


# ============================================================
# CREATE DATABASE
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PATHOVISION AI")
    print("AI-ASSISTED DIGITAL PATHOLOGY")
    print("=" * 60)
    print()
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

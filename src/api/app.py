"""
Flask REST API for the Interview Trainer Agent.
Exposes endpoints consumed by the web UI.
"""

import os
import logging
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.agent.trainer_agent import agent
from src.parsers.resume_parser import resume_parser

logger = logging.getLogger(__name__)

# Resolve absolute paths relative to this file so they work regardless of cwd
_HERE = Path(__file__).resolve().parent          # src/api/
_ROOT = _HERE.parent.parent                      # project root
_TEMPLATES = str(_ROOT / "templates")
_STATIC    = str(_ROOT / "static")

UPLOAD_FOLDER = _ROOT / "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}


def create_app() -> Flask:
    app = Flask(__name__, static_folder=_STATIC, template_folder=_TEMPLATES)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    CORS(app)

    # ------------------------------------------------------------------ helpers
    def _allowed(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def _err(msg: str, code: int = 400):
        return jsonify({"error": msg}), code

    # ------------------------------------------------------------------ routes

    @app.route("/")
    def index():
        return send_from_directory(_TEMPLATES, "index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "service": "Interview Trainer Agent"})

    # ── /api/prepare/profile ───────────────────────────────────────────────────
    @app.route("/api/prepare/profile", methods=["POST"])
    def prepare_from_profile():
        """
        Prepare an interview session from manual profile input.
        Body JSON:
          name, job_title, experience_level, skills (list), years_of_experience, num_questions
        """
        data = request.get_json(force=True)
        if not data:
            return _err("Request body must be JSON.")
        required = ["name", "job_title", "experience_level"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return _err(f"Missing required fields: {', '.join(missing)}")

        try:
            session = agent.prepare_from_profile(
                name=data["name"],
                job_title=data["job_title"],
                experience_level=data["experience_level"],
                skills=data.get("skills", []),
                years_of_experience=int(data.get("years_of_experience", 0)),
                num_questions=data.get("num_questions"),
            )
            return jsonify(session.to_dict())
        except Exception as exc:
            logger.error("prepare_from_profile error: %s", exc)
            return _err(str(exc), 500)

    # ── /api/prepare/resume ────────────────────────────────────────────────────
    @app.route("/api/prepare/resume", methods=["POST"])
    def prepare_from_resume():
        """
        Upload a resume file (multipart/form-data).
        Optional form fields: job_title, num_questions
        """
        if "resume" not in request.files:
            return _err("No resume file provided.")
        file = request.files["resume"]
        if not file.filename or not _allowed(file.filename):
            return _err("Invalid file. Allowed types: pdf, docx, doc, txt")

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = UPLOAD_FOLDER / filename
        file.save(str(file_path))

        job_title = request.form.get("job_title")
        num_questions = request.form.get("num_questions")
        if num_questions:
            num_questions = int(num_questions)

        try:
            session = agent.prepare_from_resume(
                file_path=str(file_path),
                job_title=job_title,
                num_questions=num_questions,
            )
            return jsonify(session.to_dict())
        except Exception as exc:
            logger.error("prepare_from_resume error: %s", exc)
            return _err(str(exc), 500)
        finally:
            file_path.unlink(missing_ok=True)

    # ── /api/prepare/text ─────────────────────────────────────────────────────
    @app.route("/api/prepare/text", methods=["POST"])
    def prepare_from_text():
        """
        Submit resume as raw text.
        Body JSON: resume_text, job_title (optional), num_questions (optional)
        """
        data = request.get_json(force=True)
        if not data or not data.get("resume_text"):
            return _err("Field 'resume_text' is required.")
        try:
            session = agent.prepare_from_text(
                resume_text=data["resume_text"],
                job_title=data.get("job_title"),
                num_questions=data.get("num_questions"),
            )
            return jsonify(session.to_dict())
        except Exception as exc:
            logger.error("prepare_from_text error: %s", exc)
            return _err(str(exc), 500)

    # ── /api/evaluate ─────────────────────────────────────────────────────────
    @app.route("/api/evaluate", methods=["POST"])
    def evaluate_answer():
        """
        Evaluate a candidate's answer.
        Body JSON: question, user_answer, profile (optional dict)
        """
        data = request.get_json(force=True)
        if not data or not data.get("question") or not data.get("user_answer"):
            return _err("Fields 'question' and 'user_answer' are required.")

        profile = None
        if data.get("profile"):
            pd = data["profile"]
            profile = resume_parser.parse_manual(
                name=pd.get("name", ""),
                job_title=pd.get("job_title", ""),
                experience_level=pd.get("experience_level", "mid"),
                skills=pd.get("skills", []),
                years=pd.get("years_of_experience", 0),
            )
        try:
            result = agent.evaluate_answer(data["question"], data["user_answer"], profile)
            return jsonify(result)
        except Exception as exc:
            logger.error("evaluate_answer error: %s", exc)
            return _err(str(exc), 500)

    # ── /api/knowledge ────────────────────────────────────────────────────────
    @app.route("/api/knowledge", methods=["POST"])
    def add_knowledge():
        """Add custom text to the RAG knowledge base."""
        data = request.get_json(force=True)
        if not data or not data.get("text"):
            return _err("Field 'text' is required.")
        count = agent.add_custom_knowledge(data["text"], data.get("metadata"))
        return jsonify({"chunks_added": count})

    return app

"""
Resume and job profile parser.
Extracts structured information (name, skills, experience, education) from:
  - PDF files
  - DOCX files
  - Plain text
  - Raw job title / experience level strings (no file)
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    job_title: str = ""
    experience_level: str = "mid"          # fresher | junior | mid | senior | lead
    years_of_experience: int = 0
    skills: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    work_history: List[Dict[str, str]] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "job_title": self.job_title,
            "experience_level": self.experience_level,
            "years_of_experience": self.years_of_experience,
            "skills": self.skills,
            "education": self.education,
            "certifications": self.certifications,
            "work_history": self.work_history,
        }

    def summary(self) -> str:
        parts = []
        if self.name:
            parts.append(f"Candidate: {self.name}")
        if self.job_title:
            parts.append(f"Target Role: {self.job_title}")
        parts.append(f"Experience Level: {self.experience_level.title()} ({self.years_of_experience} years)")
        if self.skills:
            parts.append(f"Key Skills: {', '.join(self.skills[:10])}")
        if self.education:
            parts.append(f"Education: {'; '.join(self.education[:2])}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ResumeParser:
    """Parse resumes and extract structured profile data."""

    # Common technical skill keywords (expandable)
    TECH_SKILLS = {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "fastapi",
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform",
        "machine learning", "deep learning", "nlp", "computer vision", "ai", "ml",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
        "git", "ci/cd", "agile", "scrum", "devops", "microservices", "rest api",
        "linux", "bash", "powershell", "ansible", "jenkins", "kafka",
        "data science", "data engineering", "etl", "spark", "hadoop",
        "html", "css", "graphql", "grpc", "rabbitmq", "celery",
    }

    SOFT_SKILLS = {
        "leadership", "communication", "teamwork", "problem-solving", "critical thinking",
        "time management", "adaptability", "creativity", "collaboration", "mentoring",
        "presentation", "negotiation", "decision making", "conflict resolution",
    }

    EXPERIENCE_PATTERNS = re.compile(
        r"(\d+)\s*\+?\s*(?:years?|yrs?)[\s\w]*experience", re.IGNORECASE
    )
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    PHONE_PATTERN = re.compile(r"[\+\(]?[\d\s\-\(\)]{10,15}")

    def parse_file(self, file_path: str) -> UserProfile:
        """Parse a resume file (PDF or DOCX) and return a UserProfile."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = self._read_docx(path)
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return self.parse_text(text)

    def parse_text(self, text: str) -> UserProfile:
        """Parse raw resume text and return a UserProfile."""
        profile = UserProfile(raw_text=text)
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.years_of_experience = self._extract_years(text)
        profile.experience_level = self._classify_level(profile.years_of_experience)
        profile.skills = self._extract_skills(text)
        profile.education = self._extract_education(text)
        profile.certifications = self._extract_certifications(text)
        profile.name = self._extract_name(text)
        return profile

    def parse_manual(
        self,
        name: str,
        job_title: str,
        experience_level: str,
        skills: Optional[List[str]] = None,
        years: int = 0,
    ) -> UserProfile:
        """Build a UserProfile from manually entered data (no file upload)."""
        level = experience_level.lower().strip()
        if level not in ("fresher", "junior", "mid", "senior", "lead"):
            level = self._classify_level(years)
        return UserProfile(
            name=name,
            job_title=job_title,
            experience_level=level,
            years_of_experience=years,
            skills=skills or [],
        )

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    def _read_pdf(self, path: Path) -> str:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber not installed; trying PyPDF2")
        try:
            import PyPDF2
            text_parts = []
            with open(str(path), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc)
            return ""

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as exc:
            logger.error("DOCX extraction failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------

    def _extract_email(self, text: str) -> str:
        match = self.EMAIL_PATTERN.search(text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = self.PHONE_PATTERN.search(text)
        return match.group(0).strip() if match else ""

    def _extract_years(self, text: str) -> int:
        matches = self.EXPERIENCE_PATTERNS.findall(text)
        return max((int(m) for m in matches), default=0)

    def _classify_level(self, years: int) -> str:
        if years == 0:
            return "fresher"
        if years < 2:
            return "junior"
        if years < 5:
            return "mid"
        if years < 10:
            return "senior"
        return "lead"

    def _extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for skill in self.TECH_SKILLS | self.SOFT_SKILLS:
            if skill in text_lower:
                found.append(skill)
        return sorted(set(found))

    def _extract_education(self, text: str) -> List[str]:
        degrees = re.findall(
            r"(b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc|phd|bachelor|master|"
            r"mba|bca|mca|diploma)[^\n,;]*",
            text, re.IGNORECASE
        )
        return [d.strip() for d in degrees[:5]]

    def _extract_certifications(self, text: str) -> List[str]:
        certs = re.findall(
            r"(aws certified|azure certified|gcp certified|ibm certified|"
            r"cka|ckad|pmp|scrum master|cissp|ccna|ccnp|oracle certified)[^\n,;]*",
            text, re.IGNORECASE
        )
        return [c.strip() for c in certs[:5]]

    def _extract_name(self, text: str) -> str:
        """Heuristic: first non-empty line that looks like a name."""
        for line in text.splitlines():
            line = line.strip()
            if 2 <= len(line.split()) <= 4 and re.match(r"^[A-Za-z\s\.\-]+$", line):
                return line
        return ""


# Singleton
resume_parser = ResumeParser()

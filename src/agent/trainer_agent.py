"""
Main Interview Trainer Agent orchestrator.
Ties together profile parsing, RAG retrieval, and Granite generation
into a single, clean public API used by both the CLI and Flask web server.
"""

import logging
from typing import Optional, Dict, Any, List

from src.parsers.resume_parser import resume_parser, UserProfile
from src.agent.interview_generator import interview_generator, InterviewSession
from src.rag.pipeline import rag_pipeline
from src.config import config

logger = logging.getLogger(__name__)


class InterviewTrainerAgent:
    """
    Top-level agent.

    Usage:
        agent = InterviewTrainerAgent()

        # From resume file
        session = agent.prepare_from_resume("resume.pdf", job_title="Data Scientist")

        # From manual input
        session = agent.prepare_from_profile(
            name="Alice", job_title="Backend Engineer",
            experience_level="senior", skills=["python", "kafka"]
        )

        # Evaluate a user's answer
        feedback = agent.evaluate_answer(question, user_answer, profile)
    """

    def __init__(self):
        # Ensure knowledge base is loaded into the vector store on first run
        self._kb_loaded = False

    def _ensure_kb(self):
        if not self._kb_loaded:
            count = rag_pipeline.collection_count()
            if count == 0:
                logger.info("Vector store empty — ingesting built-in knowledge base.")
                rag_pipeline.ingest_knowledge_base()
            else:
                logger.info("Vector store already contains %d chunks.", count)
            self._kb_loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_from_resume(
        self,
        file_path: str,
        job_title: Optional[str] = None,
        num_questions: Optional[int] = None,
    ) -> InterviewSession:
        """Parse a resume file and generate a tailored interview session."""
        self._ensure_kb()
        profile = resume_parser.parse_file(file_path)
        if job_title:
            profile.job_title = job_title
        logger.info("Parsed resume: %s", profile.summary())
        return interview_generator.generate_session(profile, num_questions)

    def prepare_from_text(
        self,
        resume_text: str,
        job_title: Optional[str] = None,
        num_questions: Optional[int] = None,
    ) -> InterviewSession:
        """Parse raw resume text and generate a tailored interview session."""
        self._ensure_kb()
        profile = resume_parser.parse_text(resume_text)
        if job_title:
            profile.job_title = job_title
        return interview_generator.generate_session(profile, num_questions)

    def prepare_from_profile(
        self,
        name: str,
        job_title: str,
        experience_level: str,
        skills: Optional[List[str]] = None,
        years_of_experience: int = 0,
        num_questions: Optional[int] = None,
    ) -> InterviewSession:
        """Generate a session purely from manually entered profile data."""
        self._ensure_kb()
        profile = resume_parser.parse_manual(
            name=name,
            job_title=job_title,
            experience_level=experience_level,
            skills=skills or [],
            years=years_of_experience,
        )
        return interview_generator.generate_session(profile, num_questions)

    def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        profile: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate's spoken/written answer against the expected model response.

        Returns:
            dict with keys: score (1-10), strengths, weaknesses, improved_answer, tips
        """
        from src.granite_client import granite

        context = rag_pipeline.retrieve_as_context(question, k=3)
        profile_str = profile.summary() if profile else "Not provided"

        system_prompt = (
            "You are an expert interview evaluator. Assess the candidate's answer "
            "objectively and provide actionable, specific feedback. Return JSON only."
        )
        user_message = f"""Evaluate this interview answer:

QUESTION: {question}

CANDIDATE'S ANSWER: {user_answer}

CANDIDATE PROFILE:
{profile_str}

RELEVANT CONTEXT:
{context[:1000] if context else 'Use expert knowledge.'}

Return JSON with:
- "score": integer 1-10
- "strengths": array of 2-3 strength strings
- "weaknesses": array of 2-3 weakness strings  
- "improved_answer": a better version of the answer (100-150 words)
- "tips": array of 3 specific improvement tips

JSON only:"""

        try:
            from src.granite_client import granite
            import json, re
            raw = granite.generate_with_context(system_prompt, user_message)
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
        except Exception as exc:
            logger.error("Answer evaluation failed: %s", exc)

        return {
            "score": 0,
            "strengths": [],
            "weaknesses": ["Evaluation service unavailable."],
            "improved_answer": "",
            "tips": ["Please retry when the IBM Granite service is available."],
        }

    def add_custom_knowledge(self, text: str, metadata: Optional[Dict[str, str]] = None) -> int:
        """Add custom text (job description, company notes, etc.) to the RAG knowledge base."""
        return rag_pipeline.ingest_text(text, metadata)


# Singleton agent
agent = InterviewTrainerAgent()

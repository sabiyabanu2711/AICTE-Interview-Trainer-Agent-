"""
Interview question generation engine.
Combines RAG retrieval with IBM Granite to produce tailored question sets,
model answers, and improvement tips.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.granite_client import granite
from src.rag.pipeline import rag_pipeline
from src.parsers.resume_parser import UserProfile
from src.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InterviewQuestion:
    question: str
    category: str          # technical | behavioral | situational | hr
    difficulty: str        # easy | medium | hard
    model_answer: str = ""
    follow_up: str = ""
    improvement_tips: str = ""
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "model_answer": self.model_answer,
            "follow_up": self.follow_up,
            "improvement_tips": self.improvement_tips,
            "keywords": self.keywords,
        }


@dataclass
class InterviewSession:
    profile: UserProfile
    questions: List[InterviewQuestion] = field(default_factory=list)
    preparation_strategy: str = ""
    industry_insights: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "preparation_strategy": self.preparation_strategy,
            "industry_insights": self.industry_insights,
            "questions": [q.to_dict() for q in self.questions],
            "total_questions": len(self.questions),
        }


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_GENERATE_QUESTIONS = """You are an expert interview coach with 15+ years of experience
in technical recruiting, HR, and career development. You specialize in creating comprehensive,
tailored interview question sets based on a candidate's profile and target role.

Your questions must be:
- Specific to the role and experience level provided
- Progressive in difficulty (easy → medium → hard)
- Cover both technical skills and soft/behavioral competencies
- Grounded in real industry expectations

Always return your response as a valid JSON array."""

SYSTEM_MODEL_ANSWER = """You are a senior interview coach providing model answers and guidance.
Your answers must be:
- Structured using the STAR method for behavioral questions (Situation, Task, Action, Result)
- Technically accurate and detailed for technical questions
- Concise (150-250 words per answer)
- Practical and honest

Also provide 2-3 specific improvement tips for the candidate."""

SYSTEM_PREP_STRATEGY = """You are a career coach creating personalized interview preparation strategies.
Based on the candidate's profile, provide:
1. A 7-day preparation plan
2. Key focus areas
3. Common pitfalls for this role/level
4. Confidence-building techniques
5. Questions the candidate should ask the interviewer"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class InterviewGenerator:
    """Orchestrates question generation, answer modeling, and prep strategies."""

    def generate_session(
        self,
        profile: UserProfile,
        num_questions: Optional[int] = None,
    ) -> InterviewSession:
        """
        Full pipeline: retrieve context → generate questions → model answers → prep strategy.
        """
        num_questions = num_questions or config.max_questions
        session = InterviewSession(profile=profile)

        # 1. Retrieve relevant knowledge base context
        query = f"{profile.job_title} {profile.experience_level} interview questions"
        context = rag_pipeline.retrieve_as_context(
            query,
            filter_metadata={"role": profile.job_title} if profile.job_title else None,
        )
        if not context:
            context = f"General {profile.job_title} interview best practices."

        # 2. Generate questions
        logger.info("Generating %d questions for %s (%s)", num_questions, profile.job_title, profile.experience_level)
        questions = self._generate_questions(profile, context, num_questions)
        session.questions = questions

        # 3. Attach model answers and tips
        for question in session.questions:
            self._enrich_question(question, profile)

        # 4. Preparation strategy
        session.preparation_strategy = self._generate_prep_strategy(profile, context)

        # 5. Industry insights
        session.industry_insights = self._generate_industry_insights(profile, context)

        return session

    # ------------------------------------------------------------------
    # Step 1: Question generation
    # ------------------------------------------------------------------

    def _generate_questions(
        self,
        profile: UserProfile,
        context: str,
        num_questions: int,
    ) -> List[InterviewQuestion]:
        """Call Granite to generate a structured list of interview questions."""

        user_message = f"""Generate exactly {num_questions} interview questions for the following candidate.

CANDIDATE PROFILE:
{profile.summary()}

RELEVANT INDUSTRY CONTEXT:
{context[:2000]}

Return a JSON array where each element has these exact keys:
- "question": the interview question string
- "category": one of "technical", "behavioral", "situational", "hr"
- "difficulty": one of "easy", "medium", "hard"
- "keywords": array of 2-4 key concept strings

Distribute questions: 40% technical, 30% behavioral, 20% situational, 10% HR.
Scale difficulty to the experience level: {profile.experience_level}.

JSON array only — no markdown, no explanation:"""

        try:
            raw = granite.generate_with_context(SYSTEM_GENERATE_QUESTIONS, user_message)
            return self._parse_questions_json(raw)
        except Exception as exc:
            logger.error("Question generation failed: %s", exc)
            return self._fallback_questions(profile, num_questions)

    def _parse_questions_json(self, raw: str) -> List[InterviewQuestion]:
        """Extract and parse a JSON array from the model response."""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        # Find the first '[' to last ']'
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in model response.")
        data = json.loads(raw[start:end + 1])
        questions = []
        for item in data:
            questions.append(InterviewQuestion(
                question=item.get("question", ""),
                category=item.get("category", "technical"),
                difficulty=item.get("difficulty", "medium"),
                keywords=item.get("keywords", []),
            ))
        return questions

    # ------------------------------------------------------------------
    # Step 2: Model answers + improvement tips
    # ------------------------------------------------------------------

    def _enrich_question(self, question: InterviewQuestion, profile: UserProfile):
        """Generate model answer, follow-up, and improvement tips for a question."""
        context = rag_pipeline.retrieve_as_context(question.question, k=3)
        user_message = f"""Provide a model answer for the following interview question.

CANDIDATE PROFILE:
{profile.summary()}

QUESTION: {question.question}
CATEGORY: {question.category}
DIFFICULTY: {question.difficulty}

RELEVANT CONTEXT:
{context[:1000] if context else 'Use your expert knowledge.'}

Return JSON with keys:
- "model_answer": detailed sample answer (150-250 words)
- "follow_up": one likely follow-up question the interviewer might ask
- "improvement_tips": 2-3 specific tips for this candidate to improve their answer

JSON only:"""

        try:
            raw = granite.generate_with_context(SYSTEM_MODEL_ANSWER, user_message)
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw[start:end + 1])
                question.model_answer = data.get("model_answer", "")
                question.follow_up = data.get("follow_up", "")
                question.improvement_tips = data.get("improvement_tips", "")
        except Exception as exc:
            logger.warning("Enrichment failed for question '%s': %s", question.question[:50], exc)

    # ------------------------------------------------------------------
    # Step 3: Preparation strategy
    # ------------------------------------------------------------------

    def _generate_prep_strategy(self, profile: UserProfile, context: str) -> str:
        user_message = f"""Create a personalized 7-day interview preparation strategy for:

{profile.summary()}

INDUSTRY CONTEXT:
{context[:1500]}

Include:
1. Day-by-day study plan
2. Key technical topics to revise
3. Behavioral question practice approach
4. Mock interview schedule
5. Confidence-building exercises
6. Questions the candidate should prepare to ask"""

        try:
            return granite.generate_with_context(SYSTEM_PREP_STRATEGY, user_message)
        except Exception as exc:
            logger.error("Prep strategy generation failed: %s", exc)
            return "Preparation strategy generation failed. Please retry."

    # ------------------------------------------------------------------
    # Step 4: Industry insights
    # ------------------------------------------------------------------

    def _generate_industry_insights(self, profile: UserProfile, context: str) -> str:
        prompt = (
            f"Provide 5 key industry expectations and hiring trends for a "
            f"{profile.experience_level}-level {profile.job_title} role in 2024-2025. "
            f"Context: {context[:800]}"
        )
        try:
            return granite.generate(prompt, instruct=True)
        except Exception as exc:
            logger.error("Industry insights failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Fallback questions (when Granite is unavailable)
    # ------------------------------------------------------------------

    def _fallback_questions(
        self, profile: UserProfile, num_questions: int
    ) -> List[InterviewQuestion]:
        """Return generic template questions as a graceful fallback."""
        templates = [
            ("Tell me about yourself and your background.", "hr", "easy"),
            (f"What interests you about the {profile.job_title} role?", "hr", "easy"),
            ("Describe a challenging project you worked on and how you overcame obstacles.", "behavioral", "medium"),
            ("How do you stay up-to-date with industry trends and new technologies?", "behavioral", "easy"),
            (f"What are your core technical strengths relevant to {profile.job_title}?", "technical", "easy"),
            ("Describe a situation where you had to work under pressure. How did you handle it?", "situational", "medium"),
            ("Walk me through your problem-solving approach for a complex technical challenge.", "technical", "medium"),
            ("How do you handle disagreements with team members or stakeholders?", "behavioral", "medium"),
            (f"What is your experience with the key tools used in {profile.job_title}?", "technical", "medium"),
            ("Where do you see yourself in 3-5 years?", "hr", "easy"),
        ]
        questions = []
        for i, (q, cat, diff) in enumerate(templates[:num_questions]):
            questions.append(InterviewQuestion(question=q, category=cat, difficulty=diff))
        return questions


# Singleton
interview_generator = InterviewGenerator()

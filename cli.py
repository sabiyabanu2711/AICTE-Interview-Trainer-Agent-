#!/usr/bin/env python3
"""
Interactive CLI for the AICTE Interview Trainer Agent.
Run:  python cli.py
"""

import sys
import os
import logging
import textwrap
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from dotenv import load_dotenv
load_dotenv()

from src.agent.trainer_agent import agent
from src.config import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING")),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Rich console helpers (degrade gracefully without rich)
# ---------------------------------------------------------------------------
if HAS_RICH:
    console = Console()

    def print_header():
        console.print(Panel.fit(
            "[bold cyan]🎯 AICTE Interview Trainer Agent[/bold cyan]\n"
            "[dim]Powered by IBM Granite (watsonx.ai) + RAG[/dim]",
            border_style="cyan",
        ))

    def print_success(msg: str):
        console.print(f"[bold green]✓[/bold green] {msg}")

    def print_info(msg: str):
        console.print(f"[cyan]ℹ[/cyan] {msg}")

    def print_error(msg: str):
        console.print(f"[bold red]✗[/bold red] {msg}")

    def print_question(num: int, q_obj) -> None:
        cat_colors = {"technical": "blue", "behavioral": "green",
                      "situational": "yellow", "hr": "magenta"}
        diff_colors = {"easy": "green", "medium": "yellow", "hard": "red"}
        color = cat_colors.get(q_obj.category, "white")
        diff_color = diff_colors.get(q_obj.difficulty, "white")
        console.print(
            f"\n[bold]{num}.[/bold] [{color}]{q_obj.category.upper()}[/{color}] "
            f"[{diff_color}]({q_obj.difficulty})[/{diff_color}]"
        )
        console.print(f"   [bold white]{q_obj.question}[/bold white]")
        if q_obj.keywords:
            console.print(f"   [dim]Keywords: {', '.join(q_obj.keywords)}[/dim]")

    def print_model_answer(q_obj) -> None:
        if q_obj.model_answer:
            console.print(Panel(
                q_obj.model_answer,
                title="[green]Model Answer[/green]",
                border_style="green",
            ))
        if q_obj.follow_up:
            console.print(f"  [yellow]Follow-up:[/yellow] {q_obj.follow_up}")
        if q_obj.improvement_tips:
            console.print(Panel(
                q_obj.improvement_tips,
                title="[cyan]Improvement Tips[/cyan]",
                border_style="cyan",
            ))

else:
    def print_header():
        print("=" * 60)
        print("  AICTE Interview Trainer Agent")
        print("  Powered by IBM Granite (watsonx.ai) + RAG")
        print("=" * 60)

    def print_success(msg): print(f"[OK] {msg}")
    def print_info(msg): print(f"[INFO] {msg}")
    def print_error(msg): print(f"[ERROR] {msg}")

    def print_question(num, q_obj):
        print(f"\n{num}. [{q_obj.category.upper()}] ({q_obj.difficulty})")
        print(f"   {q_obj.question}")
        if q_obj.keywords:
            print(f"   Keywords: {', '.join(q_obj.keywords)}")

    def print_model_answer(q_obj):
        if q_obj.model_answer:
            print("\n--- Model Answer ---")
            print(textwrap.fill(q_obj.model_answer, 80))
        if q_obj.follow_up:
            print(f"\nFollow-up: {q_obj.follow_up}")
        if q_obj.improvement_tips:
            print("\n--- Improvement Tips ---")
            print(q_obj.improvement_tips)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    if HAS_RICH:
        return Prompt.ask(prompt, default=default).strip()
    val = input(f"{prompt} [{default}]: ").strip()
    return val or default


def confirm(prompt: str, default: bool = True) -> bool:
    if HAS_RICH:
        return Confirm.ask(prompt, default=default)
    val = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


# ---------------------------------------------------------------------------
# Main menu flows
# ---------------------------------------------------------------------------
def flow_manual_profile() -> None:
    print_info("Enter your profile details:")
    name = ask("Your name", "Candidate")
    job_title = ask("Target job title", "Software Engineer")
    exp_level = ask("Experience level (fresher/junior/mid/senior/lead)", "mid")
    years = int(ask("Years of experience", "0"))
    skills_raw = ask("Key skills (comma-separated)", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
    num_q = int(ask("Number of questions to generate", "10"))

    if HAS_RICH:
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), transient=True) as progress:
            progress.add_task("Generating your personalised interview session with IBM Granite...", total=None)
            session = agent.prepare_from_profile(
                name=name, job_title=job_title, experience_level=exp_level,
                skills=skills, years_of_experience=years, num_questions=num_q,
            )
    else:
        print_info("Generating your personalised interview session with IBM Granite...")
        session = agent.prepare_from_profile(
            name=name, job_title=job_title, experience_level=exp_level,
            skills=skills, years_of_experience=years, num_questions=num_q,
        )

    _display_session(session)


def flow_resume_file() -> None:
    file_path = ask("Path to your resume file (PDF/DOCX/TXT)")
    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        return
    job_title = ask("Target job title (leave blank to auto-detect)")
    num_q = int(ask("Number of questions", "10"))

    if HAS_RICH:
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), transient=True) as progress:
            progress.add_task("Parsing resume and generating session...", total=None)
            session = agent.prepare_from_resume(
                file_path=file_path, job_title=job_title or None, num_questions=num_q
            )
    else:
        print_info("Parsing resume and generating session...")
        session = agent.prepare_from_resume(
            file_path=file_path, job_title=job_title or None, num_questions=num_q
        )

    _display_session(session)


def _display_session(session) -> None:
    """Render the full session to the terminal."""
    profile = session.profile

    print_success(f"Session ready for {profile.name or 'Candidate'} — {profile.job_title}")

    # Questions table
    if HAS_RICH:
        table = Table(title=f"Interview Questions — {profile.job_title}", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Category", style="cyan", width=14)
        table.add_column("Difficulty", width=10)
        table.add_column("Question")
        for i, q in enumerate(session.questions, 1):
            diff_map = {"easy": "[green]easy[/green]", "medium": "[yellow]medium[/yellow]", "hard": "[red]hard[/red]"}
            table.add_row(str(i), q.category, diff_map.get(q.difficulty, q.difficulty), q.question)
        console.print(table)
    else:
        print(f"\n=== Interview Questions for {profile.job_title} ===")
        for i, q in enumerate(session.questions, 1):
            print_question(i, q)

    # Prep strategy
    if session.preparation_strategy:
        if HAS_RICH:
            console.print(Panel(
                Markdown(session.preparation_strategy),
                title="[bold cyan]📋 7-Day Preparation Strategy[/bold cyan]",
                border_style="cyan", padding=(1, 2),
            ))
        else:
            print("\n=== 7-Day Preparation Strategy ===")
            print(session.preparation_strategy)

    # Industry insights
    if session.industry_insights:
        if HAS_RICH:
            console.print(Panel(
                session.industry_insights,
                title="[bold yellow]🏭 Industry Insights[/bold yellow]",
                border_style="yellow",
            ))
        else:
            print("\n=== Industry Insights ===")
            print(session.industry_insights)

    # Optional: show model answers
    if confirm("\nWould you like to see model answers for all questions?", default=False):
        for i, q in enumerate(session.questions, 1):
            print_question(i, q)
            print_model_answer(q)

    # Optional: practice mode
    if confirm("\nWould you like to enter practice mode (answer questions and get scored)?", default=False):
        _practice_mode(session)


def _practice_mode(session) -> None:
    """Let the user answer each question and get feedback."""
    print_info("Practice Mode — type your answer, then press Enter twice when done.\n")
    for i, q in enumerate(session.questions, 1):
        print_question(i, q)
        lines = []
        print("Your answer (blank line to submit):")
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        user_answer = " ".join(lines).strip()
        if not user_answer:
            print_info("Skipped.")
            continue

        if HAS_RICH:
            with Progress(SpinnerColumn(), TextColumn("[cyan]Evaluating your answer..."), transient=True) as progress:
                progress.add_task("", total=None)
                feedback = agent.evaluate_answer(q.question, user_answer, session.profile)
        else:
            print_info("Evaluating your answer...")
            feedback = agent.evaluate_answer(q.question, user_answer, session.profile)

        score = feedback.get("score", "N/A")
        if HAS_RICH:
            score_color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
            console.print(f"\n  Score: [{score_color}]{score}/10[/{score_color}]")
            if feedback.get("strengths"):
                console.print("  [green]Strengths:[/green] " + "; ".join(feedback["strengths"]))
            if feedback.get("weaknesses"):
                console.print("  [red]Weaknesses:[/red] " + "; ".join(feedback["weaknesses"]))
            if feedback.get("improved_answer"):
                console.print(Panel(feedback["improved_answer"], title="Improved Answer", border_style="green"))
            if feedback.get("tips"):
                console.print("  [cyan]Tips:[/cyan]\n" + "\n".join(f"  • {t}" for t in feedback["tips"]))
        else:
            print(f"\n  Score: {score}/10")
            print("  Strengths:", "; ".join(feedback.get("strengths", [])))
            print("  Weaknesses:", "; ".join(feedback.get("weaknesses", [])))
            if feedback.get("improved_answer"):
                print("  Improved Answer:", feedback["improved_answer"])
            print("  Tips:", "; ".join(feedback.get("tips", [])))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print_header()

    if not config.validate():
        print_error(
            "IBM Cloud credentials not configured.\n"
            "Please copy .env.example to .env and fill in your IBM Cloud API key and project ID."
        )
        print_info("Running in DEMO MODE with fallback questions (no IBM Granite).")

    choices = {
        "1": ("Enter profile manually", flow_manual_profile),
        "2": ("Upload / enter path to resume file", flow_resume_file),
        "3": ("Exit", None),
    }

    while True:
        if HAS_RICH:
            console.print("\n[bold]What would you like to do?[/bold]")
            for key, (desc, _) in choices.items():
                console.print(f"  [cyan]{key}[/cyan]. {desc}")
        else:
            print("\nWhat would you like to do?")
            for key, (desc, _) in choices.items():
                print(f"  {key}. {desc}")

        choice = ask("Enter choice", "1")
        if choice not in choices:
            print_error("Invalid choice.")
            continue

        desc, fn = choices[choice]
        if fn is None:
            print_success("Goodbye! Good luck with your interview! 🚀")
            break
        try:
            fn()
        except KeyboardInterrupt:
            print("\n")
        except Exception as exc:
            print_error(f"An error occurred: {exc}")
            logging.exception("CLI error")


if __name__ == "__main__":
    main()

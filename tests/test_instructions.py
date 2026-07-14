import json

from agent import (
    BASE_INSTRUCTIONS,
    build_greeting,
    build_instructions,
    format_candidate_profile,
    parse_summary,
)

SAMPLE_SKILL_FOCUS = (
    "\n# Interview focus\n\n"
    "This is a skill-specific practice interview on React. There is no resume to "
    "draw on — ground every question in the topic areas below.\n\n"
    "Skill: React\n"
    "- Hooks: useState/useEffect, useReducer, custom hooks"
)

SAMPLE_SUMMARY = {
    "name": "Aman Yadav",
    "role": "Full-Stack Engineer",
    "summary": "Full-Stack Engineer building scalable web apps.",
    "yearOfExp": "<1 year",
    "technicalSkills": [
        {"name": "React", "usedIn": ["SyncSides"]},
        {"name": "WebRTC"},
        {"name": None},
    ],
    "experience": [
        {
            "role": "Full-Stack Developer",
            "company": "Crewbella",
            "duration": "Oct 2025 - Present",
            "work": ["Added Redis caching and Sentry monitoring."],
        }
    ],
    "projects": [
        {
            "name": "SyncSides",
            "skills": ["WebRTC", "Socket.io"],
            "about": ["Real-time P2P video collaboration platform."],
            "readmeSummary": ["Synchronized local recordings."],
        }
    ],
    "education": [
        {
            "qualification": "B.Tech CSE",
            "institution": "GEC Ajmer",
            "startingYear": "2024",
        }
    ],
}


def test_parse_summary_handles_double_encoded_json():
    # The backend stores summary as a JSON string, then nests it in the job metadata.
    raw = json.dumps(SAMPLE_SUMMARY)
    assert parse_summary(raw) == SAMPLE_SUMMARY


def test_parse_summary_handles_missing_or_invalid():
    assert parse_summary(None) is None
    assert parse_summary("") is None
    assert parse_summary("null") is None
    assert parse_summary("not json") is None
    assert parse_summary(SAMPLE_SUMMARY) == SAMPLE_SUMMARY


def test_format_profile_includes_resume_specifics():
    profile = format_candidate_profile(SAMPLE_SUMMARY)
    for expected in (
        "Aman Yadav",
        "React",
        "WebRTC",
        "Crewbella",
        "SyncSides",
        "B.Tech CSE",
    ):
        assert expected in profile
    # None skill names must not leak into the profile.
    assert "None" not in profile


def test_build_instructions_appends_profile_when_summary_present():
    instructions = build_instructions(SAMPLE_SUMMARY)
    assert BASE_INSTRUCTIONS in instructions
    assert "Candidate resume" in instructions
    assert "SyncSides" in instructions


def test_build_instructions_falls_back_without_summary():
    assert build_instructions(None) == BASE_INSTRUCTIONS
    assert build_instructions({}) == BASE_INSTRUCTIONS


def test_build_instructions_includes_target_role_and_seniority():
    instructions = build_instructions(None, "Backend Engineer", "senior")
    assert "Interview target" in instructions
    assert "Backend Engineer" in instructions
    # The raw enum is expanded into a human-readable seniority description.
    assert "Senior" in instructions


def test_build_instructions_handles_unknown_experience():
    instructions = build_instructions(None, "AI Engineer", "principal")
    assert "AI Engineer" in instructions
    assert "principal" in instructions


def test_build_instructions_combines_role_and_resume():
    instructions = build_instructions(SAMPLE_SUMMARY, "Full Stack Engineer", "mid")
    assert "Interview target" in instructions
    assert "Full Stack Engineer" in instructions
    assert "Candidate resume" in instructions
    assert "SyncSides" in instructions


def test_build_instructions_appends_skill_focus_for_practice():
    instructions = build_instructions(None, "React", "senior", SAMPLE_SKILL_FOCUS)
    assert BASE_INSTRUCTIONS in instructions
    assert "Interview focus" in instructions
    assert "Hooks" in instructions
    # Practice has no resume, so the resume profile block must not appear.
    assert "Candidate resume" not in instructions


def test_build_instructions_prefers_skill_focus_over_resume():
    # If a skill focus is present it grounds the interview instead of any resume.
    instructions = build_instructions(
        SAMPLE_SUMMARY, "React", "senior", SAMPLE_SKILL_FOCUS
    )
    assert "Interview focus" in instructions
    assert "Candidate resume" not in instructions
    assert "SyncSides" not in instructions


def test_build_greeting_resume_mode_mentions_resume():
    greeting = build_greeting(SAMPLE_SUMMARY, "Backend Engineer", is_practice=False)
    assert "Aman" in greeting
    assert "resume" in greeting.lower()


def test_build_greeting_practice_mode_mentions_skill_not_resume():
    greeting = build_greeting(None, "React", is_practice=True)
    assert "React" in greeting
    assert "resume" not in greeting.lower()

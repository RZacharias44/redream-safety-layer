from AI.prompts import (
    SCOPE_BOUNDARY_DE,
    SCOPE_BOUNDARY_EN,
    SYSTEM_PROMPT_TEMPLATES_DE,
    SYSTEM_PROMPT_TEMPLATES_EN,
)


def test_summary_prompts_require_grounded_non_cinematic_details():
    english = SYSTEM_PROMPT_TEMPLATES_EN["summary"]
    german = SYSTEM_PROMPT_TEMPLATES_DE["summary"]

    assert "Invent nothing" in english
    assert "Every place, person, action, emotion, and sensory detail must trace directly" in english
    assert "do not draw conclusions the user did not state" in english
    assert "Do not make the scene more vivid, warm, or complete" in english
    assert "leave it out or stay general" in english
    assert "3 to 5 sentences, faithful, using only details the user gave" in english
    assert "Erfinde nichts" in german
    assert "Jeder Ort, jede Person, Handlung, Emotion und jedes sinnliche Detail" in german
    assert "ziehe keine Schlüsse" in german
    assert "Mach die Szene nicht lebendiger, wärmer oder vollständiger" in german
    assert "lass es weg oder bleib allgemein" in german
    assert "3 bis 5 Sätze, getreu, nur mit Details, die der Nutzer genannt hat" in german


def test_summary_prompts_require_final_confirmation_line():
    english = SYSTEM_PROMPT_TEMPLATES_EN["summary"]
    german = SYSTEM_PROMPT_TEMPLATES_DE["summary"]

    assert "The final line of your response must be exactly: Are you satisfied with the generated summary?" in english
    assert "Die letzte Zeile deiner Antwort muss exakt lauten: Bist du mit der generierten Zusammenfassung zufrieden?" in german


def test_scope_boundary_prompts_define_chatbot_role():
    assert "You are solely a guide through this IRT self-help exercise" in SCOPE_BOUNDARY_EN
    assert "You are not a therapist, doctor, counselor, or diagnostician" in SCOPE_BOUNDARY_EN
    assert "Telefonseelsorge" in SCOPE_BOUNDARY_EN
    assert "Du bist ausschließlich ein Begleiter durch diese Selbsthilfeübung" in SCOPE_BOUNDARY_DE
    assert "Du bist kein Therapeut, kein Arzt, kein Berater und kein Diagnostiker" in SCOPE_BOUNDARY_DE
    assert "Telefonseelsorge" in SCOPE_BOUNDARY_DE

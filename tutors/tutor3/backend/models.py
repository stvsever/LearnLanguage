from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VocabularyItem(BaseModel):
    english: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    pronunciation: str = ""
    note: str = ""
    tags: List[str] = Field(default_factory=list)


class VocabularyPack(BaseModel):
    target_language: str
    concept: str
    difficulty: str
    items: List[VocabularyItem] = Field(..., min_length=1)


class TranslationBatch(BaseModel):
    translations: List[str]


class ScenarioQuestion(BaseModel):
    question: str = Field(..., min_length=8)
    choices: List[str] = Field(..., min_length=4, max_length=4)
    reasoning_note: str = Field(..., min_length=1)
    question_en: str = Field(..., min_length=8)
    choices_en: List[str] = Field(..., min_length=4, max_length=4)
    reasoning_note_en: str = Field(..., min_length=1)
    correct_choice: int = Field(..., ge=0, le=3)
    accepted_answers: List[str] = Field(..., min_length=1)

    @field_validator("choices", "choices_en")
    @classmethod
    def no_meta_answers(cls, value: List[str]) -> List[str]:
        banned = {
            "all of the above",
            "none of the above",
            "both a and b",
            "todas las anteriores",
            "ninguna de las anteriores",
            "toutes les réponses ci-dessus",
            "aucune des réponses ci-dessus",
            "все вышеперечисленное",
            "ничего из вышеперечисленного",
            "以上皆是",
            "以上都不是",
        }
        lowered = {item.strip().lower() for item in value}
        if lowered & banned:
            raise ValueError("Question choices must not use meta answers.")
        return value


class ScenarioPack(BaseModel):
    target_language: str
    scenario_title: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    passage: str = Field(..., min_length=20)
    passage_en_lines: List[str] = Field(..., min_length=1)
    items: List[ScenarioQuestion] = Field(..., min_length=5, max_length=5)


class ApiError(BaseModel):
    error: str
    detail: Optional[str] = None

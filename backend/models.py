"""Pydantic schemas for structured LLM content. Every generation endpoint
validates model output against these before anything reaches the client."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LessonItem(BaseModel):
    """One learnable unit: a word, chunk, or short sentence with rich encoding cues."""

    target: str = Field(..., min_length=1, description="The item in the target language.")
    english: str = Field(..., min_length=1, description="Natural English equivalent.")
    pronunciation: str = Field("", description="IPA (or pinyin with tone marks) for the item.")
    example: str = Field("", description="A short natural example sentence using the item.")
    example_en: str = Field("", description="English translation of the example sentence.")
    note: str = Field("", description="Optional grammar/usage/register note, one short sentence.")
    tags: List[str] = Field(default_factory=list)


class LessonPack(BaseModel):
    language: str = ""
    topic: str = ""
    level: str = ""
    items: List[LessonItem] = Field(..., min_length=1)
    grammar_features: List[str] = Field(
        default_factory=list,
        description="Ids from the provided grammar feature list that this lesson genuinely touches.",
    )


class Question(BaseModel):
    question: str = Field(..., min_length=4)
    choices: List[str] = Field(..., min_length=4, max_length=4)
    correct_choice: int = Field(..., ge=0, le=3)
    explanation: str = Field("", description="Why the correct choice is right, in English.")

    @field_validator("choices")
    @classmethod
    def no_meta_answers(cls, value: List[str]) -> List[str]:
        banned = {
            "all of the above", "none of the above",
            "toutes les réponses ci-dessus", "aucune des réponses ci-dessus",
            "todas las anteriores", "ninguna de las anteriores",
            "все вышеперечисленное", "以上都对",
        }
        if {c.strip().lower() for c in value} & banned:
            raise ValueError("Choices must not use meta answers.")
        return value


class GlossEntry(BaseModel):
    word: str = Field(..., min_length=1)
    gloss: str = Field(..., min_length=1, description="Short contextual English gloss.")


class Segment(BaseModel):
    """One unit of a composition: a sentence, or one turn of a dialogue."""

    speaker: str = Field("", description="Speaker name for dialogues; empty for prose.")
    text: str = Field(..., min_length=1, description="Target-language text (one sentence or one short turn).")
    text_en: str = Field(..., min_length=1, description="Faithful natural English translation.")


class GrammarSpotlight(BaseModel):
    """A grammar feature deliberately woven into the composition."""

    feature: str = Field(..., description="Feature id from the provided grammar feature list.")
    excerpt: str = Field(..., min_length=1, description="The exact phrase from the text showing it.")
    explanation: str = Field(..., min_length=1, description="One-sentence English explanation of the structure.")


CompositionFormat = Literal["dialogue", "monologue", "story", "article"]


class CompositionPack(BaseModel):
    """A generated piece of comprehensible input.

    The model both CLASSIFIES the right format from the user's free-form request
    and produces the content in one call.
    """

    language: str = ""
    format: CompositionFormat = Field(..., description="Chosen by you to best fit the user's request.")
    title: str = Field(..., min_length=1, description="Short title in the target language.")
    level: str = ""
    scene: str = Field("", description="One-line English description of the setting/premise.")
    participants: List[str] = Field(
        default_factory=list,
        description="Speaker names for dialogues (2-3), in order of first appearance. Empty otherwise.",
    )
    segments: List[Segment] = Field(..., min_length=1)
    glossary: List[GlossEntry] = Field(default_factory=list, description="8-14 tricky words/chunks with glosses.")
    grammar_spotlights: List[GrammarSpotlight] = Field(
        default_factory=list, description="2-4 target-level structures actually used in the text."
    )
    questions: List[Question] = Field(..., min_length=3, max_length=6)


class Gloss(BaseModel):
    """On-demand explanation of a selected word or phrase in context."""

    text: str = ""
    gloss: str = Field(..., min_length=1, description="Concise English meaning in this context.")
    lemma: str = Field("", description="Dictionary form if different.")
    pronunciation: str = Field("", description="IPA or pinyin.")
    note: str = Field("", description="One-sentence grammar or usage note.")


class ApiError(BaseModel):
    error: str
    detail: Optional[str] = None

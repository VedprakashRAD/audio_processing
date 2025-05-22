from pydantic import BaseModel
from typing import List, Optional

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    sentiment: Optional[str]
    emotion: Optional[str]

class AudioAnalysisResponse(BaseModel):
    language: str
    transcription: str
    segments: List[Segment]
    overall_sentiment: str
    speaker_stats: Optional[dict]
    summary: Optional[str]
    speaker_identification:Optional[dict]
    important_details:Optional[dict]



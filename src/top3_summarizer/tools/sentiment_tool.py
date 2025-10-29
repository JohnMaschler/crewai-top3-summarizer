from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type
from openai import OpenAI
import os
import json

class LLMSentimentInput(BaseModel):
    """Input schema for LLMSentimentTool."""
    text: str = Field(
        ...,
        description="Full article body text to analyze for market tone."
    )

class LLMSentimentTool(BaseTool):
    name: str = "llm_sentiment_tool"
    description: str = (
        "Classify the market tone of an article as 'bullish', 'bearish', or 'neutral' "
        "and explain why in one sentence. Use this when you need to score sentiment "
        "for executives or traders."
    )
    args_schema: Type[BaseModel] = LLMSentimentInput

    _client: OpenAI = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self._model = os.getenv("MODEL", "gpt-4o-mini")

    def _run(self, text: str) -> dict:
        """
        Returns dict:
        {
          "sentiment": "bullish" | "bearish" | "neutral",
          "rationale": "one-sentence explanation"
        }
        """

        prompt = (
            "You are a market sentiment rater for a bank.\n"
            "Read the article text below.\n"
            "Classify overall sentiment toward the market as exactly one of:\n"
            "  'bullish' (optimistic / risk-on),\n"
            "  'bearish' (pessimistic / risk-off), or\n"
            "  'neutral' (mixed / uncertain).\n"
            "Then give ONE short sentence explaining why.\n"
            "Respond ONLY in valid minified JSON with these keys:\n"
            '{"sentiment": "...", "rationale": "..."}\n\n'
            f"ARTICLE:\n{text[:4000]}\n"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw = resp.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except Exception:
            lowered = raw.lower()
            if "bullish" in lowered:
                sent = "bullish"
            elif "bearish" in lowered:
                sent = "bearish"
            elif "neutral" in lowered:
                sent = "neutral"
            else:
                sent = "neutral"

            return {
                "sentiment": sent,
                "rationale": raw
            }

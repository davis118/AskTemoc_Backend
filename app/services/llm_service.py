from langchain_ollama import OllamaLLM
from typing import Optional
import os
import random
from pathlib import Path
from app.core.config import get_settings

class LLMService:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        )

        self.base_url = base_url or settings.OLLAMA_BASE_URL

        self.llm = OllamaLLM(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    def call(self, query: Optional[str] = None) -> str:
        if not query:
            query = self._random_capital_prompt()

        response = self.llm.invoke(query)
        return response

    async def a_call(self, query: Optional[str] = None) -> str:
        if not query:
            query = self._random_capital_prompt()

        response = await self.llm.ainvoke(query)
        return response

    def get_llm(self) -> OllamaLLM:
        return self.llm

    @staticmethod
    def _random_capital_prompt() -> str:
        countries = ["Indonesia", "Germany", "China", "France", "Japan", "Brazil", "Russia", "Algeria", "Canada"]
        country = random.choice(countries)
        return f"What is the capital of {country}?"

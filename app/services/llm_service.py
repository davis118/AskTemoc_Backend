from langchain_ollama import OllamaLLM
from typing import Optional
import os
import random

class LLMService:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("OLLAMA_TEMPERATURE", 0.4))
        )
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

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

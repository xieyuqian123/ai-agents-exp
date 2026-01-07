import re
from typing import List
from travel_agent.llm_client import OpenAICompatibleClient
from PlanAndSolve.prompts import PLANNER_PROMPT

class Planner:
    def __init__(self, llm: OpenAICompatibleClient):
        self.llm = llm

    def plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT.format(question=question)
        response = self.llm.generate(prompt, system_prompt="You are a strategic planner.")
        print(f"\n[Planner Output]\n{response}\n")
        
        # Parse steps
        steps = []
        for line in response.strip().split('\n'):
            # Match "1. Step description"
            match = re.match(r'\d+\.\s*(.*)', line)
            if match:
                steps.append(match.group(1))
        return steps

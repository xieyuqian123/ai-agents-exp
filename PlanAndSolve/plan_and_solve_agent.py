import sys
import os
from typing import List, Optional, Callable
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.search import serpapi_search_text
from travel_agent.llm_client import OpenAICompatibleClient
from PlanAndSolve.planner import Planner
from PlanAndSolve.solver import Solver
from PlanAndSolve.prompts import FINAL_ANSWER_PROMPT

class PlanAndSolveAgent:
    def __init__(self, llm: OpenAICompatibleClient, tools: List[Callable], planner: Optional[Planner] = None, solver: Optional[Solver] = None):
        self.llm = llm
        self.planner = planner if planner else Planner(llm)
        self.solver = solver if solver else Solver(llm, tools)

    def run(self, question: str):
        # 1. Plan
        print(f"Original Question: {question}")
        steps = self.planner.plan(question)
        if not steps:
            print("Failed to generate a plan.")
            return

        # 2. Solve
        context = ""
        results = []
        
        for i, step in enumerate(steps):
            print(f"\n--- Executing Step {i+1}: {step} ---")
            result = self.solver.solve_step(step, context)
            print(f"Step Result: {result}")
            
            context += f"Step {i+1}: {step}\nResult: {result}\n\n"
            results.append(f"Step {i+1}: {step}\nResult: {result}")

        # 3. Synthesize Final Answer
        final_prompt = FINAL_ANSWER_PROMPT.format(
            question=question,
            execution_results="\n".join(results)
        )
        final_answer = self.llm.generate(final_prompt, system_prompt="You are a helpful assistant.")
        return final_answer

if __name__ == "__main__":
    load_dotenv()
    llm = OpenAICompatibleClient()
    
    def search(query: str):
        """Search the web for the given query."""
        return serpapi_search_text(query)

    agent = PlanAndSolveAgent(llm=llm, tools=[search])
    
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        
    final_res = agent.run(question)
    print(f"\n[Final Answer]\n{final_res}")

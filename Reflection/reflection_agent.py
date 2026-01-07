import sys
import os
from typing import List, Callable, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReAct.ReAct_agent import ReActAgent
from travel_agent.llm_client import OpenAICompatibleClient

REFLECTION_PROMPT = """
You are a strict critic.
Review the following User Question and the Agent's Answer.
Check for correctness, completeness, and clarity.
If the answer is incorrect or incomplete, provide specific feedback and suggestions for improvement.
If the answer is satisfactory, simply output "SATISFACTORY".

User Question: {question}
Agent Answer: {answer}

Critique:
"""

class ReflectionAgent:
    def __init__(self, llm: OpenAICompatibleClient, react_agent: ReActAgent):
        self.llm = llm
        self.react_agent = react_agent

    def reflect(self, question: str, answer: str) -> str:
        prompt = REFLECTION_PROMPT.format(question=question, answer=answer)
        response = self.llm.generate(prompt, system_prompt="You are a helpful critic.")
        return response

    def run(self, question: str, max_retries: int = 3) -> str:
        current_question = question
        history = ""

        for i in range(max_retries):
            print(f"\n=== Attempt {i+1} ===")
            
            # If we have history (previous attempts and critiques), append it to the question
            # or treat it as context. Here we append it to the question for the ReAct agent
            # to be aware of previous failures.
            if i > 0:
                input_to_agent = f"{question}\n\nPrevious Attempts and Critiques:\n{history}\n\nPlease try again, addressing the critiques."
            else:
                input_to_agent = question

            answer = self.react_agent.run(input_to_agent)
            print(f"\n[Agent Answer]\n{answer}")

            # Reflect
            critique = self.reflect(question, answer)
            print(f"\n[Critique]\n{critique}")

            if "SATISFACTORY" in critique.upper():
                print("\nAnswer deemed satisfactory.")
                return answer
            
            # Append to history
            history += f"Attempt {i+1} Answer: {answer}\nCritique: {critique}\n\n"

        return f"Final Answer (after {max_retries} retries): {answer}"

if __name__ == "__main__":
    from dotenv import load_dotenv
    from common.search import serpapi_search_text

    load_dotenv()
    llm = OpenAICompatibleClient()

    def search(query: str):
        """Search the web for the given query."""
        return serpapi_search_text(query)

    # Initialize ReAct Agent
    react_agent = ReActAgent(llm=llm, tools=[search])
    
    # Initialize Reflection Agent
    reflection_agent = ReflectionAgent(llm=llm, react_agent=react_agent)

    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        # A question that might require iteration or could be tricky
        question = "2024年图灵奖得主是谁？"

    final_result = reflection_agent.run(question)
    print(f"\n[Final Result]\n{final_result}")

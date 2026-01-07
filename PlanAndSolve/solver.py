import re
from typing import List, Callable
from common.available_tools import ToolExecutor
from travel_agent.llm_client import OpenAICompatibleClient
from PlanAndSolve.prompts import SOLVER_PROMPT

class Solver:
    def __init__(self, llm: OpenAICompatibleClient, tools: List[Callable]):
        self.llm = llm
        self.tools = tools
        tool_dict = {tool.__name__: tool for tool in tools}
        self.tool_executor = ToolExecutor(tool_dict)

    def _get_tool_descriptions(self) -> str:
        return "\n".join([f"{tool.__name__}: {tool.__doc__}" for tool in self.tools])

    def _get_tool_names(self) -> str:
        return ", ".join([tool.__name__ for tool in self.tools])

    def solve_step(self, step: str, context: str) -> str:
        tool_descriptions = self._get_tool_descriptions()
        tool_names = self._get_tool_names()
        
        prompt = SOLVER_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            current_step=step,
            previous_context=context
        )
        
        # Simple ReAct-like loop for a single step
        # For simplicity, we'll allow a few turns per step
        max_turns = 3
        history = ""
        
        for i in range(max_turns):
            current_input = prompt + history
            
            # Force a wrap-up in the last turn
            if i == max_turns - 1:
                current_input += "\nObservation: You have reached the maximum number of turns. Please provide the Final Answer now based on what you have found so far.\n"

            response = self.llm.generate(
                current_input, 
                system_prompt="You are a capable solver.",
                stop=["Observation:"]
            )
            
            if "Observation:" in response:
                response = response.split("Observation:")[0].strip()
                
            print(f"  [Solver Output]\n{response}")
            history += response + "\n"
            
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()
            
            # Parse Action
            action_match = re.search(r"Action:\s*(.*?)\n", response)
            action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response)
            
            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()
                
                print(f"  [Executing Tool] {action} with input: {action_input}")
                if self.tool_executor.has(action):
                    try:
                        observation = self.tool_executor.execute(action, query=action_input)
                        obs_str = f"Observation: {observation}\n"
                    except Exception as e:
                        obs_str = f"Observation: Error: {e}\n"
                else:
                    obs_str = f"Observation: Tool not found.\n"
                
                print(f"  {obs_str.strip()}")
                history += obs_str
            else:
                if "Thought:" not in response:
                     history += "Observation: Please provide Thought, Action, and Action Input, or Final Answer.\n"

        return "Step execution failed or incomplete."

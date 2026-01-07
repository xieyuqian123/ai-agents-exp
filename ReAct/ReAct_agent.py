import re
import sys
import os

# Add the project root to sys.path to allow importing from common and travel_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Optional, Callable
from common.available_tools import ToolExecutor
from travel_agent.llm_client import OpenAICompatibleClient

REACT_PROMPT_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
1. Always start your response with "Thought:".
2. Do not hallucinate the "Observation:" field. The observation will be provided to you.
3. Only produce one Action at a time.

Begin!

Question: {question}
"""

class ReActAgent:
    def __init__(self, llm: OpenAICompatibleClient, tools: List[Callable]):
        self.llm = llm
        # Create a dictionary of tools for the executor
        tool_dict = {tool.__name__: tool for tool in tools}
        self.tool_executor = ToolExecutor(tool_dict)
        self.tools = tools

    def _get_tool_descriptions(self) -> str:
        descriptions = []
        for tool in self.tools:
            # Use docstring or just name if no docstring
            desc = f"{tool.__name__}: {tool.__doc__.strip() if tool.__doc__ else 'No description available'}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _get_tool_names(self) -> str:
        return ", ".join([tool.__name__ for tool in self.tools])

    def _parse_response(self, response: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses the LLM response to extract Final Answer or Action/Action Input.
        Returns a tuple: (final_answer, action, action_input)
        """
        # Check for Final Answer
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer, None, None
        
        # Parse Action
        action_match = re.search(r"Action:\s*(.*?)\n", response)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return None, action, action_input
            
        return None, None, None

    def _execute_action(self, action: str, action_input: str) -> str:
        """
        Executes the parsed action.
        """
        if self.tool_executor.has(action):
            try:
                # Currently assuming tool takes 'query' argument, primarily for search
                # TODO: Improve argument parsing for more complex tools
                observation = self.tool_executor.execute(action, query=action_input)
                return f"Observation: {observation}\n"
            except Exception as e:
                return f"Observation: Error executing tool: {e}\n"
        else:
            return f"Observation: Tool '{action}' not found. Available tools: {self._get_tool_names()}\n"

    def run(self, question: str, max_turns: int = 5) -> str:
        tool_descriptions = self._get_tool_descriptions()
        tool_names = self._get_tool_names()
        
        prompt = REACT_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            question=question
        )
        
        history = []
        
        print(f"Question: {question}")

        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")
            
            # Construct the full input for the LLM
            current_input = prompt + "".join(history)
            
            # Call LLM
            response = self.llm.generate(
                current_input, 
                system_prompt="You are a helpful assistant that follows the ReAct pattern.",
                stream=False,
                stop=["Observation:"]
            )
            
            # Manual truncation
            if "Observation:" in response:
                response = response.split("Observation:")[0].strip()
            
            print(f"LLM Output:\n{response}")
            history.append(response + "\n")
            
            # Parse response
            final_answer, action, action_input = self._parse_response(response)
            
            if final_answer:
                return final_answer
            
            if action and action_input:
                print(f"Parsed Action: {action}")
                print(f"Parsed Input: {action_input}")
                
                observation_str = self._execute_action(action, action_input)
                print(observation_str.strip())
                history.append(observation_str)
            else:
                print("No action parsed.")
                if "Thought:" not in response:
                     history.append("Observation: Invalid format. Please provide 'Thought:', 'Action:', and 'Action Input:'.\n")

        return f"Agent stopped due to max turns ({max_turns}) without finding a final answer."

if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    from common.search import serpapi_search_text

    load_dotenv()
    
    # Initialize LLM
    # Ensure you have SILICONFLOW_API_KEY and SERPAPI_API_KEY in .env
    llm = OpenAICompatibleClient()
    
    # Define tools
    # We wrap search to have a nice name and docstring if needed, 
    # but serpapi_search_text already has a name.
    # Let's alias it for clarity in prompt
    def search(query: str):
        """Search the web for the given query."""
        return serpapi_search_text(query)

    agent = ReActAgent(llm=llm, tools=[search])
    
    # Use command line argument if provided, otherwise default
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "2025年销量最高新能源汽车是哪款?现在已经是2026年了。"
        
    result = agent.run(question)
    print(f"\nFinal Result: {result}")

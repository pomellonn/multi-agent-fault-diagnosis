# Agent orchestration API for IACoP.

from agents.llm_client import build_agent
import json

def send_task_to_agents(task_description: str) -> dict:
	# Instantiate agents
	planner = build_agent("planner")
	retriever = build_agent("retriever")
	reasoner = build_agent("reasoner")

	# Planner creates a plan
	plan = planner.generate_reply(messages=[{"role": "user", "content": task_description}])
   
	# Retriever queries knowledge for each step
	retrieved_knowledge = retriever.generate_reply(plan)

	# Reasoner produces final diagnosis
	reasoning_json = reasoner.generate_reply(f"Plan: {plan}\nKnowledge: {retrieved_knowledge}")
	try:
		result = json.loads(reasoning_json)

		for k in ("diagnosis", "confidence", "reasoning"):
			if k not in result:
				raise ValueError(f"Missing key: {k}")
		return result
	except Exception as e:
		return {"diagnosis": "error", "confidence": 0.0, "reasoning": f"Reasoner output parse error: {e}. Output: {reasoning_json}"}

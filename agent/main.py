import os
from dotenv import load_dotenv
import yaml
from autogen import ConversableAgent

load_dotenv()

with open("config/agents.yaml", "r") as f:
    config = yaml.safe_load(f)

llm_config = {
    "config_list": [
        {
            "model": "deepseek-chat",
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com",
            "api_type": "openai",
        }
    ],
    "temperature": 0.3,
}

planner = ConversableAgent(
    name=config["planner"]["name"],
    system_message=config["planner"]["system_message"],
    llm_config=llm_config,
    human_input_mode="NEVER",
)
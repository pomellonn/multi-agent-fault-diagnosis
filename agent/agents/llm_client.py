import os
import yaml
from autogen import ConversableAgent
from dotenv import load_dotenv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/agents.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "deepseek-chat",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.deepseek.com",
            "api_type": "openai",
        }
    ],
    "temperature": 0.3,
}

def build_agent(role: str) -> ConversableAgent:
    agent_cfg = config[role]
    return ConversableAgent(
        name=agent_cfg["name"],
        system_message=agent_cfg["system_message"],
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

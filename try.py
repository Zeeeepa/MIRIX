import os
from mirix import Mirix
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

mirix_agent = Mirix(
  api_key=API_KEY,
  model_provider="azure_opena",
  config_path=".local/mirix_jplml_azure.yaml",
)
alice = mirix_agent.create_user(user_name="John Doe")
mirix_agent.update_core_memory(text="My name is AAJohn Doe", label="persona", user_id=alice.id)



# mirix_agent.add("Remember my name is Alice.", user_id=alice.id)

# default_user = mirix_agent.list_users()[0]
# mirix_agent.add("Remember my name is Default", user_id=default_user.id)

# mirix_agent.add("Alice is Single", user_id=alice.id)
# mirix_agent.add("Default is Married", user_id=default_user.id)

# response = mirix_agent.chat("what is my name?", user_id=alice.id)
# print(response)

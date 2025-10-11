import os

# Load .env BEFORE importing mirix - this is critical!
# The database engine is initialized at module import time
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
mirix_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(mirix_root, ".env"))

# Now import mirix after environment variables are loaded
from mirix import Mirix

# Initialize LangChain and Mirix
API_KEY = os.getenv("GEMINI_API_KEY")

print(os.getenv("GEMINI_API_KEY"))
print(os.getenv("MIRIX_PG_URI"))

mirix_agent = Mirix(
    api_key=API_KEY,
    model_provider="gemini",
    config_path=f"{mirix_root}/mirix/configs/mirix_gemini.yaml",
)

###
# Sample memory query results:
#
#   'success': True,
#   'user_id': target_user.id,
#   'user_name': target_user.name,
#   'memories': memories,
#   'summary': {
#      'episodic_count': len(memories.get('episodic', [])),
#      'semantic_count': len(memories.get('semantic', [])),
#      'procedural_count': len(memories.get('procedural', [])),
#      'resources_count': len(memories.get('resources', [])),
#      'core_count': len(memories.get('core', [])),
#      'credentials_count': len(memories.get('credentials', []))
#    }
###

# Assume that we have tested with a user named "temp_test_user"
mirix_user = mirix_agent.create_user(user_name="alice")

# Option #1, extract the memory for the system prompt
system_prompt = mirix_agent.extract_memory_for_system_prompt(
    message="What is the current time?", user_id=mirix_user.id
)
print(f"Restrieved memory for system prompt: {system_prompt}")

# Option #2, visualize the Core and Episodic memories
results = mirix_agent.visualize_memories(user_id=mirix_user.id)

if results is not None:
    # First, print out the summary
    summary = results["summary"]

    print("\n\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Prepare to print our memories
    memories = results["memories"]

    # print out core memory
    core_memories = memories["core"]
    print("\n--- Core memory:---")

    for core_mem in core_memories:
        for k, v in core_mem.items():
            print(f"{k}: {v}")
        print("\n")

    # print out episodic memory
    episodic_memories = memories["episodic"]
    print("\n--- Episodic memory:---")

    for episodic_mem in episodic_memories:
        for k, v in episodic_mem.items():
            print(f"{k}: {v}")
        print("\n")

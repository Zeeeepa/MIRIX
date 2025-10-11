import os
import sys

# Add the mirix module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
mirix_root = os.path.dirname(current_dir)
sys.path.insert(0, mirix_root)

# Load .env BEFORE importing mirix - this is critical!
# The database engine is initialized at module import time
from dotenv import load_dotenv

load_dotenv(os.path.join(mirix_root, ".env"))

from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from mirix import Mirix

# Initialize LangChain and Mirix
API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=API_KEY,
)

mirix_agent = Mirix(
    api_key=API_KEY,
    model_provider="gemini",
    config_path=f"{mirix_root}/mirix/configs/mirix_gemini.yaml",
)


class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    user_id: str


graph = StateGraph(State)


def chatbot(state: State):
    messages = state["messages"]
    user_id = state["user_id"]

    try:
        memories = mirix_agent.extract_memory_for_system_prompt(
            messages[-1].content, user_id=user_id
        )

        system_message = (
            "You are a helpful assistant that can answer questions and help with tasks. You have the following memories:\n\n"
            + memories
        )

        full_messages = [system_message] + messages

        response = llm.invoke(full_messages)

        # Store the interaction with Mirix
        try:
            interaction = (
                f"User: {messages[-1].content}\n\nAssistant: {response.content}"
            )
            mirix_agent.add(interaction, user_id=user_id)
        except Exception as e:
            print(f"Error saving memory: {e}")

        return {"messages": [response]}

    except Exception as e:
        print(f"Error in chatbot: {e}")
        # Fallback response without memory context
        response = llm.invoke(messages)
        return {"messages": [response]}


graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()


def run_conversation(user_input: str, user_name: str):
    user = mirix_agent.create_user(user_name=user_name)

    config = {"configurable": {"thread_id": user.id}}
    state = {"messages": [HumanMessage(content=user_input)], "user_id": user.id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            # Check if value is not None and has messages
            if value is not None and value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return


if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")
    user_name = "John Doe"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, user_name)

import gradio as gr
import os
import sys
import importlib.util
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import uuid

# Load environment variables from .env file
load_dotenv()

# Load the agent module from the my_first_agent folder
agent_path = os.path.join(os.path.dirname(__file__), "my_first_agent", "agent.py")
spec = importlib.util.spec_from_file_location("agent_module", agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

# Get the root_agent from the loaded module
root_agent = agent_module.root_agent

# Create session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="my_first_agent", session_service=session_service)

# Track sessions per user
sessions = {}

def chat_with_agent(message, history):
    """
    Chat function that processes user input and returns agent response
    """
    try:
        # Check if API key is set
        if not os.environ.get("GOOGLE_API_KEY"):
            return "❌ GOOGLE_API_KEY not set!\n\nPlease set your Google API key:\n1. Get it from: https://aistudio.google.com/app/apikey\n2. Set it in your environment: $env:GOOGLE_API_KEY='your-key-here'\n3. Or set it in Hugging Face Space Settings → Repository secrets"

        # Get or create session for this user
        user_id = "gradio_user"
        if user_id not in sessions:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            # Create session synchronously
            session_service.create_session_sync(user_id=user_id, session_id=session_id, app_name="my_first_agent")
            sessions[user_id] = session_id
        else:
            session_id = sessions[user_id]
        
        # Create Content object from the message
        content = types.Content(parts=[types.Part(text=message)])
        
        # Run the agent and collect response
        response_text = ""
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            # Extract text from events
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        
        return response_text if response_text else "No response generated"

    except Exception as e:
        import traceback
        return f"❌ Agent Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

# Create Gradio interface
with gr.Blocks(title="AgentHub - Hierarchical Multi-Agent System") as demo:
    gr.Markdown("""
    # 🤖 AgentHub: Hierarchical Multi-Agent System
    
    This agent can help you with:
    - 🔍 **Web Search**: Search the internet for information
    - 💻 **Code Execution**: Perform calculations and run Python code
    
    ## Example Prompts:
    - "What is the current population of Tokyo?"
    - "Calculate 25 * 17 + 42"
    - "What's the weather in Paris and convert 20°C to Fahrenheit"
    """)
    
    chatbot = gr.ChatInterface(
        fn=chat_with_agent,
        examples=[
            "What is the current population of Tokyo?",
            "Calculate 25 * 17 + 42",
            "What's the capital of France and its population?",
            "Convert 100 USD to EUR",
            "What's 15% of 250?"
        ],
        title="💬 Chat with the Agent",
        description="Ask questions or request calculations",
        theme=gr.themes.Soft(),
    )

if __name__ == "__main__":
    demo.launch()
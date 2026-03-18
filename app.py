import streamlit as st
import os
import httpx
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")

# API Keys from Streamlit Secrets
OPENAI_KEY = st.secrets.get("OPENAI_KEY")
TAVILY_KEY = st.secrets.get("TAVILY_KEY")

if not OPENAI_KEY or not TAVILY_KEY:
    st.error("Missing API Keys! Please add OPENAI_KEY and TAVILY_KEY to your Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ==========================================
# 2. THE "SPY" (CALLBACK HANDLER)
# ==========================================
class SearchDebugHandler(BaseCallbackHandler):
    """Custom handler to capture tool outputs across threads safely."""
    def __init__(self):
        self.search_data = []

    def on_tool_end(self, output: str, **kwargs):
        # When the search tool finishes, save its raw output to a list
        self.search_data.append(output)

# ==========================================
# 3. TOOL & AGENT DEFINITION
# ==========================================
@tool
def search_web(query: str) -> str:
    """Use this tool to search the internet for current events, news, and facts."""
    client = httpx.Client(verify=False)
    response = client.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_KEY, "query": query}
    )
    # We return the full JSON string so the Callback Handler captures everything
    return response.text

# Initialize LLM
custom_client = httpx.Client(verify=False)
llm = ChatOpenAI(model="gpt-4o", http_client=custom_client, temperature=0)
tools = [search_web]
current_date = datetime.now().strftime("%B %d, %Y")

# Build the Agent
agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=f"""You are an elite, objective Research Assistant. 
    [CRITICAL CONTEXT]
    Today's exact date is {current_date}. 
    
    [RULES]
    1. MANDATORY SEARCH: Use `search_web` for any real-time data or facts.
    2. STRICT FIDELITY: Only answer based on search results. Do not hallucinate.
    3. SOURCES: You MUST provide clickable markdown links [Title](URL) for your sources at the end.
    """
)

# ==========================================
# 4. STREAMLIT UI & SESSION STATE
# ==========================================
st.title("🔎 My AI Research Assistant")
st.caption(f"Current Date: {current_date} | Powered by GPT-4o & Tavily")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.write("The **Raw Data Accordion** will appear automatically below the AI's response whenever a web search is triggered.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. THE CHAT LOOP
# ==========================================
if prompt := st.chat_input("What should we research today?"):
    
    # Add user message to UI and State
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Logic
    with st.chat_message("assistant"):
        # Initialize the handler for THIS specific turn
        debug_handler = SearchDebugHandler()
        
        with st.spinner("Agent is searching and thinking..."):
            # Run the agent with the callback spy attached
            response = agent.invoke(
                {"messages": st.session_state.messages},
                config={"callbacks": [debug_handler]}
            )
            
            ai_reply = response["messages"][-1].content
            st.markdown(ai_reply)

        # 🐛 THE DEBUG ACCORDION
        # This is safe because it's in the main thread AFTER invoke() is done
        if debug_handler.search_data:
            for i, raw_result in enumerate(debug_handler.search_data):
                with st.expander(f"🔍 Raw Search Result {i+1}", expanded=False):
                    try:
                        # Try to format it as pretty JSON if possible
                        parsed_json = json.loads(raw_result)
                        st.json(parsed_json)
                    except:
                        # Fallback to plain text if it's not JSON
                        st.text(raw_result)

    # Save Assistant reply to State
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
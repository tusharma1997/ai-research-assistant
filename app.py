import streamlit as st
import os
import httpx
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="AI Research Assistant", page_icon="🔎")

# Use st.secrets with fallback to avoid crashes if keys are missing
OPENAI_KEY = st.secrets.get("OPENAI_KEY")
TAVILY_KEY = st.secrets.get("TAVILY_KEY")

if not OPENAI_KEY or not TAVILY_KEY:
    st.error("Missing API Keys! Please add OPENAI_KEY and TAVILY_KEY to your Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ==========================================
# 2. BUILD THE AGENT & TOOL
# ==========================================

# Thread-safe storage for debug data
if "search_debug" not in st.session_state:
    st.session_state.search_debug = []

@tool
def search_web(query: str) -> str:
    """Use this tool to search the internet for current events, news, and facts."""
    client = httpx.Client(verify=False)
    response = client.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_KEY, "query": query}
    )
    data = response.json()
    
    # Store the data in session_state for the UI to pick up after invoke()
    st.session_state.search_debug.append({"query": query, "data": data})
        
    results = [f"Source: {res.get('url')}\nContent: {res.get('content')}" for res in data.get("results", [])]
    return "\n\n".join(results)

tools = [search_web]
current_date = datetime.now().strftime("%B %d, %Y")

# Create the Agent (No caching to allow dynamic tool/UI updates)
custom_client = httpx.Client(verify=False)
llm = ChatOpenAI(model="gpt-4o", http_client=custom_client, temperature=0)

agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=f"""You are an elite, objective Research Assistant. 
    [CRITICAL CONTEXT]
    Today's exact date is {current_date}. 
    
    [TOOL EXECUTION RULES]
    1. MANDATORY SEARCH: You must use the `search_web` tool for any questions regarding current events, live data, prices, or real-time status. Do not guess.
    
    [EPISTEMIC STRICTNESS (NO HALLUCINATIONS)]
    2. STRICT FIDELITY: Base your factual answers EXCLUSIVELY on retrieved search results. If the data is missing, explicitly state: "The search results did not provide this information."
    3. TEMPORAL AWARENESS: Distinguish clearly between past, ongoing, and scheduled future events based on today's date ({current_date}). NEVER report a scheduled event as completed.
    
    [FORMATTING]
    Always provide links to your sources at the end of your response.
    """
)

# ==========================================
# 3. STREAMLIT UI & MEMORY
# ==========================================
st.title("🔎 My AI Research Assistant")
st.caption(f"Today's Date: {current_date} | Powered by GPT-4o & Tavily")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.search_debug = []
        st.rerun()
    st.info("Raw search data will appear as an accordion after the agent finishes.")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. THE CHAT LOOP
# ==========================================
if prompt := st.chat_input("What should we research today?"):
    
    # 1. User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching the web and analyzing..."):
            # Pass memory history into the Agent
            response = agent.invoke({"messages": st.session_state.messages})
            
            ai_reply = response["messages"][-1].content
            st.markdown(ai_reply)
            
            # 3. Debug Accordion (Rendered safely in the main thread)
            if st.session_state.search_debug:
                for debug in st.session_state.search_debug:
                    with st.expander(f"🔍 Raw Data: {debug['query']}", expanded=False):
                        st.json(debug['data'])
                # Clear for the next interaction
                st.session_state.search_debug = []

    # 4. Save reply to memory
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
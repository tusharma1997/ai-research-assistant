import streamlit as st
import os
import httpx
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ==========================================
# 1. SETUP 
# ==========================================
OPENAI_KEY = st.secrets["OPENAI_KEY"]
TAVILY_KEY = st.secrets["TAVILY_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ==========================================
# 2. BUILD THE AGENT & TOOL (No caching)
# ==========================================
custom_client = httpx.Client(verify=False)
llm = ChatOpenAI(model="gpt-4o", http_client=custom_client)

@tool
def search_web(query: str) -> str:
    """Use this tool to search the internet for current events, news, and facts."""
    client = httpx.Client(verify=False)
    response = client.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_KEY, "query": query}
    )
    data = response.json()
    
    # 🐛 DEBUG INJECTION: Draw an expander right in the main chat!
    with st.expander(f"🔍 Debug: Raw Search Data for '{query}'", expanded=False):
        st.json(data)
        
    results = [f"Source: {res.get('url')}\nContent: {res.get('content')}" for res in data.get("results", [])]
    return "\n\n".join(results)

tools = [search_web]
current_date = datetime.now().strftime("%B %d, %Y")

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
    """
)

# ==========================================
# 3. STREAMLIT UI & MEMORY
# ==========================================
st.title("🔎 My AI Research Assistant")
st.caption("Powered by LangChain, OpenAI, and Tavily Search")

# Initialize chat history in Streamlit's session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. THE CHAT LOOP
# ==========================================
# This creates the text box at the bottom of the screen
if prompt := st.chat_input("What should we research today?"):
    
    # 1. Show the user's message on screen and save it
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Send the whole history to the Agent and get a response
    with st.chat_message("assistant"):
        with st.spinner("Searching the web and thinking..."):
            
            # Pass Streamlit's memory directly into the LangChain agent
            response = agent.invoke({"messages": st.session_state.messages})
            
            # Extract the final answer and show it
            ai_reply = response["messages"][-1].content
            st.markdown(ai_reply)
            
    # 3. Save the AI's reply to memory
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
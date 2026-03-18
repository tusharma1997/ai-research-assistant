import streamlit as st
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ==========================================
# 1. SETUP (Replace with your actual keys)
# ==========================================
OPENAI_KEY = ""
TAVILY_KEY = ""

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ==========================================
# 2. CACHE THE AGENT (So it doesn't rebuild on every click)
# ==========================================
@st.cache_resource
def get_agent():
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
        results = [f"Source: {res.get('url')}\nContent: {res.get('content')}" for res in data.get("results", [])]
        return "\n\n".join(results)

    tools = [search_web]
    
    return create_agent(
        model=llm, 
        tools=tools, 
        system_prompt="You are a helpful research assistant. Use your search_web tool to find current facts. Always cite your sources."
    )

agent = get_agent()

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
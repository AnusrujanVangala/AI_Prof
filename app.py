import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tavily import TavilyClient


# --- Load Groq LLM ---
def load_llm(groq_api_key):
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        groq_api_key=groq_api_key
    )


# --- Prompt Template ---
prompt_template = """
You are an {role} who speaks in {tone}.
Answer this question: "{question}"

Include:
- Metaphors: {metaphors}
- Jokes: {jokes}
- Real world examples: {examples}
- Recent news: {news}

Make sure the answer matches the learner’s level.
"""

prompt = PromptTemplate.from_template(prompt_template)


# --- News Integration (Optional) ---
def get_news(topic, api_key):
    tavily = TavilyClient(api_key=api_key)
    response = tavily.search(query=topic)
    return "\n".join([r["content"] for r in response["results"]])


# --- Streamlit UI ---
st.set_page_config(page_title="🧠 AI Professor", layout="centered")
st.title("🧠 AI Professor")
st.subheader("Ask anything — I'll explain it just the way you want!")

groq_api_key = st.text_input("Enter your Groq API Key", type="password")
tavily_api_key = st.text_input("Enter your Tavily API Key (optional)", type="password")

with st.form("input_form"):
    question = st.text_input("What would you like to know?")
    role = st.selectbox("Choose Role", ["Kindergarten Teacher", "School Teacher", "Lecturer", "Professor"])
    tone = st.selectbox("Tone", ["Explain Like I'm 5", "Simple Language", "Scientific Complex"])
    metaphors = st.checkbox("Include Metaphors")
    jokes = st.checkbox("Add Jokes")
    examples = st.checkbox("Real World Examples")
    news = st.checkbox("Include Recent News")

    submit_button = st.form_submit_button("Get Answer")

if submit_button:
    if not groq_api_key:
        st.warning("Please enter your Groq API key.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            # Optional news context
            context = ""
            if news and tavily_api_key:
                context += "\n\nRecent news:\n" + get_news(question, tavily_api_key)

            # Build chain
            llm = load_llm(groq_api_key)
            chain = LLMChain(llm=llm, prompt=prompt)

            # Run chain
            response = chain.invoke({
                "role": role,
                "tone": tone,
                "metaphors": metaphors,
                "jokes": jokes,
                "examples": examples,
                "news": news,
                "question": question + context
            })

            st.markdown("### 📘 Answer:")
            st.write(response["text"])
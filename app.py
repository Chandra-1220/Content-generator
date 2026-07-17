import streamlit as st
from transformers import pipeline

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Content Generator",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

generator = load_model()

# ---------------- Header ----------------
st.title("🤖 AI Content Generator")
st.write("Generate notes, essays, emails, explanations, and more using AI.")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

max_tokens = st.sidebar.slider(
    "Maximum Output Length",
    min_value=50,
    max_value=500,
    value=200
)

# ---------------- Input ----------------
prompt = st.text_area(
    "Enter your prompt",
    height=200,
    placeholder="""Examples:
• Write short notes on Artificial Intelligence.
• Explain Machine Learning.
• Write an email requesting leave.
• Write a paragraph on Climate Change."""
)

# ---------------- Generate ----------------
if st.button("Generate Content", use_container_width=True):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:

        with st.spinner("Generating content..."):

            result = generator(
                prompt,
                max_new_tokens=max_tokens
            )

            answer = result[0]["generated_text"]

        st.success("Content Generated Successfully!")

        st.subheader("Generated Content")
        st.write(answer)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prompt Words", len(prompt.split()))

        with col2:
            st.metric("Generated Words", len(answer.split()))

        st.download_button(
            "Download Result",
            answer,
            file_name="generated_content.txt",
            mime="text/plain"
        )

# ---------------- Footer ----------------
st.divider()
st.caption("Built with ❤️ using Streamlit and Hugging Face")

import streamlit as st
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Content Generator",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.title{
    text-align:center;
    font-size:48px;
    color:#2563EB;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

.output-box{
    padding:20px;
    border-radius:12px;
    background:white;
    border-left:6px solid #2563EB;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

.footer{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

generator = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<p class="title">🤖 AI Content Generator</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Generate creative content instantly using GPT-2</p>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Generation Settings")

max_tokens = st.sidebar.slider(
    "Maximum New Tokens",
    20,
    300,
    100
)

temperature = st.sidebar.slider(
    "Temperature",
    0.1,
    1.5,
    0.7,
    0.1
)

top_p = st.sidebar.slider(
    "Top P",
    0.1,
    1.0,
    0.9,
    0.05
)

# -----------------------------
# User Input
# -----------------------------
prompt = st.text_area(
    "📝 Enter a topic or prompt",
    placeholder="Example: Artificial Intelligence in Healthcare",
    height=180
)

# -----------------------------
# Generate Button
# -----------------------------
if st.button("✨ Generate Content"):

    if prompt.strip() == "":
        st.warning("Please enter a topic or prompt.")

    else:

        with st.spinner("Generating content..."):

            output = generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=50256
            )

            generated_text = output[0]["generated_text"]

        st.success("Content Generated Successfully!")

        st.markdown("## 📄 Generated Content")

        st.markdown(
            f"""
            <div class="output-box">
            {generated_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Input Words",
                len(prompt.split())
            )

        with col2:
            st.metric(
                "Generated Words",
                len(generated_text.split())
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    '<p class="footer">🚀 Built with Streamlit & Hugging Face GPT-2</p>',
    unsafe_allow_html=True
)

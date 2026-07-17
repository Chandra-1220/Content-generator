import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Content Generator",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🤖 AI Content Generator")
st.write("Generate notes, essays, emails, explanations, and more using AI.")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")

max_tokens = st.sidebar.slider(
    "Maximum Output Tokens",
    min_value=50,
    max_value=500,
    value=200
)

# -------------------------------
# User Input
# -------------------------------
prompt = st.text_area(
    "Enter your prompt",
    height=200,
    placeholder="""Examples:

• Write short notes on Artificial Intelligence.
• Explain Machine Learning.
• Write an essay on Climate Change.
• Write an email requesting leave.
"""
)

# -------------------------------
# Generate Button
# -------------------------------
if st.button("✨ Generate Content"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")

    else:

        with st.spinner("Generating content..."):

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                text,
                return_tensors="pt"
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

            answer = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

        st.success("Content Generated Successfully!")

        st.subheader("Generated Content")
        st.write(answer)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prompt Words", len(prompt.split()))

        with col2:
            st.metric("Generated Words", len(answer.split()))

        st.download_button(
            label="📥 Download Result",
            data=answer,
            file_name="generated_content.txt",
            mime="text/plain"
        )

st.divider()

st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers")

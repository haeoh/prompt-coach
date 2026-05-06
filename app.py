import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os

# 1. SETUP THE "KITCHEN" (LangChain)
# We now securely fetch the API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the Brain (The Model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class PromptAnalysis(BaseModel):
    has_persona: bool = Field(description="Does the prompt define a persona or role?")
    has_context: bool = Field(description="Does the prompt provide background context?")
    has_task: bool = Field(description="Does the prompt clearly state the task to be done?")
    has_format: bool = Field(description="Does the prompt specify the output format?")
    feedback: str = Field(description="Detailed feedback for the student on why their prompt is good or bad.")

# Bind the model to our structured output schema
structured_llm = llm.with_structured_output(PromptAnalysis)

# Create the Recipe (The Prompt Template)
# This tells the AI *how* to behave.
template = """
You are an expert Writing Coach.
The student has written the following prompt: "{student_input}"

Please analyze their prompt for the following 4 components: Persona, Context, Task, and Format.
Provide detailed feedback on what is good, what is missing, and how to improve.
"""

prompt_template = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# 2. SETUP THE "WAITER" (Streamlit)
st.title("The Prompt Coach 🤖")
st.header("Draft Your Prompt Below")

# Create a text box for the user
user_input = st.text_area("Type your prompt here:", height=150)

# Track attempt count in session state
if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0

# Create a button
if st.button("Analyze My Prompt"):
    if st.session_state.attempt_count >= 3:
        st.error("⚠️ You have reached the maximum of 3 attempts. To request more attempts, please reach out to Harvey Oh at haeoh@iu.edu")
    elif user_input:
        st.session_state.attempt_count += 1
        
        with st.spinner("Analyzing prompt..."):
            # Format the prompt with the user's input
            formatted_prompt = prompt_template.format(student_input=user_input)
            
            # Send it to the AI to get both feedback AND the boolean flags
            response = structured_llm.invoke(formatted_prompt)
            
            # Figure out what's missing
            missing_components = []
            if not response.has_persona: missing_components.append("persona")
            if not response.has_context: missing_components.append("context")
            if not response.has_task: missing_components.append("task")
            if not response.has_format: missing_components.append("format")

        # Display feedback images on the sidebar
        with st.sidebar:
            if missing_components:
                st.info("⚠️ **Missing Components Detected**")
                st.write("Your prompt is missing:")
                for component in missing_components:
                    st.write(f"- {component.capitalize()}")
                    image_path = os.path.join("images", f"missing_{component}.PNG")
                    if os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
            else:
                st.success("🌟 **Great job! All components included.**")
                image_path = os.path.join("images", "good.PNG")
                if os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
        
        # Display the result
        st.subheader("Coach's Feedback:")
        st.write(response.feedback)
    else:
        st.warning("Please enter a prompt first!")
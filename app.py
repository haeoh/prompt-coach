import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# 1. SETUP THE "KITCHEN" (LangChain)
# We now securely fetch the API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the Brain (The Model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create the Recipe (The Prompt Template)
# This tells the AI *how* to behave.
template = """
You are an expert Writing Coach.
The student has written the following prompt: "{student_input}"

Please analyze their prompt. Tell them if it is good or bad, and why.
"""

prompt_template = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# Helper function to check for key components
def check_prompt_components(text):
    """Check if the prompt includes key components: persona, context, task, format"""
    text_lower = text.lower()
    
    has_persona = any(keyword in text_lower for keyword in [
        "you are", "assume you", "act as", "role", "persona", "character"
    ])
    
    has_context = any(keyword in text_lower for keyword in [
        "background", "context", "situation", "given", "scenario", "environment"
    ])
    
    has_task = any(keyword in text_lower for keyword in [
        "do", "perform", "create", "generate", "write", "analyze", "explain", "task", "action"
    ])
    
    has_format = any(keyword in text_lower for keyword in [
        "format", "output", "structure", "organize", "style", "template", "json", "list", "paragraph"
    ])
    
    return {
        "persona": has_persona,
        "context": has_context,
        "task": has_task,
        "format": has_format
    }

# 2. SETUP THE "WAITER" (Streamlit)
st.title("The Prompt Coach 🤖")
st.header("Draft Your Prompt Below")

# Create a text box for the user
user_input = st.text_area("Type your prompt here:", height=150)

# Create a button
if st.button("Analyze My Prompt"):
    if user_input:
        # Check for missing components
        components = check_prompt_components(user_input)
        missing_components = [key for key, value in components.items() if not value]
        
        # Display missing components on the sidebar if any are missing
        if missing_components:
            with st.sidebar:
                st.info("⚠️ **Missing Components Detected**")
                st.write("Your prompt is missing:")
                for component in missing_components:
                    st.write(f"- {component.capitalize()}")
                st.divider()
                st.subheader("✨ Animation")
                # video_file = open(r"c:\Users\haeoh\Untitled.mp4", "rb")
                # st.video(video_file)
        
        # 3. CONNECT THEM
        # Format the prompt with the user's input
        formatted_prompt = prompt_template.format(student_input=user_input)
        
        # Send it to the AI (The Kitchen cooks the meal)
        response = llm.invoke(formatted_prompt)
        
        # Display the result
        st.subheader("Coach's Feedback:")
        st.write(response.content)
    else:
        st.warning("Please enter a prompt first!")
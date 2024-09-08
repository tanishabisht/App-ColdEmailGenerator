import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

from chains import Chain
import util

def create_streamlit_app(chain, clean_text):
    st.title("📧 Cold Mail Generator")

    # Resume Upload and Processing
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

    if uploaded_file:
        temp_file = "app/resource/temp.pdf"

        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Extract and store resume information
        resume_loader = PyPDFLoader(temp_file)
        # documents = resume_loader.load()
        # st.write(str(documents))

        resume_data = resume_loader.load()
        resume_text = clean_text(resume_data[0].page_content)
        
        # Extract resume info and store in Chroma DB
        resume_info = chain.extract_resume_info(resume_text)

    # Job Posting Extraction and Email Generation

    # New User Inputs for Detailed Descriptions
    user_name = st.text_input("Enter your name:", value="Tanisha Bisht")
    college_name = st.text_input("Enter the name of your university:", value="Columbia University")

    user_personal_traits = st.text_area("Describe your unique personal traits and characteristics:", 
                                        "What makes you unique? What qualities or experiences define you?")

    user_corporate_life_description = st.text_area("Describe how you perceive yourself in a corporate environment:", 
                                                   "How do you approach work in a corporate setting? What are your values and work style?")

    url_input = st.text_input("Enter a URL:", value="https://www.amazon.jobs/en/jobs/2644301/software-development-engineer-2024-us")

    submit_button = st.button("Submit")


    if submit_button:
        try:
            # Extract job postings from the URL
            loader = WebBaseLoader([url_input])
            page_data = clean_text(loader.load().pop().page_content)
            job_postings = chain.extract_job_postings(page_data)

            # Generate emails for each job posting
            for job in job_postings:
                # Generate email content
                email = chain.compose_email(
                    student_name=user_name,
                    college=college_name,
                    job_desc=job,
                    resume=resume_info,
                    personal_content=user_personal_traits,
                    corporate_life_values=user_corporate_life_description
                )

                st.code(email, language='markdown')
                
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, util.clean_text)
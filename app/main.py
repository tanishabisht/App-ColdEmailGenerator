# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# from chains import Chain
# import util


# def create_streamlit_app(llm, clean_text):
#     st.title("📧 Cold Mail Generator")

#     uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

#     if uploaded_file :

#         temp_file = "app/resource/temp.pdf"

#         with open(temp_file, "wb") as file:
#             file.write(uploaded_file.getvalue())

#         loader = PyPDFLoader(temp_file)
#         documents = loader.load_and_split()
        
#         st.write(str(documents))


#     # if uploaded_file is not None:
#     #     loader = PyPDFLoader('app/resource/resume.pdf')
#     #     resume_content = loader.load()
        

#     #     st.write(str(resume_content))
#     #     # st.write("Resume Content:")
#     #     # st.write(resume_text)
#     # else:
#     #     st.write("Please upload a resume to proceed.")
    
        
#     url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
#     submit_button = st.button("Submit")

#     if submit_button:
#         try:
#             loader = WebBaseLoader([url_input])
#             data = clean_text(loader.load().pop().page_content)
#             # portfolio.load_portfolio()
#             jobs = llm.extract_jobs(data)
#             for job in jobs:
#                 skills = job.get('skills', [])
#                 # links = portfolio.query_links(skills)
#                 # email = llm.write_mail(job, links)
#                 # st.code(email, language='markdown')
#         except Exception as e:
#             st.error(f"An Error Occurred: {e}")


# if __name__ == "__main__":
#     chain = Chain()
#     # portfolio = Portfolio()
#     st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
#     create_streamlit_app(chain, util.clean_text)


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
        chain.store_in_chroma_db(resume_info)

    # Job Posting Extraction and Email Generation

    # New User Inputs for Detailed Descriptions
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
            print(job_postings)

            # Generate emails for each job posting
            for job in job_postings:
                role = job.get('role', '')
                skills = job.get('skills', [])
                experience = job.get('experience', '')

                # Retrieve portfolio links based on skills (assumed to be part of portfolio functionality)
                print('worked before resume')
                resume = chain.query([role], [experience], skills)  # Assuming a utility function or method for this
                print('worked after resume')
                
                # Generate email content
                email = chain.compose_email(
                    student_name="Tanisha Bisht",  # Replace with actual name or input
                    college="Columbia University",  # Replace with actual college or input
                    job_desc=job,
                    resume=resume,
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
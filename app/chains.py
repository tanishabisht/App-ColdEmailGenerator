import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.1-70b-versatile"
        )

    def extract_resume_info(self, resume_text):
        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM RESUME: 
            {resume_text}
            Extract the following information and return as valid JSON:
            1. **Personal Info**: Website, LinkedIn, GitHub, email, phone.
            2. **Skills**: List of skills.
            3. **Work Experience**: 
                - **Roles**: Titles and descriptions.
                - **Experience**: Dates and descriptions.
            4. **Projects**: List with descriptions.
            Return only the JSON.
            ### VALID JSON (NO PREAMBLE):
            ```yaml
                personal_info:
                    website: "string|null"
                    linkedin: "string|null"
                    github: "string|null"
                    email: "string|null"
                    phone: "string|null"
                skills:
                    - "string|null"
                work_experience:
                    - role: "string|null"
                    company: "string|null"
                    experience: "string|null"
                projects:
                    - name: "string|null"
                    description: "string|null"
            ```
            """
        )

        chain = prompt | self.llm
        response = chain.invoke(input={"resume_text": resume_text})

        try:
            parser = JsonOutputParser()
            result = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse resume.")

        return result

    def extract_job_postings(self, page_text):
        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_text}
            ### INSTRUCTION:
            Extract job postings from the careers page and return as JSON with keys:
            - `role`
            - `experience`
            - `skills`
            - `description`
            Return only the JSON. No preamble.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain = prompt | self.llm
        response = chain.invoke(input={"page_text": page_text})

        try:
            parser = JsonOutputParser()
            result = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse job postings.")

        return result if isinstance(result, list) else [result]

    def compose_email(self, student_name, college, job_desc, resume, personal_content, corporate_life_values):
        prompt = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_desc}

            ### {student_name}'s PORTFOLIO:
            {resume}

            ### INSTRUCTION:
            You are {student_name}, a computer science student at {college}. Write a compelling cold email to the hiring manager for the job described above. The email should sound natural and include the following elements:
            1. **Align Skills and Experiences**: Clearly highlight how your skills and past work experiences align with the key requirements and responsibilities mentioned in the job description.
            2. **Showcase Relevant Projects**: Mention specific projects from your portfolio that demonstrate the relevant skills and experience applicable to the role.
            3. **Highlight Unique Traits**: Explain what makes you stand out as a candidate by including:
                - **Personal Traits**: {personal_content}
                - **Corporate Life Perspective**: {corporate_life_values}
            4. **Align Values with Company Culture**: Illustrate how your values and experiences align with the company's culture, mission, and values. Connect your personal traits and corporate perspective with the company's ethos.
            5. **Explain Motivation and Fit**: Articulate why you are interested in this company and role. Mention specific aspects of the company’s work, reputation, culture, or mission that appeal to you and align with your career goals.
            6. **Call to Action**: Request a brief call to discuss how your skills and passion could benefit the team.
            Ensure the email is concise, engaging, and directly relevant to the job description. The goal is to express your interest, demonstrate you are the right fit, and include a clear call to action to discuss how your skills and passion can benefit their team.

            ### GOAL OF THE EMAIL:
            You have already applied for the job. Now you are sending a follow-up email to the hiring manager to express your interest, demonstrate fit, and request a discussion about how your skills and passion can benefit their team.

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "student_name": student_name, 
            "college": college, 
            "job_desc": job_desc, 
            "resume": resume,
            "personal_content": personal_content,
            "corporate_life_values": corporate_life_values
        })
        
        return response.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
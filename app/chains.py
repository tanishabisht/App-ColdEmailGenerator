import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.1-70b-versatile"
        )

        self.client = chromadb.PersistentClient('vectorstore')
        self.collection = self.client.get_or_create_collection("resume_data")

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
            You are {student_name}, a computer science student at {college}. Write a compelling cold email for the job described above. Make sure to personalize the email based on the following instructions:
                1. **Align Skills and Experiences**: Highlight how {student_name}'s skills and past work experiences directly match the key requirements and responsibilities mentioned in the job description.
                2. **Showcase Relevant Projects**: Provide examples of specific projects from {student_name}'s portfolio that demonstrate relevant skills and experience that are highly applicable to the role.
                3. **Highlight Unique Traits**: Use the unique traits provided by {student_name} to explain what makes them stand out as a candidate. Specifically include these details:
                    - **Personal Traits**: {personal_content}
                    - **Corporate Life Perspective**: {corporate_life_values}
                4. **Align Values with Company Culture**: Illustrate how {student_name}'s values and experiences align with the company's culture, mission, and values. Make sure to draw connections between {student_name}'s personal traits and corporate life perspective with the company's ethos.
                5. **Explain Motivation and Fit**: Clearly articulate why {student_name} is interested in this particular company and role. Mention specific aspects of the company’s work, reputation, culture, or mission that appeal to {student_name} and how these align with their career aspirations.
            Ensure that the email is concise, engaging, and directly relevant to the job description.
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

    def store_in_chroma_db(self, resume_info):

        if not resume_info:
            raise ValueError("No resume information provided to store.")

        # Prepare documents and metadata for Chroma DB
        documents = []
        metadata_list = []

        if 'personal_info' in resume_info:
            documents.append(str(resume_info['personal_info']))
            metadata_list.append({"type": "personal_info"})
        if 'skills' in resume_info:
            documents.append(str(resume_info['skills']))
            metadata_list.append({"type": "skills"})
        if 'work_experience' in resume_info:
            documents.append(str(resume_info['work_experience']))
            metadata_list.append({"type": "work_experience"})
        if 'projects' in resume_info:
            documents.append(str(resume_info['projects']))
            metadata_list.append({"type": "projects"})

        # Add documents to Chroma DB collection
        self.collection.add(
            ids=[f"doc_{i}" for i in range(len(documents))],  # Unique IDs for each document
            documents=documents,
            metadatas=metadata_list
        )

    def query(self, role, experience, skills):
        combined_query_texts = role + experience + skills
        # Perform the query on the Chroma collection
        query_result = self.collection.query(query_texts=combined_query_texts, n_results=2)
        # Extract and return the metadata from the query results
        return query_result.get('metadatas', [])

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
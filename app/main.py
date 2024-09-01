from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import uuid
import chromadb


# create an llm instance
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key="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",
)


# Scrape the website
loader = WebBaseLoader("https://www.amazon.jobs/en/jobs/2644301/software-development-engineer-2024-us")
page_data = loader.load().pop().page_content


# Write prompt to get json formatted info: role, experience, skills and description
prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of the website.
    Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)


# getting a prompt and passing to the llm so that we receive llm response
chain_extract = prompt_extract | llm 


res = chain_extract.invoke(input={"page_data": page_data})
# print(res.content) # this is in str format


# convert string format to json format
json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
# print(json_res) # this is in json format


# import your records as dataframe
df = pd.read_csv('my_portfolio.csv')


# insert records into chromadb
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=row["Techstack"],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )


link_list = collection.query(query_texts=json_res['skills'], n_results=2).get('metadatas', [])

# prompt template to write an email 
prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Tanisha, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        Remember you are Mohan, BDE at AtliQ. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

chain_email = prompt_email | llm
res_email = chain_email.invoke({"job_description": str(json_res), "link_list": link_list})
print(res_email.content)
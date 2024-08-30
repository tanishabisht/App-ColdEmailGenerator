from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.amazon.jobs/en/jobs/2644301/software-development-engineer-2024-us")

page_data = loader.load().pop().page_content
print(page_data)
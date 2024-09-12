# App - Cold Email Generator

The Cold Email Generator helps computer science students craft personalized cold emails to hiring managers using Llama3.1 LLM, Langchain, and Streamlit and Streamlit Community Cloud for deployment.

## Features

- **Resume Processing**: Upload and process your resume to extract key information.
- **Job Posting Extraction**: Input a job URL to extract job details.
- **Personalized Emails**: Generate tailored cold emails based on your resume and job description.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:tanishabisht/App-ColdEmailGenerator.git
   cd App-ColdEmailGenerator
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda create --name cold_email_env python=3.8
   conda activate cold_email_env
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app/main.py
   ```

## Usage

1. Upload your resume.
2. Enter the job posting URL.
3. Generate and review your cold email.
# AIonOS_Training_Assignment1

Note :- 
1. Create the folder named "Resume" in root Folder.
2. Set Groq API Key
- Create a .env file in the project root: .env
- GROQ_API_KEY=your_groq_api_key_here

Steps to run the code :- 

Terminal 1 :- 

1. create virtual environment
python -m venv venv

2. activate it
source venv/bin/activate

3. install all requirements
pip install -r requirements.txt



Terminal 2 :-

1. Downaload Ollama - gemma3:1b

2. serve the model
ollama serve

3. Pull the model 
ollama pull gemma3:1b



Terminal 3 :- 

1. activate venv
source venv/bin/activate

2. Run the streamlit commands :- 
streamlit run app.py



now upload the documents.

Sample User mesaage :- 

1. Summarize the PDF <pdf name>.

2. I need 10 quiz questions to test my knowledge. Include a few multiple-choice questions (each with 4 options), a few true/false questions, and a few short-answer questions. Also provide the correct answers for each question so I can check my knowledge.

3. General Question answer regarding the PDF's 



Reference:-

For Demo :-
- Google Drive Link :- https://drive.google.com/file/d/1ZSA17HPtmV7j5XH-cLKMxv2kxS9yvOS_/view?usp=drivesdk

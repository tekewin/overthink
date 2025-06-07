# Overthink
Improved LLM Reasoning with Self-Para-Consistency

A python web app based on this Association for Computational Linguistics research paper by Wenqing Chen, et al., 2024: https://aclanthology.org/2024.findings-acl.842/. 

The core idea was to generate paraphrases of the original prompt, then have the LLM vote on which response was best. Overthink makes few changes in the implementation.

1. It does 16 iterations on the original user prompt (calling an OpenAI model).
2. The first call to an LLM API uses the original prompt, the rest get the original plus a paraphrased version.
3. Each call gets a different temperature setting (from 0.0 to 1.0). This was done to try to increase creative responses.
4. Six of the sixteen get a system prompt telling it to be creative.
5. Instead of the same LLM voting on the best response, all responses are sent to Google Gemini for a consolidated final answer.
6. The UI is updated in real time.
7. All responses and token counts are saved to a local Sqlite3 database for future reference.

![overthink-ui](https://github.com/user-attachments/assets/c0587e2e-6886-4230-a6b6-21da73af0088)

### Setup

1. install python 3.13+ and sqlite3
2. create a python virtual environment and activate
3. pip install -r requirements.txt
4. create the logs.db database: sqlite3 logs.db (the app will create the logs table the first time it runs)
5. edit the .env file to add your OpenAI and Google Gemini API keys


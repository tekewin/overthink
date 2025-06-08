# OverThink
Improved LLM Reasoning with Self-Para-Consistency

A python web app based on this Association for Computational Linguistics research paper by Wenqing Chen, et al., 2024: https://aclanthology.org/2024.findings-acl.842/. 

The core idea is to generate paraphrases of the original prompt, then have the LLM vote on which response was best. OverThink makes a few changes in the implementation.

- It does 16 iterations on the original user prompt (calling an OpenAI model).
- The first call to an LLM API uses the original prompt, the rest get the original plus a paraphrased version.
- Each call gets a different temperature setting (from 0.0 to 1.0). This was done to try to increase creative responses.
- Six of the sixteen get a system prompt telling it to be creative.
- Instead of the same LLM voting on the best response, all responses are sent to Google Gemini for a consolidated final answer.
- The UI is updated in real time.
- All responses and token counts are saved to a local Sqlite3 database for future reference.

![overthink-ui](https://github.com/user-attachments/assets/c0587e2e-6886-4230-a6b6-21da73af0088)

### Setup

1. install python 3.13+ and sqlite3
2. create a python virtual environment and activate
3. pip install -r requirements.txt
4. edit the .env file to add your OpenAI and Google Gemini API keys
5. run: python app.py
6. open a web browser and go to http://127.0.0.1:5000/

### API pricing

Here are links to API pricing. Cached pricing probably doesn't apply in any use case since the app sends different paraphrased prompts on each call.

By default, the app calls OpenAI GPT 4o mini for paraphrases and initial responses. Then, Gemini 2.5 pro to consolidate all responses and create a final answer. You can experiment with different combinations of models.

[OpenAI pricing](https://platform.openai.com/docs/pricing)

[Google pricing](https://ai.google.dev/gemini-api/docs/pricing)

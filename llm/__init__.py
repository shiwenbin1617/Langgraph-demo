from langchain.chat_models import init_chat_model

llm_openai = init_chat_model(model="gpt-4o-mini", model_provider="openai")
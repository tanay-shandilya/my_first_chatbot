from dotenv import load_dotenv
load_dotenv()
import warnings;
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

chat_history =[
    SystemMessage(content='You are a helpful assistant'),
]


while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))

    if user_input=='Exit' or user_input=='exit':
        break;
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ',result.content)

print(chat_history)


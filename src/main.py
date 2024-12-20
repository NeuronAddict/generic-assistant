import argparse
import os
from typing import List
from dotenv import dotenv_values
import gradio as gr

from ai_session import AISession
from assistant import Assistant
from model_factory import ModelFactory, MistralFactory

parser = argparse.ArgumentParser('ChatBot Interface')
parser.add_argument('--env_file', help='Env file to use.', default='.env')
args = parser.parse_args()

config = dotenv_values(args.env_file, verbose=True)

factory: ModelFactory = MistralFactory(os.environ['MISTRAL_API_KEY'])
assistant: Assistant = Assistant(factory)

ai_session: AISession | None = None

def chat(question: str, _: List[str]) -> str:
    global ai_session
    if ai_session is None:
        if config['MODEL_NAME'] is None or config['SYSTEM_PROMPT'] is None or config['TEMPERATURE'] is None:
            raise Exception('Please provide MODEL_NAME, SYSTEM_PROMPT and TEMPERATURE')
        else:
            ai_session = assistant.new_session(config['MODEL_NAME'], config['SYSTEM_PROMPT'], float(config['TEMPERATURE']),
                                               config['PREFIX'])
    return ai_session.ask(question)


def clear():
    global ai_session
    if config['MODEL_NAME'] is None or config['SYSTEM_PROMPT'] is None or config['TEMPERATURE'] is None:
        raise Exception('Please provide MODEL_NAME, SYSTEM_PROMPT and TEMPERATURE')
    else:
        ai_session = assistant.new_session(config['MODEL_NAME'], config['SYSTEM_PROMPT'], float(config['TEMPERATURE']),
                                           config['PREFIX'])


if __name__ == '__main__':
    with gr.Blocks() as demo:

        with gr.Column():

            with gr.Column(scale=4, variant='panel'):

                chatbot=gr.Chatbot(height='40vh', render=False, type='messages')
                chatbot.clear(clear)
                gr.ChatInterface(
                    chat, type="messages",
                    description=config['DESCRIPTION'],
                    chatbot=chatbot,
                    theme="default",
                    cache_examples=False,
                    fill_height=True
                )


    demo.launch()
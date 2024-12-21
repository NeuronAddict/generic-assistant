import argparse
import os
from typing import List
from dotenv import dotenv_values
import gradio as gr

from ai_session import AISession
from assistant import Assistant
from model_factory import ModelFactory, MistralFactory

parser = argparse.ArgumentParser('ChatBot Interface')
parser.add_argument('--env-file', help='Env file to use.', default='.env')
args = parser.parse_args()

config = dotenv_values(args.env_file, verbose=True)

factory: ModelFactory = MistralFactory(os.environ['MISTRAL_API_KEY'])
assistant: Assistant = Assistant(factory)

ai_session: AISession | None = None

def chat(question: str, _: List[str], model: str, system_prompt: str, temperature: float, prefix: str) -> str:
    global ai_session
    if ai_session is None:
        ai_session = assistant.new_session(model, system_prompt, temperature, prefix)
    return ai_session.ask(question)

def clear(model: str, system_prompt: str, temperature: float, prefix: str):
    global ai_session
    ai_session = assistant.new_session(model, system_prompt, temperature, prefix)


if __name__ == '__main__':
    with gr.Blocks() as demo:

        with gr.Column():

            with gr.Row():

                model_choose = gr.Dropdown(['mistral-small-latest', 'mistral-large-latest', 'ministral-8b-latest', 'ministral-3b-latest'],
                                       type='value', label="Choisissez un modèle")

                temp_slider = gr.Slider(0, 1, value=0.2, label="Temperature",
                                    info="Choose temperature https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post")

                system_prompt = gr.Textbox(label="System prompt", lines=3,
                                       value=config['SYSTEM_PROMPT'])
                prefix = gr.Textbox(label="Prefix", lines=3,
                                value=config['PREFIX'])

            with gr.Column(scale=4, variant='panel'):

                chatbot=gr.Chatbot(height='40vh', render=False, type='messages')
                chatbot.clear(clear, [model_choose, system_prompt, temp_slider, prefix])
                gr.ChatInterface(
                    chat, type="messages",
                    description=config['DESCRIPTION'],
                    additional_inputs=[model_choose, system_prompt, temp_slider, prefix],
                    chatbot=chatbot,
                    theme="default",
                    cache_examples=False,
                    fill_height=True
                )


    demo.launch()
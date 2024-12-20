import os
from typing import List

import gradio as gr
from mistralai import Mistral

from ai_session import LogAISession, BaseAISession, AISession
from model import BaseModel

client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])

ai_session: AISession | None = None

def chat(question: str, _: List[str], model: str, system_prompt: str, temperature: float, prefix: str) -> str:
    global ai_session
    if ai_session is None:
        ai_session = LogAISession(BaseAISession( BaseModel(client, model), system_prompt, temperature, prefix))
    return ai_session.ask(question)

if __name__ == '__main__':

    with gr.Blocks(theme='gradio/monochrome') as demo:

        with gr.Column():

            with gr.Row():
                model_choose = gr.Dropdown(['mistral-small-latest', 'mistral-large-latest', 'ministral-8b-latest', 'ministral-3b-latest'],
                                       type='value', )
                temp_slider = gr.Slider(0, 1, value=0.2, label="Temperature",
                                    info="Choose temperature https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post")


            with gr.Column(scale=4, variant='panel'):
                with gr.Row():
                    system_prompt = gr.Textbox(label="System prompt", lines=3, value='Vous êtes un assistant utile et courtois. Vous répondez avec des références à des sources fiables. Vous répondez en francais.')
                    prefix = gr.Textbox(label="Prefix", lines=3, value='Réponse en français de l\'assistant, avec des sources fiables :')

                gr.ChatInterface(
                    chat, additional_inputs=[model_choose, system_prompt, temp_slider, prefix], type="messages",
                    description="Posez votre question, attention à ne rien dire de confidentiel",
                    theme="default",
                    chatbot=gr.Chatbot(height='40vh', render=False, type='messages'),
                    cache_examples=False,
                    fill_height=True
                )

    demo.launch()
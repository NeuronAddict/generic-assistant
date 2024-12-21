from typing import List

import gradio as gr

from ai.assistant import Assistant
from ai.ai_session import AISession
from .ui_factory import UI


class GenericUI(UI):

    def launch(self):
        self.blocks.launch()

    def chat(self, question: str, _: List[str], model: str, system_prompt: str, temperature: float, prefix: str) -> str:
        if self.ai_session is None:
            self.ai_session = self.assistant.new_session(model, system_prompt, temperature, prefix)
        return self.ai_session.ask(question)

    def clear(self, model: str, system_prompt: str, temperature: float, prefix: str):
        self.ai_session = self.assistant.new_session(model, system_prompt, temperature, prefix)

    def __blocks(self):

        with gr.Blocks() as demo:
            with gr.Column():
                with gr.Row():
                    model_choose = gr.Dropdown(
                        ['mistral-small-latest', 'mistral-large-latest', 'ministral-8b-latest', 'ministral-3b-latest'],
                        type='value', label="Choisissez un modèle")

                    temp_slider = gr.Slider(0, 1, value=0.2, label="Temperature",
                                            info="Choose temperature https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post")

                    system_prompt = gr.Textbox(label="System prompt", lines=3,
                                               value=self.config['SYSTEM_PROMPT'])
                    prefix = gr.Textbox(label="Prefix", lines=3,
                                        value=self.config['PREFIX'])

                with gr.Column(scale=4, variant='panel'):
                    chatbot = gr.Chatbot(height='40vh', render=False, type='messages')
                    chatbot.clear(self.clear, [model_choose, system_prompt, temp_slider, prefix])
                    gr.ChatInterface(
                        self.chat, type="messages",
                        description=self.config['DESCRIPTION'],
                        additional_inputs=[model_choose, system_prompt, temp_slider, prefix],
                        chatbot=chatbot,
                        theme="default",
                        cache_examples=False,
                        fill_height=True
                    )
        return demo

    def __init__(self, config: dict, assistant: Assistant):
        self.config = config
        self.assistant = assistant
        self.blocks = self.__blocks()
        self.ai_session: AISession | None = None
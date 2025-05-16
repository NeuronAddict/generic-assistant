from typing import List

import gradio as gr

from ai.assistant import Assistant
from .ui_factory import UI


class SpecificUI(UI):

    def launch(self, ip: str, port: int):
        self.blocks.launch(server_name=ip, server_port=port)

    def chat(self, question: str, _: List[str]) -> str:
        if self.ai_session is None:
            self.ai_session = self.assistant.new_session(self.config['MODEL_NAME'], self.config['SYSTEM_PROMPT'],
                                                   float(self.config['TEMPERATURE']),
                                                   self.config['PREFIX'])
        return self.ai_session.ask(question)

    def clear(self):
        if self.config['MODEL_NAME'] is None or self.config['SYSTEM_PROMPT'] is None or self.config['TEMPERATURE'] is None:
            raise Exception('Please provide MODEL_NAME, SYSTEM_PROMPT and TEMPERATURE')
        else:
            self.ai_session = self.assistant.new_session(self.config['MODEL_NAME'], self.config['SYSTEM_PROMPT'],
                                               float(self.config['TEMPERATURE']),
                                               self.config['PREFIX'])

    def __blocks(self):
        with gr.Blocks() as demo:
            with gr.Column():
                with gr.Column(scale=4, variant='panel'):
                    chatbot = gr.Chatbot(height='40vh', render=False, type='messages')
                    chatbot.clear(self.clear)
                    gr.ChatInterface(
                        self.chat, type="messages",
                        description=self.config['DESCRIPTION'],
                        chatbot=chatbot,
                        theme="default",
                        cache_examples=False,
                        fill_height=True
                    )
        return demo

    def __init__(self, config: dict, assistant: Assistant):
        if config['MODEL_NAME'] is None or config['SYSTEM_PROMPT'] is None or config['TEMPERATURE'] is None:
            raise Exception('Please provide MODEL_NAME, SYSTEM_PROMPT and TEMPERATURE')

        self.config = config
        self.assistant = assistant
        self.blocks = self.__blocks()
        self.ai_session = None
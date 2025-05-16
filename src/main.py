import os

from dotenv import dotenv_values

from ai.ai_session import AISession
from ai.assistant import Assistant
from ai.model_factory import ModelFactory, MistralFactory
from ui import parser
from ui.ui_factory_impl import GenericUIFactory, SpecificUIFactory

if __name__ == '__main__':
    args = parser.arguments()

    config = dotenv_values(args.env_file, verbose=True)

    factory: ModelFactory = MistralFactory(os.environ['MISTRAL_API_KEY'])
    assistant: Assistant = Assistant(factory)

    ai_session: AISession | None = None

    if 'GENERIC' in config and config['GENERIC']:
        ui_factory = GenericUIFactory()
    else:
        ui_factory = SpecificUIFactory()

    print(f"Start Server {config["IP"]}:{config["PORT"]}...")

    ui_factory.get_ui(config, assistant).launch(config["IP"], int(config["PORT"]))

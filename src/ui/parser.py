import argparse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser('ChatBot Interface')
    parser.add_argument('--env-file', help='Env file to use.', default='.env')
    return parser.parse_args()
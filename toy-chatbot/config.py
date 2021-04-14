from dataclasses import dataclass


@dataclass
class Config:
    Data: str='./data/chatbot_data.txt'
    BERTMultiLingual: str='bert-base-multilingual-cased'
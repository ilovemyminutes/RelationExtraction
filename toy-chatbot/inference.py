import pandas as pd
from transformers import AutoModel, AutoTokenizer
from config import Config


chats_raw = pd.read_csv(Config.Data)

questions =  chats_raw['Q'].tolist()
answers = chats_raw['A'].tolist()

model = AutoModel.from_pretrained(Config.BERTMulti)

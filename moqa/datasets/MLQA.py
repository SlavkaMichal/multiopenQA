from torchtext.data import Dataset, Field, Exmaple, RawField
from tqdm import tqdm

class MLQADataset(Dataset):

    question = Field(sequential=True, use_vacab=True, tokenize=tokenizer, lower=True)
    question = Field(sequential=True, use_vacab=True, tokenize=tokenizer, lower=True)

    fields = {'question':('q',question), 'id':('id', id)}





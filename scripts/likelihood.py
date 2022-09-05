import kenlm
import sentencepiece as spm
model = kenlm.LanguageModel('/home/ubuntu/kenlm/kenlm_saya_sent_3gram.binary')

SPM = "/home/ubuntu/Platform/data/dicts/sp_oall_32k.model"
sp = spm.SentencePieceProcessor()
sp.Load(SPM)

def clc_saya_likelihood(src: str, gram: int=3) -> float:
    src_tokenized = sp.encode_as_pieces(src) 
    return model.score(" ".join(src_tokenized)) / len(src_tokenized)

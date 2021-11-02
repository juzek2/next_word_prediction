# %%
import torch
import string

from transformers import RobertaTokenizerFast, RobertaForMaskedLM
bartolomej_tokenizer = RobertaTokenizerFast.from_pretrained('./bartolomej')
bartolomej_tokenizer.save_pretrained('./bartolomej')
bartolomej_model = RobertaForMaskedLM.from_pretrained('bartolomej').eval()
bartolomej_model.save_pretrained('./bartolomej')

from transformers import RobertaTokenizer, RobertaModel
slovakbert_tokenizer = RobertaTokenizer.from_pretrained('gerulata/slovakbert')
slovakbert_tokenizer.save_pretrained('./slovakbert')
slovakbert_model = RobertaModel.from_pretrained('gerulata/slovakbert').eval()
slovakbert_model.save_pretrained('./slovakbert')

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BARTOLOMEJ =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bartolomej_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bartolomej_model(input_ids)[0]
    bartolomej = decode(bartolomej_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= SLOVAKBERT =================================
    input_ids, mask_idx = encode(slovakbert_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    slovakbert = decode(slovakbert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLM ROBERTA BASE =================================
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'bartolomej': bartolomej,
            'slovakbert': slovakbert,
            'xlm': xlm}

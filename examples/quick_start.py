from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# model_id = 'charent/ChatLM-mini-Chinese'
model_id = '/home/kevin/huggingface/ChatLM-mini-Chinese'

# 如果无法连接huggingface，打开以下两行代码的注释，将从modelscope下载模型文件，模型文件保存到'./model_save'目录
# from modelscope import snapshot_download
# model_id = snapshot_download(model_id, cache_dir='../model_save')
# print(model_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

txt = '如何评价Apple这家公司？'

encode_ids = tokenizer([txt])
input_ids, attention_mask = torch.LongTensor(encode_ids['input_ids']), torch.LongTensor(encode_ids['attention_mask'])

outs = model.my_generate(
    input_ids=input_ids.to(device),
    attention_mask=attention_mask.to(device),
    max_seq_len=256,
    search_type='beam',
)

outs_txt = tokenizer.batch_decode(outs.cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(outs_txt[0])
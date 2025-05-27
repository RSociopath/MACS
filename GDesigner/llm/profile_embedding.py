from sentence_transformers import SentenceTransformer
import torch


# 选择 GPU 0

def get_sentence_embedding(sentence):
    model = SentenceTransformer('/public/home/wxhu_br/GNN/GDesigner-main/GDesigner/llm/all')


# 使用 GPU 0

    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)
def get_sim_embedding(sentence):
    model = SentenceTransformer('/public/home/wxhu_br/GNN/GDesigner-main/GDesigner/llm/all')


# 使用 GPU 0
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)
# import torch
# import torch.nn.functional as F
# from sentence_transformers import SentenceTransformer, models

# # 原始模型
# base_model = SentenceTransformer('/home/teachhu/wyh/GDesigner-main/GDesigner/llm/all-distilroberta-v1', local_files_only=True)

# # 降维：添加一个 Dense 层，投影到 384 维
# dense = models.Dense(in_features=768, out_features=384, activation_function=F.tanh)
# base_model.add_module('dense', dense)

# # 保存这个新模型（可选）
# # base_model.save('your/path/all-distilroberta-384')

# # 使用时就会输出 384 维
# def get_sentence_embedding(sentence):
#     embedding = base_model.encode(sentence)
#     return torch.from_numpy(embedding)

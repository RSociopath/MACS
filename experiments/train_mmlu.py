import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            dir=""
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=lr)    
    graph.gcn.train()
    print(f"training gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    start_time = time.time()
    LLM_consumed=0
    GNN_consumed=0
    LLM_list=[]
    GNN_list=[]
    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        LLM_start=time.time()
        LLM_list.append(LLM_start)
        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds=num_rounds,ix=i_record+batch_size,bs=batch_size,dir=dir)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)


        raw_results = await asyncio.gather(*answer_log_probs)
        LLM_end = time.time()
        LLM_consumed += LLM_end - LLM_start
        raw_answers, log_probs,i = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")
        GNN_start=time.time()
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        GNN_end=time.time()
        GNN_list.append(GNN_end)
        GNN_consumed+=GNN_end-GNN_start
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        print("loss:", total_loss.item()) # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
    print(f"LLM_consumed: {LLM_consumed}")
    print(f"GNN_consumed: {GNN_consumed}")


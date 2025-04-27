import os,torch

save_dir = "/home/teachhu/model_saved"
os.makedirs(save_dir, exist_ok=True)
acc=0.66
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy,torch

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        dir="",
        ) -> float:

    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    graph.gcn=torch.load("/home/teachhu/model_saved/model_full.pth")
    graph.gcn.eval()
    accuracy = Accuracy()
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))
    count=0
    correct_sim=0
    
    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        
        for index,record in enumerate(record_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            # print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds=num_rounds,ix=index,bs=eval_batch_size,dir=dir)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs,similarities = zip(*raw_results)
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record,similarity in zip(raw_answers, record_batch,similarities):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            if answer == correct_answer:
                correct_sim+=similarity
            accuracy.print()
            count+=similarity
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
    print(count/limit_questions)
    accuracy.print()
    print(f"Correct_sim: {correct_sim/accuracy._num_correct}")
    print(f"false_sim: {(count-correct_sim)/(accuracy._num_total-accuracy._num_correct)}")
    print("Done!")
    # if accuracy.get()>acc:
    #     torch.save(graph.gcn, os.path.join(save_dir, "model_full.pth"))
    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)

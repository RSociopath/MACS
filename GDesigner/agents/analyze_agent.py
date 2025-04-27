from typing import List,Any,Dict
import re,os
import torch
import torch.nn.functional as F
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.search.wiki import search_wiki_main
from GDesigner.llm.profile_embedding import get_sentence_embedding,get_sim_embedding



    
def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None =None, role:str = None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""
        self.sim={}
        
    async def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """              
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        spatial_str = ""
        temporal_str = ""
        st=""
        for id, info in spatial_info.items():
            if self.role == 'Wiki Searcher' and info['role']=='Knowlegable Expert':
                queries = find_strings_between_pluses(info['output'])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
            st+=f"{info['output']}"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
            
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        
        return system_prompt, user_prompt,st
        
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], ix: int, bs: int, dir:str,**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        def compute_sim(S_i, S_j, sigma=2):
            cosine_similarity = torch.dot(F.normalize(S_i,p=2,dim=0), F.normalize(S_j,p=2,dim=0)) / (torch.norm(S_i, p=2) * torch.norm(S_j, p=2))
            euclidean_distance_squared = torch.norm(S_i - S_j, p=2) ** 2
            C_ij = cosine_similarity * torch.exp(-euclidean_distance_squared / (sigma ** 2))
            return C_ij
        async def get_sum(sentence):
            message = [{'role': 'system', 'content': f"Please read the input content carefully and tell me from two clear sections:\n\n1. What has been done – Describe the actions or progress that have already been completed.\n2. What to do next – Suggest the next steps, plans, or recommendations based on the current situation.\n\nMake the summary concise, structured, and easy to follow."
           f"What I send you is the content you need to analyze."
}, {'role': 'user', 'content': sentence}]
            response = await self.llm.agen(message)
            return response
        system_prompt, user_prompt,spatial_str = await self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        if len(spatial_str) > 0:
            spatial_str=await get_sum(spatial_str)
            response=await get_sum(response)
            s_i=get_sim_embedding(spatial_str)
            s_j=get_sim_embedding(response)
            sim=compute_sim(s_i,s_j)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        
        # 定义输出目录和文件名
        output_dir = f"/home/teachhu/wyh/{dir}"
        output_file = os.path.join(output_dir, f"output_{ix}.txt")
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 将内容写入文件
        with open(output_file, 'a') as f:
            f.write(f"################system prompt:{system_prompt}\n")
            f.write(f"################user prompt:{user_prompt}\n")
            f.write(f"################{self.id}：response:{response}\n")
        if len(spatial_str) > 0:
            with open(output_file, 'a') as f:
                f.write(f"################ similarity:{sim}\n")
            self.sim[ix] = sim
        return response
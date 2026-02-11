import json
import logging
import random
import os
import pathlib
import pandas as pd
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
)
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from modeling_qwen_TtT import Qwen2ForARDiffLM


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    unmasked_audio_prob: Optional[float] = field(default=0.1)
    prefix_preservation_ratio: Optional[float] = field(default=0.3)
    quad_span_truncation_prob: Optional[float] = field(default=0.5)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def truncate_last_quad_span(content: str, tokenizer: transformers.PreTrainedTokenizer, truncation_prob: float = 0.5) -> tuple[str, bool]:
    """
    随机截断最后一个被<|begin_of_quad|>和<|end_of_audio_span_last|>包裹的audio span
    
    Args:
        content: 原始内容字符串
        tokenizer: tokenizer实例  
        truncation_prob: 截断概率，默认50%的概率进行截断
    
    Returns:
        tuple: (处理后的内容字符串, 是否进行了截断)
    """
    import random
    
    # 如果随机数大于截断概率，不进行截断
    if random.random() > truncation_prob:
        return content, False
    
    # 查找最后一个<|begin_of_quad|>的位置
    begin_quad_token = "<|begin_of_quad|>"
    end_span_token = "<|end_of_audio_span_last|>"
    
    last_begin_pos = content.rfind(begin_quad_token)
    if last_begin_pos == -1:
        return content, False  # 没有找到quad span，返回原内容
    
    # 查找对应的<|end_of_audio_span_last|>
    end_pos = content.find(end_span_token, last_begin_pos)
    if end_pos == -1:
        return content, False  # 没有找到结束标记，返回原内容
    
    # 提取quad span内容（不包括开始和结束标记）
    quad_span_start = last_begin_pos + len(begin_quad_token)
    quad_span_content = content[quad_span_start:end_pos]
    
    # 将quad span内容tokenize以便精确截断
    quad_tokens = tokenizer.encode(quad_span_content, add_special_tokens=False)
    
    if len(quad_tokens) <= 10:
        return content, False  # 如果内容太短，不进行截断
    
    # 随机选择截断位置（保留至少30%的内容，最多截断70%）
    min_keep_tokens = max(int(len(quad_tokens) * 0.3), 5)
    max_keep_tokens = int(len(quad_tokens) * 0.9)
    
    if min_keep_tokens >= max_keep_tokens:
        return content, False  # 如果范围不合理，不截断
    
    keep_tokens = random.randint(min_keep_tokens, max_keep_tokens)
    truncated_tokens = quad_tokens[:keep_tokens]
    
    # 将truncated tokens转回文本
    truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
    
    # 重构完整内容 - 截断后不包含结束标记，因为序列未完成
    new_content = (
        content[:last_begin_pos] +  # quad span之前的内容
        begin_quad_token +           # 开始标记
        truncated_content            # 截断后的内容（不添加结束标记）
    )
    
    return new_content, True


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    quad_span_truncation_prob: float = 0.5,
) -> Dict:
    roles = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant",
        "system": "system\n",
    }

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer("\n").input_ids
    _system = (
        tokenizer("system").input_ids + nl_tokens
    )  # this can be changed to formal system

    # Apply prompt templates
    input_ids, targets = [], []
    prompt_lengths = []
    for i, source in enumerate(messages):
        for _, source_ in enumerate(source):
            if source_["role"] == "system":
                system_message = source_["content"]
                break
            else:
                system_message = ""

        if roles[source[0]["role"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        prompt_length = 0
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        prompt_length += len(system)
        target += [IGNORE_TOKEN_ID] * len(system)

        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            
            # 对assistant的回复进行最后一个quad span的随机截断
            sentence_content = sentence["content"]
            is_truncated = False
            if role == "<|im_start|>assistant":
                sentence_content, is_truncated = truncate_last_quad_span(sentence_content, tokenizer, quad_span_truncation_prob)
            
            # 根据是否截断来决定是否添加结束标记
            if is_truncated:
                # 截断后不添加im_end和nl_tokens，因为序列未完成
                _input_id = (
                    tokenizer(role).input_ids
                    + nl_tokens
                    + tokenizer(sentence_content).input_ids
                )
            else:
                # 正常情况下添加结束标记
                _input_id = (
                    tokenizer(role).input_ids
                    + nl_tokens
                    + tokenizer(sentence_content).input_ids
                    + [im_end]
                    + nl_tokens
                )
            
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
                prompt_length += len(_input_id)
            elif role == "<|im_start|>assistant":
                if is_truncated:
                    # 截断情况下的target处理
                    _target = (
                        [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1)
                        + _input_id[len(tokenizer(role).input_ids) + 1 :]
                    )
                else:
                    # 正常情况下的target处理
                    _target = (
                        [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1)
                        + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                        + [im_end]
                        + nl_tokens
                    )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        prompt_lengths.append(prompt_length)
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask, prompt_lengths=prompt_lengths
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, quad_span_truncation_prob: float = 0.5
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        
        # 分别处理messages和text字段的数据
        messages_data = []
        text_data = []
        
        for example in raw_data:
            if "messages" in example:
                messages_data.append(example)
            elif "text" in example:
                text_data.append(example)
        
        # 处理messages字段的数据
        if messages_data:
            messages = [example["messages"] for example in messages_data]
            messages_dict = preprocess(messages, tokenizer, max_len, quad_span_truncation_prob)
        
        # 处理text字段的数据
        if text_data:
            text_input_ids = []
            text_target_ids = []
            text_attention_mask = []
            text_prompt_lengths = []
            
            for example in text_data:
                # 直接tokenize文本
                tokens = tokenizer.encode(example["text"], add_special_tokens=False, max_length=max_len, truncation=True)
                
                # padding到最大长度
                if len(tokens) < max_len:
                    tokens = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
                
                # input_ids和target_ids是一样的
                input_ids = torch.tensor(tokens, dtype=torch.long)
                target_ids = input_ids.clone()
                
                # 将padding token对应的target设为IGNORE_TOKEN_ID
                target_ids[input_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
                
                # attention_mask是不等于padding token的地方
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                
                # prompt_lengths设为0
                prompt_lengths = 0
                
                text_input_ids.append(input_ids)
                text_target_ids.append(target_ids)
                text_attention_mask.append(attention_mask)
                text_prompt_lengths.append(prompt_lengths)
            
            # 转换为tensor
            text_prompt_lengths = torch.tensor(text_prompt_lengths, dtype=torch.long)
        
        # 创建索引列表来保持原始顺序
        all_indices = []
        if messages_data:
            all_indices.extend([('messages', i) for i in range(len(messages_data))])
        if text_data:
            all_indices.extend([('text', i) for i in range(len(text_data))])
        
        # 随机打乱索引
        random.shuffle(all_indices)
        
        # 根据打乱后的索引重新组织数据
        final_input_ids = []
        final_target_ids = []
        final_attention_mask = []
        final_prompt_lengths = []
        
        for data_type, idx in all_indices:
            if data_type == 'messages':
                final_input_ids.append(messages_dict["input_ids"][idx])
                final_target_ids.append(messages_dict["target_ids"][idx])
                final_attention_mask.append(messages_dict["attention_mask"][idx])
                final_prompt_lengths.append(messages_dict["prompt_lengths"][idx])
            else:  # text
                final_input_ids.append(text_input_ids[idx])
                final_target_ids.append(text_target_ids[idx])
                final_attention_mask.append(text_attention_mask[idx])
                final_prompt_lengths.append(text_prompt_lengths[idx])
        
        # 转换为tensor
        self.input_ids = torch.stack(final_input_ids)
        self.target_ids = torch.stack(final_target_ids)
        self.attention_mask = torch.stack(final_attention_mask)
        self.prompt_lengths = torch.stack(final_prompt_lengths)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
            prompt_lengths=self.prompt_lengths[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, quad_span_truncation_prob: float = 0.5
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.quad_span_truncation_prob = quad_span_truncation_prob

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        raw_example = self.raw_data[i]
        
        # 处理不同的数据格式
        if "text" in raw_example:
            # 处理纯文本数据
            tokens = self.tokenizer.encode(
                raw_example["text"], 
                add_special_tokens=False, 
                max_length=self.max_len, 
                truncation=True
            )
            
            # padding到最大长度
            if len(tokens) < self.max_len:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            
            # input_ids和target_ids是一样的
            input_ids = torch.tensor(tokens, dtype=torch.long)
            target_ids = input_ids.clone()
            
            # 将padding token对应的target设为IGNORE_TOKEN_ID
            target_ids[input_ids == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            
            # attention_mask是不等于padding token的地方
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
            # prompt_lengths设为0
            prompt_lengths = torch.tensor(0, dtype=torch.long)
            
            ret = dict(
                input_ids=input_ids,
                labels=target_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
            )
        else:
            # 处理对话数据
            # 先进行数据格式转换
            if "conversations" in raw_example:
                converted_example = data_convert(raw_example, raw_example.get("data_type", "IT"))
                messages = converted_example["conversations"]
            elif "messages" in raw_example:
                messages = raw_example["messages"]
            else:
                raise ValueError(f"Unknown data format for example {i}: {raw_example.keys()}")
            
            # 调用preprocess函数处理对话数据
            processed = preprocess(
                [messages],
                self.tokenizer,
                self.max_len,
                self.quad_span_truncation_prob,
            )
            ret = dict(
                input_ids=processed["input_ids"][0],
                labels=processed["target_ids"][0],
                attention_mask=processed["attention_mask"][0],
                prompt_lengths=processed["prompt_lengths"][0],
            )
        
        self.cached_data_dict[i] = ret
        return ret


def sample_jsonl(path, sample_ratio):
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    random.shuffle(data)  # 随机打乱
    data = data[: int(len(data) * sample_ratio)]  # 取样
    return data


def data_convert(data, data_type):
    if "prompt" in data and "response" in data:
        conversation = [
            {"from": "human", "value": data["prompt"]},
            {"from": "gpt", "value": data["response"]},
        ]
        data["conversations"] = conversation
        data["data_type"] = data_type
        return data
    else:
        new_data = []
        for one_dict in data["conversations"]:
            if "from" in one_dict:
                if one_dict["from"] == "human":
                    new_data.append({"role": "user", "content": one_dict["value"]})
                elif one_dict["from"] == "gpt":
                    new_data.append(
                        {
                            "role": "assistant",
                            "content": one_dict["value"],
                        }
                    )
                elif one_dict["from"] == "system":
                    new_data.append(
                        {
                            "role": "system",
                            "content": one_dict["value"],
                        }
                    )
            else:
                new_data.append(one_dict)
        data["conversations"] = new_data
        data["data_type"] = data_type
        return data


def load_parquet_data(path, sample_ratio):
    """Load data from parquet file"""
    df = pd.read_parquet(path)
    data = df.to_dict('records')
    random.shuffle(data)
    data = data[:int(len(data) * sample_ratio)]
    return data


def load_one_data(one_data):
    path = one_data["path"]
    sample_ratio = float(one_data["sample_ratio"])
    if sample_ratio == 0:
        return []
    
    filetype = path.split(".")[-1]
    if filetype == "json":
        with open(path, "r") as f:
            one_data = json.load(f)
        random.shuffle(one_data)
        one_data = one_data[:int(len(one_data) * sample_ratio)]
    elif filetype == "jsonl":
        one_data = sample_jsonl(path, sample_ratio)
    elif filetype == "parquet":
        one_data = load_parquet_data(path, sample_ratio)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")
    
    print(f"{path} has {len(one_data)} data, sample ratio {sample_ratio}")
    return one_data


def load_all_data(config_path):
    """Load data from configuration file or direct parquet path"""
    if config_path.endswith('.parquet'):
        # Direct parquet file loading
        rank0_print(f"Loading data directly from parquet file: {config_path}")
        df = pd.read_parquet(config_path)
        
        # Convert to list of records
        raw_data = []
        for _, row in df.iterrows():
            if 'messages' in row and row['messages'] is not None:
                # Extract messages field which contains the actual conversation data
                messages = row['messages']
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except:
                        continue
                elif isinstance(messages, list):
                    pass  # Already a list
                else:
                    continue
                    
                # Convert to expected format
                conversation_data = {
                    "conversations": messages,
                    "data_type": "IT"  # Default to instruction tuning format
                }
                raw_data.append(conversation_data)
        
        rank0_print(f"Loaded {len(raw_data)} conversations from parquet file")
        return raw_data
    else:
        # Configuration file loading (existing logic)
        data_sources = json.load(open(config_path, "r"))
        raw_data = []
        for one_data in data_sources:
            one_data = load_one_data(one_data)
            raw_data += one_data
        print("total data:", len(raw_data))
        return raw_data


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
    quad_span_truncation_prob: float = 0.5,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    raw_data = load_all_data(data_args.data_path)
    random.seed(42)
    random.shuffle(raw_data)
    train_dataset = dataset_cls(raw_data, tokenizer=tokenizer, max_len=max_len, quad_span_truncation_prob=quad_span_truncation_prob)

    if data_args.eval_data_path:
        eval_data = load_all_data(data_args.eval_data_path)
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len, quad_span_truncation_prob=quad_span_truncation_prob)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": False,  # 强制设置为False以避免Zero2模式下的NaN问题
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    
    # 确保mask token存在
    mask_token_str = "<|mask_token|>"
    if mask_token_str not in tokenizer.get_vocab():
        tokenizer.add_tokens([mask_token_str], special_tokens=True)
        rank0_print(f"Added mask token: {mask_token_str}")
    else:
        rank0_print(f"Mask token already exists: {mask_token_str}")
    
    # 确保end of audio token存在，需要学习，代表着diffusion阶段的结束
    end_of_audio_token_str = "<|end_of_audio_new|>"
    if end_of_audio_token_str not in tokenizer.get_vocab():
        tokenizer.add_tokens([end_of_audio_token_str], special_tokens=True)
        rank0_print(f"Added end of audio token: {end_of_audio_token_str}")
    else:
        rank0_print(f"End of audio token already exists: {end_of_audio_token_str}")
    
    end_of_audio_final_patch = "<|end_of_audio_span_last|>"
    if end_of_audio_final_patch not in tokenizer.get_vocab():
        tokenizer.add_tokens([end_of_audio_final_patch], special_tokens=True)
        rank0_print(f"Added end of audio final patch token: {end_of_audio_final_patch}")
    else:
        rank0_print(f"End of audio final patch token already exists: {end_of_audio_final_patch}")

    # Load model configuration
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False
    
    # Add mask token configuration
    config.mask_token = tokenizer.convert_tokens_to_ids(mask_token_str)
    config.audio_token_id_start = tokenizer.convert_tokens_to_ids("<|audio|>")
    config.unmasked_audio_prob = model_args.unmasked_audio_prob
    config.prefix_preservation_ratio = model_args.prefix_preservation_ratio

    # Load model - use mask token model instead of AutoModelForCausalLM
    model = Qwen2ForARDiffLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None
        ),
        **model_load_kwargs,
    )

    # Resize token embeddings if necessary
    if len(tokenizer) > model.config.vocab_size:
        _temp = model.config.vocab_size
        model.resize_token_embeddings(len(tokenizer))
        rank0_print(f"Resized token embeddings from {_temp} to {len(tokenizer)}")

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length, quad_span_truncation_prob=model_args.quad_span_truncation_prob
    )

    # Start trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # Resume from checkpoint logic
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()

import torch
import re
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random
import time
import serpapi
import math
import requests

@dataclass
class UserGenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    llm_ip: str = None
    user_ip: str = None
    temperature: float = 0.8
    topk: int = 5
    user_mode: str = 'user'
    end_threshold: float = 0.5
    start_threshold: float = 0.5



def ask_llm(ip_list_raw, prompt, temperature):
    ip_list = ip_list_raw.split(',')
    while(1):
        try:
            ip = random.choice(ip_list)
            openai_api_key = "EMPTY"
            openai_api_base = f"http://{ip}:8000/v1"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )

            # Prepare the content list
            content = [{"type": "text", "text": prompt}]

            chat_response = client.chat.completions.create(
                model='',
                max_tokens=600,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": content
                    },
                ],
            )

            return chat_response.choices[0].message.content
        except:
            continue

import random

def user_simulate_sft_feedback(ip, temperature, think_content, problem, ground_truth, gt_threshold):
    """
    用于生成SFT训练数据的用户反馈模拟（训练一个专门的user-simulator模型）。
    用户不会基于正确答案判断，而是根据直觉判断推理过程是否合理。
    如果用户认为思考过程合理，则不输出反馈（空字符串）。
    """
    prob = random.random()
    if prob > gt_threshold:
        # 有反馈：对不合理或不确定的思考进行提示
        prompt = f"""You are a user simulator observing the model's reasoning.

The model is trying to answer the following problem:
"{problem}"
The ground truth answer (for your reference only) is "{ground_truth}".

You should act as a user who does **not** know the correct answer,
but can intuitively detect if the reasoning looks wrong, confusing, or incomplete.

If you think the reasoning has an issue, express your intuition in plain text.
If the reasoning looks fine, output nothing.

Example 1:
<think> I think the sun moves around the Earth. </think>
That sounds incorrect. Isn’t it the other way around?

Example 2:
<think> The capital of France is Paris. </think>
(no output)

Now complete the task:
<think>
{think_content}
</think>

Output only one of the following:
1. A single feedback line
2. Nothing (if the reasoning looks fine)
"""
    else:
        # 多数情况为轻微或无反馈
        prompt = f"""You are a user simulator observing the model's reasoning.

The model is trying to answer the following problem:
"{problem}"
The ground truth answer (for your reference only) is "{ground_truth}".

You do **not** know the answer yourself. You only judge the reasoning intuitively.
If it looks fine, output nothing.
If it seems slightly uncertain or missing a detail, you can provide brief encouragement or a hint.

Example:
<think> The Eiffel Tower is located in Paris. </think>
(no output)

<think> Maybe the Great Wall is somewhere in Asia, I think Korea? </think>
That seems a bit uncertain — maybe double-check which country it’s in.

<think>
{think_content}
</think>

Output only:
Either nothing, or one feedback line.
"""

    feedback = ask_llm(ip, prompt, temperature).strip()
    feedback = feedback.replace('\n\n', '\n').strip()
    return feedback


def user_simulate_prompt_feedback(ip, temperature, think_content, problem, ground_truth, gt_threshold):
    """
    使用现有LLM通过prompt实时模拟用户反馈。
    包含两种模式：
    1. 清晰反馈（clear feedback）：指出具体错误或逻辑漏洞；
    2. 模糊反馈（vague feedback）：仅表达模糊的不确定感。
    当用户认为思考过程合理时，不输出任何反馈。
    """
    prob = random.random()

    if prob > gt_threshold:
        # 清晰反馈模式
        prompt = f"""You are a simulated interactive user carefully reviewing the model’s reasoning.

The question is: "{problem}"

You don't know the correct answer yourself,
but you can clearly identify flaws, missing reasoning steps, or contradictions in the reasoning.

If you find a problem, write a **specific and clear feedback** in plain text, explaining what seems wrong or unclear.
If the reasoning looks fine, output nothing.

Example:
<think> The Earth is flat because I can’t see curvature. </think>
The reasoning is incorrect. You ignored scientific evidence that the Earth is spherical.

<think>
{think_content}
</think>

Output only:
Either nothing, or one feedback line.
"""
    else:
        # 模糊反馈模式
        prompt = f"""You are a simulated interactive user casually reviewing the model’s reasoning.

The question is: "{problem}"

You don’t know the correct answer, and you can’t pinpoint exact errors,
but you can express if the reasoning feels uncertain, incomplete, or confusing.

If you feel something is slightly off, write a **vague feedback** such as:
“Not fully convinced,” “Seems incomplete,” or “Feels uncertain.”
If the reasoning seems okay, output nothing.

Example:
<think> The Earth is flat because I can’t see curvature. </think>
Hmm, this feels a bit doubtful.

<think>
{think_content}
</think>

Output only:
Either nothing, or one feedback line.
"""

    feedback = ask_llm(ip, prompt, temperature).strip()
    feedback = feedback.replace('\n\n', '\n').strip()
    return feedback


def user_simulate_prompt_inter_feedback(ip, temperature, think_content, problem, ground_truth, inter_threshold=0.7):
    """
    模拟用户是否选择中断模型推理。
    当模型出现明显错误、无意义循环、或偏离题意时，用户可能中断推理。
    返回：
        "<user> interrupt </user>" 表示应中断；
        "<user> continue </user>" 表示继续推理。
    """
    prob = random.random()

    # 高于阈值 => 触发中断评估
    if prob > inter_threshold:
        prompt = f"""You are a simulated interactive user monitoring the model’s reasoning process.

The question is: "{problem}"
The model's current reasoning is:
<think>
{think_content}
</think>

You are observing the reasoning in real time.
You should decide whether to INTERRUPT the reasoning based on the following intuition:

Interrupt the reasoning if:
- It is obviously wrong or nonsensical,
- It goes in circles without progress,
- It contradicts the question,
- It drifts away from the topic.

Otherwise, let the model continue.

Examples:
<think> The answer must be 2+2=5 because... </think>
<user> interrupt </user>

<think> The capital of France is Paris. Therefore... </think>
<user> continue </user>

Output exactly one of the following tags:
<user> interrupt </user>
or
<user> continue </user>
"""
    else:
        # 多数情况保持继续
        prompt = f"""You are a simulated user monitoring the reasoning of a model for the question:
"{problem}"

You will decide whether the reasoning should be interrupted.
If it seems reasonable, coherent, or still progressing logically, choose continue.
If it seems nonsense, repetitive, or stuck, choose interrupt.

<think>
{think_content}
</think>

Output exactly one of:
<user> interrupt </user>
<user> continue </user>
"""

    feedback = ask_llm(ip, prompt, temperature).strip()
    feedback = feedback.replace('\n\n', '\n').strip()

    # 保证输出标准化
    if "<user> interrupt </user>" in feedback.lower():
        return "<user> interrupt </user>"
    else:
        return "<user> continue </user>"


class UserLLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: UserGenerationConfig,

        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at user operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</user>')[0] + '</user>'
                 if '</user>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(self,
                prompt: torch.Tensor,
                prompt_with_mask: torch.Tensor,
                response: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, user_mode, current_step, total_steps, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        trajectory_turns = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_user_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            gt_threshold = self.dynamic_threshold(current_step, total_steps, step + 1, self.config.max_turns + 1)
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_user = self.execute_predictions(
                responses_str, gen_batch.non_tensor_batch['question'], gen_batch.non_tensor_batch['golden_answers'], user_mode, gt_threshold, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_user_stats += torch.tensor(is_user, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

            for idx in range(len(dones)):
                if trajectory_turns[idx] == 0 and dones[idx] == 1:
                    trajectory_turns[idx] = step + 1

        # final LLM rollout
        if active_mask.sum():
            gt_threshold = self.dynamic_threshold(current_step, total_steps, self.config.max_turns + 1, self.config.max_turns + 1)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_user = self.execute_predictions(
                responses_str, gen_batch.non_tensor_batch['question'], gen_batch.non_tensor_batch['golden_answers'], user_mode, gt_threshold, active_mask
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_user_stats += torch.tensor(is_user, dtype=torch.int)


            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['valid_user_stats'] = valid_user_stats.tolist()

            # 记录剩余活跃样本的完成轮数
            for idx in range(len(dones)):
                if trajectory_turns[idx] == 0:
                    trajectory_turns[idx] = step + 2

        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print("Interaction Turns Statistics:")
        for turns in range(1, self.config.max_turns + 2):
            count = (torch.tensor(trajectory_turns) == turns).sum().item()
            print(f"Finish at the {turns}-th turn: {count}")

        return self._compose_final_output(original_left_side, original_right_side, meta_info), trajectory_turns

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions, problem, ground_truth, user_mode, gt_threshold, active_mask=None, do_user=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        # cur_actions, contents = self.postprocess_predictions(predictions)
        nums = random.sample(range(0, len(predictions)), max(int(len(predictions) * 0.1), 1))
        nums.sort()
        cur_actions = ['user' if i in nums else None for i in range(len(predictions))]
        contents = predictions

        next_obs, dones, valid_action, is_user = [], [], [], []
        user_queries = [content for action, content in zip(cur_actions, contents) if action == 'user']

        if do_user and user_queries:
            # Step 3: 批量判断这些样本是否需要中断
            user_inters = self.batch_user(
                user_queries, problem, ground_truth, 'simulate_prompt_inter', gt_threshold
            )

            # Step 4: 根据 user_inters 的结果更新 cur_actions
            inter_ptr = 0
            for i, action in enumerate(cur_actions):
                if action == 'user':
                    feedback = user_inters[inter_ptr].strip().lower()
                    inter_ptr += 1
                    if "<user> interrupt </user>" in feedback:
                        cur_actions[i] = "user"

            # Step 5: 再次筛选未中断的样本（继续进入 user_mode）
            user_queries_active = [contents[i] for i, a in enumerate(cur_actions) if a == "user"]

            if user_queries_active:
                user_results = self.batch_user(
                    user_queries_active, problem, ground_truth, user_mode, gt_threshold
                )
                assert len(user_results) == len(user_queries_active)
            else:
                user_results = []

        else:
            user_inters, user_results = [], []

        # print("user_results", user_results)

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_user.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_user.append(0)
                elif action == 'user':
                    next_obs.append(f'\n\n<user>{user_results.pop(0).strip()}</user>\n\n')
                    dones.append(1)
                    valid_action.append(1)
                    is_user.append(1)
                else:
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_user.append(0)

        assert len(user_results) == 0
            
        return next_obs, dones, valid_action, is_user

    def dynamic_threshold(self, current_step, total_steps, current_turn=1, max_turns=5):
        if current_step >= total_steps:
            final_threshold = self.config.end_threshold
        else:
            progress = current_step / total_steps
            exp_base = getattr(self.config, 'exp_base', 4)
            exp_value = (math.pow(exp_base, progress) - 1) / (exp_base - 1)
            final_threshold = self.config.start_threshold + exp_value * (self.config.end_threshold - self.config.start_threshold)
        return final_threshold

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(user|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if not match or len(match.group(2).strip()) < 30:
                    content = ''
                    action = None
                else:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_user(self, predictions, problem, ground_truth, user_mode, gt_threshold) -> str:
        """
        Batchified user for queries.
        Args:
            queries: queries to call the user engine
        Returns:
            user results which is concatenated into a string
        """
        # results = self._batch_user(queries)
        all_user_result = ['No information available' for _ in range(len(predictions))]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._user, predictions[index],  problem[index], ground_truth[index][0], user_mode, gt_threshold, index) for index in range(len(predictions))]
            for future in as_completed(futures):
                try:
                    result, index = future.result()
                    all_user_result[index] = result
                except Exception as e:
                    continue

        return all_user_result



    def user_from_wiki(self, ip, query, topk=5):
        for _ in range(10):
            try:
                payload = {'query': query, 'top_k': topk}
                response = requests.post(f'http://{ip}:8002/user', json=payload)
                # import pdb; pdb.set_trace()
                doc_texts = '\n'.join([f"Doc {i + 1}: {doc['text']}" for i, doc in enumerate(response.json())])
                return doc_texts

            except Exception as e:
                time.sleep(1)
                print(e)
                continue
        return 'No information available'

    def user_from_google(self, query, topk, retry_attempt=3):
        SER_API_KEY = os.environ.get("SER_API_KEY", None)
        params = {
            "engine": "google",
            "q": query,
            "api_key": SER_API_KEY,
            "num": topk
        }

        for i in range(retry_attempt):
            try:
                user = serpapi.user(params)
                user_result = user["organic_results"]

                user_texts = []
                for item in user_result:
                    text_data = ''
                    if 'title' in item:
                        text_data += item['title']
                    if 'snippet' in item:
                        text_data += item['snippet']
                    user_texts.append(text_data)

                return '\n'.join([f"Doc {i + 1}: {doc}" for i, doc in enumerate(user_texts)])

            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                if i < retry_attempt - 1:
                    time.sleep(2)  # 等待2秒后重试
                else:
                    print("All retries failed.")
                    return 'No information available'

    def _user(self, prediction, problem, ground_truth, user_mode, gt_threshold, index):
        if user_mode == 'user':
            doc_texts = self.user(prediction, self.config.topk)
        elif user_mode == 'simulate_sft':
            doc_texts = user_simulate_sft_feedback(self.config.llm_ip, self.config.temperature, prediction, problem, ground_truth, gt_threshold)
        elif user_mode == 'simulate_prompt':
            doc_texts = user_simulate_prompt_feedback(self.config.llm_ip, self.config.temperature, prediction, problem, ground_truth, gt_threshold)
        elif user_mode == 'simulate_prompt_inter':
            doc_texts = user_simulate_prompt_inter_feedback(self.config.llm_ip, self.config.temperature, prediction, problem, ground_truth, gt_threshold)
        # print(doc_texts)
        return doc_texts, index

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
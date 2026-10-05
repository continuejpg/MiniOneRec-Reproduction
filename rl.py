from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score

os.environ['WANDB_MODE'] = 'disabled'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    subset_dir: str = "",
    subset_seq: str = "grpo_seq_10k.json",
    subset_seqtitle: str = "grpo_seqtitle_1k.json",
    seqtitle_sample: int = 10000,
    save_steps_frac: float = 0.1,
    dapo: bool = False,
    gspo: bool = False,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=seqtitle_sample)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    # ------------------------------------------------------------------
    # FIXED SUBSETS: if subset_dir is given, freeze the training data by
    # filtering each dataset down to a saved list of namespaced sample_ids.
    # This makes "baseline" and "optimized" runs train on identical data.
    # ------------------------------------------------------------------
    if subset_dir:
        def _load_ids(fname):
            if not fname:
                return None
            p = fname if os.path.isabs(fname) else os.path.join(subset_dir, fname)
            if not os.path.exists(p):
                raise FileNotFoundError(f"subset file not found: {p}")
            with open(p, "r", encoding="utf-8") as fh:
                return set(json.load(fh)["sample_ids"])

        _specs = [
            (train_data1, "seq_rec", _load_ids(subset_seq)),
            (train_data2, None, None),                      # RLTitle2SidDataset: always full
            (train_data3, "seqtitle2sid", _load_ids(subset_seqtitle)),
        ]
        _filtered, _report = [], {}
        for _ds, _task, _ids in _specs:
            _name = _task or _ds.__class__.__name__
            if _ids is None:
                _filtered.append(_ds)
                _report[_name] = len(_ds)
                continue
            _kept = [_x for _x in _ds if _x["sample_id"] in _ids]
            _missing = len(_ids) - len(_kept)
            _report[_name] = len(_kept)
            if _missing:
                print(f"[GRPO][subset] WARNING: task={_name} expected {len(_ids)} got {len(_kept)} "
                      f"({_missing} ids not found in dataset)")
            _filtered.append(_kept)
        train_datasets = _filtered
        print("=" * 70)
        print(f"[GRPO][subset] subset_dir = {subset_dir}")
        for _k, _v in _report.items():
            print(f"[GRPO][subset]   {_k:14s}: {_v}")
        print(f"[GRPO][subset]   TOTAL         : {sum(_report.values())}")
        print("=" * 70)

    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}


    # ------------------------------------------------------------------
    # Reward-target binding by STABLE sample_id (never prompt text).
    # Built from every dataset that exposes id2target; namespaces
    # (seq: / title: / seq_title:) guarantee no cross-dataset collision.
    # ------------------------------------------------------------------
    def build_id2target(datasets):
        # Build bindings from the ACTUAL samples that enter training/eval.
        # This also works after frozen-subset filtering turns a dataset into
        # a plain Python list.
        mapping, task_of = {}, {}

        for ds in datasets:
            local_mapping, local_task = {}, {}

            for x in ds:
                sid = x.get("sample_id")
                target = x.get("completion")
                task = x.get("task_type")

                if sid is None:
                    raise ValueError("sample without sample_id in reward binding")
                if target is None:
                    raise ValueError(f"sample_id {sid!r} has no completion/target")

                if sid in local_mapping and local_mapping[sid] != target:
                    raise ValueError(
                        f"sample_id {sid!r} maps to multiple targets inside one dataset"
                    )

                local_mapping[sid] = target
                if task is not None:
                    local_task[sid] = task

            overlap = set(local_mapping) & set(mapping)
            if overlap:
                raise ValueError(
                    f"sample_id COLLISION across datasets: {len(overlap)} ids, "
                    f"e.g. {sorted(overlap)[:5]}"
                )

            mapping.update(local_mapping)
            task_of.update(local_task)

        return mapping, task_of

    id2target, id2task = build_id2target(train_datasets)
    if hasattr(eval_data, "id2target"):
        _ev, _evt = build_id2target([eval_data])
        id2target.update(_ev)
        id2task.update(_evt)

    # id2history is only consumed by --reward_type sasrec (cf_reward). Built from each
    # dataset's own prompt2history map, keyed by the SAME namespaced sample_id.
    id2history = {}
    for _ds in train_datasets:
        _p2h = getattr(_ds, "prompt2history", None)
        if not _p2h:
            continue
        for _x in _ds:
            _sid = _x.get("sample_id")
            if _sid is None:
                continue
            _h = _p2h.get(_x["prompt"])
            if _h is not None:
                id2history[_sid] = _h

    print("=" * 70)
    print(f"[GRPO] reward-target binding built")
    print(f"[GRPO]   total bound sample_ids : {len(id2target)}")
    print(f"[GRPO]   train samples          : {len(train_dataset)}")
    print(f"[GRPO]   eval  samples          : {len(eval_dataset)}")
    _task_counts = {}
    for _t in id2task.values():
        _task_counts[_t] = _task_counts.get(_t, 0) + 1
    print(f"[GRPO]   task_type breakdown    : {_task_counts}")
    print("=" * 70)


    def _resolve_targets(sample_id, completions):
        """Return the ground-truth target string for every completion."""
        if sample_id is None:
            raise ValueError(
                "sample_id was not passed into the reward function. "
                "The dataset must return 'sample_id' and the column must survive "
                "into reward_kwargs. Refusing to fall back to prompt-text lookup."
            )
        targets = []
        for sid in sample_id:
            if sid not in id2target:
                raise KeyError(
                    f"sample_id {sid!r} has no bound target. "
                    f"Bound ids: {len(id2target)}. This indicates a dataset/mapping mismatch."
                )
            targets.append(id2target[sid])
        return targets

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]



    def ndcg_rule_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_generations])

            if (i + 1) % num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []

        return rewards

    def rule_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions, sample_id=None, **kwargs):
        if sample_id is None:
            raise ValueError("sample_id missing in cf_reward")
        history_list = [id2history.get(sid, "").split("::") for sid in sample_id]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    training_args = GRPOConfig(output_dir=output_dir,
                                save_steps=save_steps_frac,
                                save_total_limit=2,
                                eval_strategy="steps",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="wandb",
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)

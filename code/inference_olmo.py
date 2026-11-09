import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
import re
import wandb
from metrics import evaluate
from functools import partial
from load_custom_dataset import load_raw_data, prompt_formats
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def extract_relationship_explanation(input_string):
    # Regular expression patterns to extract relationship and explanation
    relationship_pattern = r'"relationship"\s*:\s*"([^"]+)"'
    explanation_pattern = r'"explanation"\s*:\s*"([^"]+)"'
    
    # Search for the patterns in the input string
    relationship_match = re.search(relationship_pattern, input_string)
    explanation_match = re.search(explanation_pattern, input_string)
    
    # Extract the values if the patterns are found
    relationship = relationship_match.group(1) if relationship_match else "none"
    explanation = explanation_match.group(1) if explanation_match else 'none'
    
    return relationship, explanation
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='few-shot-inference-olmo', type=str, required=False, help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='efever', type=str, required=False, help="training dataset name")
    parser.add_argument("--target_dataset_name", default='fm2', type=str, required=False, help="target dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, help="subset dataset")
    parser.add_argument("--n_shots", default= 32, type=int, required=False, help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, help="sample selection method")
    parser.add_argument("--seed",default=42, type=int, help="seed to replicate results")
    parser.add_argument("--model_name", default="allenai/OLMo-1.7-7B-hf", help = "The name of target model")
    parser.add_argument("--test_bsz",default=64, type=int, help="Batch size per GPU for training")
    parser.add_argument("--max_seq_length",default=512, type=int, help="max sequence length for model and packing of the dataset")
    parser.add_argument("--weight_decay",default=0.0, type=float, help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer")

    args = parser.parse_args()

    if args.n_shots==5000:
        relative_path="/".join((args.model_name.replace('/', "-"), args.source_dataset_name, 'nt'+str(args.n_shots)))
    else:
        relative_path = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.sample_selection, 'sub'+str(args.data_sub), 'nt'+str(args.n_shots)))
    
    model_path = '/'.join(('/home/few-shot-fact-checking/model', relative_path))
    run_name = "/".join((args.target_dataset_name, relative_path))
    result_path = '/'.join(('/home/few-shot-fact-checking/result-28/', run_name))
    
    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.target_dataset_name, args.model_name.replace('/', "-")],
           group=args.target_dataset_name,
           config = args,
           save_code = True)  
    
    peft_model_id=model_path
    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        # token= HF_TOKEN
    )

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding
    data=load_raw_data(args.target_dataset_name)
    data = data.map(partial(prompt_formats, validate_mode=True))

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    # Tokenize dataset
    tokenized_dataset = data.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(tokenized_dataset, batch_size=args.test_bsz, shuffle=False, collate_fn=data_collator)

    generated_texts = []
    for batch in data_loader:
        # Move inputs to the appropriate device
        # print(batch)
        batch = {key: val.to(device) for key, val in batch.items()}
        
        # Perform generation
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.inference_mode():
            generated_ids = model.generate(input_ids=input_ids, 
                                           attention_mask=attention_mask, 
                                           max_new_tokens=256)
                                           # do_sample=True,
                                           # top_p=0.95,)
                                           # temperature=0.9)
        
        # Decode generated texts
        texts=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts.extend(texts)
    # break
    generations_list = []
    for i in range(len(generated_texts)):
        ans_exp = generated_texts[i].replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
        # label = label2text[labels[i]]
        # answer_explanation = explanation_sep.join((label, explanation))
        generations_list.append(ans_exp)
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '/test_generation_batch.txt', 'w') as f:
        for line in generations_list:
            f.write(f"{line}\n")
            
    relationships, explanations = zip(*[extract_relationship_explanation(s) for s in generations_list])
    results, cm = evaluate(
        result_path,
        data,
        tokenizer,
        "test",
        task=args.target_dataset_name,
        labels=relationships,
        explanations=explanations,
        label_is_string=True,
    )
    
    # evaluate explanation acceptability
    # df_data = data.to_pandas()
    # exp_score = evaluate_score(result_path, df_data, args.target_dataset_name, args.test_bsz//2)
    # results['accept_score'] = exp_score
    # print(results)

    # df_cm = pd.DataFrame(cm)

    # wandb.log({"confusion_matrix": wandb.Table(dataframe=df_cm)})

    wandb.log(results)
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()

if __name__ == "__main__":
    main() 

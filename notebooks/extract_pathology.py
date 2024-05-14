import os
import random
import time
import json
import pandas as pd
import numpy as np
from jinja2 import Template

import torch
import vllm


def vllm_generate(prompts, model, sampling_params):

    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]

    return outputs


pathology_equiv = [
    ('pulmonary edema', 'interstitial edema', 'edema', 'reperfusion edema'),
    ('atelectasis', 'left atelectasis', 'right atelectasis', 'bibasilar atelectasis', 'basilar atelectasis', 'bibasal atelectasis', 'round atelectasis', 'focal atelectasis'),
    ('fibrosis', 'pulmonary fibrosis', 'postinfectious fibrosis'), # 'cystic fibrosis'
    ('fibrotic lung disease', 'fibrosing interstitial lung disease', 'fibrotic interstitial lung disease'),
    ('pneumonia', 'focal consolidation', 'focal consolidations', 'multifocal pneumonia', 'infectious pneumonia', 'bacterial pneumonia', 'mycobacterial pneumonia', 'fungal pneumonia', 'organizing pneumonia'),
    ('pleural effusion', 'effusion', 'bilateral pleural effusions', 'pleural effusions'),
    ('pneumothorax', 'hydro-pneumothorax', 'hydropneumothorax'),
    ('infectious process', 'infectious etiology', 'postinfectious process', 'infection'),
    ('inflammatory process', 'inflammatory etiology', 
        # sub-categories of inflammation
        'pneumonitis'),
    ('lymphoproliferative process', 'lymphoproliferative disease'),
    ('lymphadenopathy', 'metastatic lymphadenopathy', 'nodal metastases'),
    ('emphysema', 'subcutaneous emphysema'),
    ('pulmonary infarction', 'infarcts'),
    ('pulmonary nodules', 'pulmonary nodularity', 'pulmonary nodule'),
    ('scarring', 'interstitial scarring', 'peribronchial scarring', 'pleural scarring'),
    ('mucous plugging', 'mucous plug'),
]

confidence_equiv = [
    ('present', 'minimal', 'slightly worse', 'known'),
]


def map_to_canonical_names(output_formatted, pathology_equiv, confidence_equiv):

    if not pathology_equiv and not confidence_equiv:
        return [x.copy() for x in output_formatted]

    pathology_mapping = {}
    for names in pathology_equiv:
        representative_name = names[0]
        for x in names[1:]:
            pathology_mapping[x] = representative_name

    confidence_mapping = {}
    for names in confidence_equiv:
        representative_name = names[0]
        for x in names[1:]:
            confidence_mapping[x] = representative_name

    def map_to_canonical_names_one(pathologies):
        L = []
        for d in pathologies:
            d = d.copy()
            if d['pathology'] in pathology_mapping:
                d['pathology'] = pathology_mapping[d['pathology']]
            if d['confidence'] in confidence_mapping:
                d['confidence'] = confidence_mapping[d['confidence']]
            L.append(d)
        return L
    return [map_to_canonical_names_one(x) for x in output_formatted]



def get_pathology_confidence(outputs, pathology):
    """Parse out pathology/confidence for a single pathology. 
            - if not present, assume 'absent'
            - if there are multiple extracted entries, take priority 
                on 'present'/'absent' ones.
    """
    pred = []
    for output in outputs:
        l = list(filter(lambda x: x['pathology']==pathology, output))
        if len(l) == 0:
            x = {'reference': '', 'pathology': pathology, 'confidence': 'missing',}
        elif len(l) == 1:
            x = l[0]
        else:
            confidences = [x['confidence'] for x in l]
            if 'present' in confidences:
                x = l[confidences.index('present')]
            elif 'absent' in confidences:
                x = l[confidences.index('absent')]
            else:
                x = l[0]
        pred.append(x)
    confidences = [x['confidence'] for x in pred]
    return confidences


def metrics_xr_pathologies_iou(
        data_pred, 
        data_true,
        incorrect_examples_file=None, 
        pathology_whitelist=None,
        reduction='mean',
        ):
    """Compute iou between predicted pathologies and ground-truth pathologies"""

    acc_with_both_pred_and_label = \
        set([x['accession_number'] for x in data_true]).intersection(
        set([x['accession_number'] for x in data_pred]))
    data_pred = [x for x in data_pred if x['accession_number'] in acc_with_both_pred_and_label]
    data_true = [x for x in data_true if x['accession_number'] in acc_with_both_pred_and_label]

    data_pred = sorted(data_pred, key=lambda x: x['accession_number'])
    data_true = sorted(data_true, key=lambda x: x['accession_number'])
    assert(len(data_pred) == len(data_true))


    L = []
    incorrect_examples = []

    for i in range(len(data_pred)):
        example_true = data_true[i]
        example_pred = data_pred[i]
        acc = example_true['accession_number']
        report = example_true['report']
        pathologies_true = example_true['pathologies']
        pathologies_pred = example_pred['pathologies']

        assert(example_true['accession_number'] == example_pred['accession_number'])
        
        if pathology_whitelist:
            pathologies_true = [x for x in pathologies_true if x['pathology'] in pathology_whitelist]
            pathologies_pred = [x for x in pathologies_pred if x['pathology'] in pathology_whitelist]

        if len(pathologies_true) == 0:
            continue # skip examples with zero pathology in whitelist

        pathologies_true = set([f"{x['pathology']}, {x['confidence']}" for x in pathologies_true])
        pathologies_pred = set([f"{x['pathology']}, {x['confidence']}" for x in pathologies_pred])

        intersection = len(pathologies_true.intersection(pathologies_pred))
        union = len(pathologies_true.union(pathologies_pred))
        iou = intersection / union if union != 0 else 0
        if iou != 1:
            incorrect_examples.append({
                'i': len(incorrect_examples)+1,
                'accession_number': acc,
                'iou': iou,
                'true': sorted(list(pathologies_true)),
                'predicted': sorted(list(pathologies_pred)),
                'report': report,
                'predicted_full': example_pred['pathologies'],
            })

        L.append(iou)
        
    if incorrect_examples_file is not None:
        incorrect_examples = sorted(incorrect_examples, key=lambda x: x['iou'])
        with open(incorrect_examples_file, 'w') as f:
            json.dump(incorrect_examples, f, indent=4)
            

    if reduction == 'mean':
        return np.mean(L), incorrect_examples

    return L, incorrect_examples



def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else 'auto',
        max_model_len=args.max_model_len, # max seq len should be smaller than 29584
        enable_prefix_caching=True, # cache prefix
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization, # gpu memory utilization for model weights 
        trust_remote_code=True,
    )
    tokenizer = model.get_tokenizer()

    with open(args.icl_example_file, 'r') as f:
        examples = json.load(f)
    random.seed(0)
    random.shuffle(examples)
    print(f"#In-context examples: {len(examples)}")

    with open(args.prompt_template, 'r') as f:
        prompt_template = Template(f.read())
    prompt_prefix = prompt_template.render(examples=examples)
    print(f"Prompt prefix (inclouding icl examples) #Tokens: {len(tokenizer(prompt_prefix)['input_ids'])}")


    with open(args.test_label_file, 'r') as f:
        data_true = json.load(f)
    print(f"Test set size: {len(data_true)}")


    ## Construct prompts
    prompts = []
    for example in data_true:
        prompt = prompt_template.render(examples=examples + [{'report': example['report']}])
        prompts.append(prompt)

    if args.use_chat_template:
        prompts = [{'role': 'user', 'content': x} for x in prompts]
        if 'llama-2' in args.model_name_or_path.lower():
            from inference import create_prompt_with_llama2_chat_format
            prompts = [create_prompt_with_llama2_chat_format(x, tokenizer) for x in prompts]
        else:
            prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in prompts]


    ## Generate Outputs
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        stop=["#####"],
    )
    start = time.time()
    # prefix cached after first batch is processed, so need to call generate once to calculate the prefix and cache it
    outputs = vllm_generate(prompts[:1], model, sampling_params)
    outputs = vllm_generate(prompts[:len(prompts)], model, sampling_params)
    elapsed = time.time()-start

    print(f'model.generate() elapsed: {elapsed:.3f} s')


    ## Parse outputs
    data = []
    output_reference = True

    for i, example in enumerate(data_true):
        acc = example['accession_number']

        output = outputs[i]
        try:
            # more robust parsing, e.g., ignore rows that have not finished generating
            output_eval = [eval(x.strip()) for x in output.split('- ') if x!='' and x.count('"')%2==0]
            output_eval = [x for x in output_eval if len(x)==(3 if output_reference else 2)]
            if any([len(x)!=(3 if output_reference else 2) for x in output_eval]):
                raise
            if output_reference:
                output_formatted = [{'reference': x[0], 'pathology': x[1], 'confidence': x[2]} for x in output_eval]
            else:
                output_formatted = [{'pathology': x[0], 'confidence': x[1]} for x in output_eval]
        except:
            print(f'==== output cannot be evaluated/formatted properly [{i}] '+acc+'\n')
            print(output)
            output_eval = output
            output_formatted = []

        data.append({
            'accession_number': acc,
            'report': example['report'],
            'output': output,
            'output_formatted': output_formatted,
        })
        
    print(f"#Examples that cannot be parsed from model generation: {sum(x['output_formatted']==[] for x in data)}")
    

    ## Further processing model output & save to disk 
    df_pred = pd.DataFrame(data)
    df_true = pd.DataFrame(data_true)

    df_pred['output_canonical'] = map_to_canonical_names(df_pred['output_formatted'], pathology_equiv, confidence_equiv)
    if 'pathologies' in df_true:
        df_pred['label_canonical'] = map_to_canonical_names(df_true['pathologies'], pathology_equiv, confidence_equiv)

    model_output_file = os.path.join(args.output_dir, 'generations.json')
    df_pred.to_json(model_output_file, indent=4, orient='records')
    print(f"Save model generation/formatted outputs to {model_output_file}")


    ## Measure performance 
    pathology_whitelist = ['pleural effusion', 'pneumothorax', 'pulmonary edema', 'pneumonia', 'atelectasis',  'aspiration', 'fibrosis', 'diffuse alveolar damage']

    confidence_pred = {k: get_pathology_confidence(df_pred['output_canonical'], k) for k in pathology_whitelist}
    confidence_true = {k: get_pathology_confidence(df_pred['label_canonical'], k) for k in pathology_whitelist}

    y_pred = pd.DataFrame(confidence_pred)
    y_true = pd.DataFrame(confidence_true)

    mask = ((y_true != 'missing') | (y_pred != 'missing'))
    accuracy = (mask & (y_pred.replace('missing', 'absent') == y_true.replace('missing', 'absent'))).sum(axis=0) / mask.sum(axis=0)
    accuracy = accuracy.to_dict()
    accuracy['average'] = np.mean([x for x in accuracy.values()])

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(accuracy, f, indent=4)
        
    y_true.insert(loc=0, column='accession_number', value=df_pred['accession_number'])
    y_true.to_csv(os.path.join(args.output_dir, 'common_pathologies_predictions.csv'), index=False)
    print('Accuracy on common pathologies: ', json.dumps(accuracy, indent=4))



def get_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    ## prompts
    parser.add_argument("--test_label_file", type=str)
    parser.add_argument("--icl_example_file", type=str)
    parser.add_argument("--prompt_template", type=str)
    ## vllm args
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_model_len", type=int)
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    ## 
    parser.add_argument("--use_chat_template", default=False, action='store_true')
    parser.add_argument("--output_dir", type=str)

    return parser


if __name__ == "__main__":
    
    parser = get_argparse()
    args = parser.parse_args()
    
    main(args)
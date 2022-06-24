import jsonlines
import json
import os
from pathlib import Path

input_folder = 'examples/datasets'

output_folder = 'examples/datasets/order_mapping'

Path(output_folder).mkdir(parents=True, exist_ok=True)

all_examples = []

for fname in os.listdir(input_folder):
    if not fname.endswith('.jsonl'):
        continue
    with jsonlines.open(f'{input_folder}/{fname}', 'r') as reader:
        examples = []
        for line in reader.iter():
            for k, v in line.items():
                data = v
                key = k
            mapping = {}
            for i, var in enumerate(data['vars']):
                if var not in mapping:
                    mapping[var] = i
            for var in data['var_mentions']:
                if var not in mapping:
                    try:
                        mapping[var] = mapping[data['var_mention_to_first_var'][var]]
                    except:
                        print('---')
                        print(f'{k}: missing var_mention_to_first_var: {var} in file {fname}')
                        print(f'document: {data["document"]}')
                        print(f'vars: {data["vars"]}')
                        print(f'var_mention_to_first_var: {data["var_mention_to_first_var"]}')
                        print(f'var_mentions: {data["var_mentions"]}')
                        print('---')
                        pass
            data['order_mapping'] = mapping
            examples.append({key: data})

    with open(f'{output_folder}/{fname}', mode='w') as writer:
        for example in examples:
            writer.write(json.dumps(example) + '\n')
        print(f'Wrote {len(examples)} examples to {output_folder}/{fname}.')

    all_examples.extend(examples)

# to count unique directions
unique_directions = set()
for example in all_examples:
    for k, v in example.items():
        data = v
        key = k
    for const in data['const_declarations']:
        unique_directions.add(const['direction'])
print('')

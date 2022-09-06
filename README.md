# Subtask 2
In the starter kit, you will find the files required to train a baseline model using BART.
# Environment Setup
We have provided a Conda environment file `environment.yml`. To install:

```
conda env create -f environment.yml
conda activate myenv
```

Verify that it was installed:
```
conda env list
```

Next, copy the dataset files `train.jsonl` and `dev.jsonl` to the `data`. You can get the dataset files under [generation_task](https://github.com/nl4opt/nl4opt-competition).
# Dataset
Our dataset consists of many JSON records. The records contain these (and more) properties:
- `document` is the original document
- `vars` is the first occurrence of each variable
- `var_mentions` is each unique mention of a variable
- `var_mentions_to_first_var` is a mapping of variable mentions to their first occurrence
- `obj_declaration` is the objective declaration
- `const_declarations` contains a list of the constraint declarations
- `spans` correspond to the entities of the LP problem (i.e. the output of the first subtask).  Please note that the tokens and spans annotations are provided in the [Spacy v3 format](https://spacy.io/api)
- `tokens` correspond to the tokens found in the document
- `order_mapping` is a mapping of each variable mention to an index corresponding to its column in the canonical form

The constraints may have various types. The limit (if given) is the constant term. In the table, we denote the limit as $c$. The `terms` dictionary will attach a number to each variable, representing the constant multiplying it, which we denote here as $a_1, ..., a_n$.

| Constraint Type | Mathematical representation |
|:--------------- |:--------------------------- |
| sum             | $x + y \le c$               |
| upperbound      | $x \le c$                   |
| lowerbound      | $x \ge c$                   |
| linear          | $a_1 x + a_2 y \le c$       |
| ratio           | $x \le c (x+y)$             |
| xby             | $x \le a y$                 |
| xy              | $x \le y$                   |


For this task, there are 3 inputs:
1. Problem Description
2. Entities (tokens and spans)
3. Order mapping of variable mentions

Here is an example of a data sample for subtask 2:

```
{
   "-804075997":{
      "document":"A hotel employs cleaners and receptionists. Cleaners earn $500 per week and receptionists earn $350 per week. The hotel requires a minimum of 100 workers of whom at least 20 must be receptionists. To keep the hotel clean and running smoothly, the number of receptionists should be at least a third of the number of cleaners. The hotel wants to keep the weekly wage bill below $30000. Formulate a LP to minimize the wage bill.",
      "vars":[
         "cleaners",
         "receptionists"
      ],
      "var_mentions":[
         "cleaners",
         "receptionists",
         "Cleaners",
         "receptionists",
         "receptionists",
         "receptionists",
         "cleaners"
      ],
      "params":[
         "500",
         "350",
         "third"
      ],
      "var_mention_to_first_var":{
         "Cleaners":"cleaners",
         "receptionists":"receptionists",
         "cleaners":"cleaners"
      },
      "first_var_to_mentions":{
         "cleaners":[
            "cleaners",
            "Cleaners",
            "cleaners"
         ],
         "receptionists":[
            "receptionists",
            "receptionists",
            "receptionists",
            "receptionists"
         ]
      },
      "obj_declaration":{
         "type":"objective",
         "direction":"minimize",
         "name":"wage bill",
         "terms":{
            "Cleaners":"500",
            "receptionists":"350"
         }
      },
      "const_declarations":[
         {
            "type":"sum",
            "direction":"minimum",
            "limit":"100",
            "operator":"GREATER_OR_EQUAL"
         },
         {
            "type":"lowerbound",
            "direction":"at least",
            "limit":"20",
            "var":"receptionists",
            "operator":"GREATER_OR_EQUAL"
         },
         {
            "type":"xby",
            "x_var":"receptionists",
            "direction":"at least",
            "param":"third",
            "y_var":"cleaners",
            "operator":"GREATER_OR_EQUAL"
         },
         {
            "type":"linear",
            "direction":"below",
            "limit":"30000",
            "terms":{
               "Cleaners":"500",
               "receptionists":"350"
            },
            "operator":"LESS_OR_EQUAL"
         }
      ],
      "spans":[
         {
            "text":"cleaners",
            "start":16,
            "token_start":3,
            "token_end":3,
            "end":24,
            "type":"span",
            "label":"VAR"
         },
         {
            "text":"receptionists",
            "start":29,
            "token_start":5,
            "token_end":5,
            "end":42,
            "type":"span",
            "label":"VAR"
         },
         ...
      ],
      "tokens":[
         ...
         {
            "text":"cleaners",
            "start":16,
            "end":24,
            "id":3,
            "ws":true,
            "disabled":false
         }, 
         ...
         {
            "text":"receptionists",
            "start":29,
            "end":42,
            "id":5,
            "ws":false,
            "disabled":false
         }, 
         ...
      ],
      "_input_hash":-804075997,
      "order_mapping":{
         "cleaners":0,
         "receptionists":1,
         "Cleaners":0
      }
   }
}
```

# Training
The subfolder `./configs` should contain the configuration file for setting up the model configuration and the training hyperparameters. The configuration file `baseline.json` corresponds to the baseline model for subtask 2. To run the training with our configuration:

```
python train.py --config configs/baseline.json
```

The important parameters here are `use_copy`, `per_declaration`,  and `use_prompt`. 

- `use_copy` uses a copy mechanism that computes $P_\text{copy}$ over the input tokens. 
- `per_declaration` controls each training data sample to correspond to a single declaration of a given LP problem instead of the entire formulation (i.e. all declarations in the problem).
- `use_prompt` uses a declaration prompt to focus the generation. For example, the `<OBJ_DIR>` is used as a prompt for generating the objective declaration.

Note that beam search is available as an alternative to greedy search in decoding; however, we found that greedy search worked better.

# Testing
To evaluate the model:
```
python test.py --gpu <gpu id> --checkpoint <checkpoint.mdl> --test-file <test.jsonl>
```
More details about scoring can be found in the `notebooks` folder [here](/notebooks/demo.ipynb). For reference, our baseline model achieves a per-declaration accuracy of `Acc = 0.60` on the test set. Note however that the test set is held out and will not be shared to participants.

At testing time, the model is allowed to use as input everything in the data dictionary **except** the `"obj_declaration"` and `"const_declarations"` dictionaries. 

# Extending and Modifying the Code
You are permitted to modify the model code and training code. However, you are not permitted to modify the file `scoring.py`, which is used to evaluate and score models using the shared canonical format. In the testing code, you are allowed to modify the data loaders and parsers if you choose a different model output format than what the baseline uses. Precisely, the file `parsers.py` is for parsing model outputs into the canonical form. If you need to make a custom parser for your own model outputs, you can use your own classes or extend `parsers.py`. Make sure that your parser uses the `order_mapping` dictionary to correctly map outputs to their respective columns in the canonical output. 

You will also need to submit the conda environment file that you use so we can run your code. If you are extending the baseline code and need other Python libraries or different versions, please update the `environment.yml` by exporting your environment with `conda env export > environment.yml`.

# Submission Guidelines
We expect that your model prediction will be converted to a `CanonicalForm` object in the evaluation loop. Take a look at `test.py` to see how we score our model, and modify the code so that your model can be used. As mentioned in the Testing section: at testing time, the model is allowed to have as input everything in the data dictionary **except** the `"obj_declaration"` and `"const_declarations"` dictionaries. 

Each participating team is required to include the following in their submission folder.

1. The training and evaluation code
2. The trained model checkpoint
3. The Conda environment file environment.yml or pip environment file environment.txt.


"""
Sample data generation for CodeLACE training and evaluation.
"""

import random
import torch
from typing import List, Tuple, Dict


def generate_sample_code(language: str = 'python') -> Tuple[str, Dict[str, int]]:
    """Generate sample code with hierarchical labels."""

    if language == 'python':
        templates = [
            # Simple function
            {
                'code': '''def {func_name}({params}):
    {body}
    return {return_val}''',
                'syntactic': 0,  # function_definition
                'semantic': 0,  # simple_function
                'pragmatic': 0  # good_quality
            },
            # Loop
            {
                'code': '''for {var} in {iterable}:
    {body}''',
                'syntactic': 1,  # loop_statement
                'semantic': 1,  # iteration
                'pragmatic': 1  # moderate_quality
            },
            # Conditional
            {
                'code': '''if {condition}:
    {body}
else:
    {else_body}''',
                'syntactic': 2,  # conditional_statement
                'semantic': 2,  # branching
                'pragmatic': 0  # good_quality
            },
            # Class definition
            {
                'code': '''class {class_name}:
    def __init__(self, {params}):
        {init_body}

    def {method_name}(self):
        {method_body}''',
                'syntactic': 3,  # class_definition
                'semantic': 3,  # object_oriented
                'pragmatic': 2  # complex_quality
            }
        ]

    elif language == 'java':
        templates = [
            {
                'code': '''public {return_type} {method_name}({params}) {{
    {body}
    return {return_val};
}}''',
                'syntactic': 4,  # method_definition
                'semantic': 0,  # simple_function
                'pragmatic': 1  # moderate_quality
            },
            {
                'code': '''for (int {var} = 0; {var} < {limit}; {var}++) {{
    {body}
}}''',
                'syntactic': 1,  # loop_statement
                'semantic': 1,  # iteration
                'pragmatic': 0  # good_quality
            }
        ]

    elif language == 'javascript':
        templates = [
            {
                'code': '''function {func_name}({params}) {{
    {body}
    return {return_val};
}}''',
                'syntactic': 0,  # function_definition
                'semantic': 0,  # simple_function
                'pragmatic': 1  # moderate_quality
            },
            {
                'code': '''const {var_name} = ({params}) => {{
    {body}
    return {return_val};
}};''',
                'syntactic': 5,  # arrow_function
                'semantic': 4,  # functional_programming
                'pragmatic': 2  # complex_quality
            }
        ]

    else:  # cpp
        templates = [
            {
                'code': '''{return_type} {func_name}({params}) {{
    {body}
    return {return_val};
}}''',
                'syntactic': 4,  # method_definition
                'semantic': 0,  # simple_function
                'pragmatic': 1  # moderate_quality
            }
        ]

    # Select random template
    template = random.choice(templates)

    # Fill in template variables
    variables = {
        'func_name': random.choice(['calculate', 'process', 'handle', 'execute', 'run']),
        'method_name': random.choice(['getValue', 'setValue', 'process', 'update']),
        'class_name': random.choice(['DataProcessor', 'Calculator', 'Handler', 'Manager']),
        'var': random.choice(['i', 'j', 'item', 'element']),
        'var_name': random.choice(['result', 'data', 'value', 'output']),
        'params': random.choice(['x', 'data', 'x, y', 'value, index']),
        'body': random.choice(['    pass', '    print("Hello")', '    x += 1', '    result = x * 2']),
        'init_body': '        self.value = value',
        'method_body': '        return self.value',
        'return_val': random.choice(['x', 'result', 'True', '0']),
        'return_type': random.choice(['int', 'String', 'boolean', 'void']),
        'condition': random.choice(['x > 0', 'data is not None', 'len(items) > 0']),
        'else_body': '    pass',
        'iterable': random.choice(['range(10)', 'items', 'data_list']),
        'limit': random.choice(['10', 'n', 'size'])
    }

    # Format code
    code = template['code'].format(**variables)

    # Return code and labels
    labels = {
        'syntactic': template['syntactic'],
        'semantic': template['semantic'],
        'pragmatic': template['pragmatic']
    }

    return code, labels


def generate_dataset(num_samples: int = 1000, languages: List[str] = None) -> List[Dict]:
    """Generate dataset for training/evaluation."""
    if languages is None:
        languages = ['python', 'java', 'javascript', 'cpp']

    dataset = []

    for _ in range(num_samples):
        language = random.choice(languages)
        code, labels = generate_sample_code(language)

        dataset.append({
            'code': code,
            'language': language,
            'syntactic_label': labels['syntactic'],
            'semantic_label': labels['semantic'],
            'pragmatic_label': labels['pragmatic']
        })

    return dataset


def create_data_loaders(train_size: int = 800, val_size: int = 200, batch_size: int = 16):
    """Create data loaders for training and validation."""
    from torch.utils.data import Dataset, DataLoader
    from tokenizer import CodeTokenizer

    class CodeDataset(Dataset):
        def __init__(self, data: List[Dict], tokenizer: CodeTokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # Tokenize code
            input_ids = self.tokenizer.encode(item['code'], item['language'])
            attention_mask = self.tokenizer.create_attention_mask(input_ids)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'syntactic_label': torch.tensor(item['syntactic_label'], dtype=torch.long),
                'semantic_label': torch.tensor(item['semantic_label'], dtype=torch.long),
                'pragmatic_label': torch.tensor(item['pragmatic_label'], dtype=torch.long)
            }

    # Generate data
    train_data = generate_dataset(train_size)
    val_data = generate_dataset(val_size)

    # Create tokenizer
    tokenizer = CodeTokenizer()

    # Create datasets
    train_dataset = CodeDataset(train_data, tokenizer)
    val_dataset = CodeDataset(val_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Test data generation
if __name__ == "__main__":
    # Test sample generation
    for lang in ['python', 'java', 'javascript', 'cpp']:
        code, labels = generate_sample_code(lang)
        print(f"\n{lang.upper()} Example:")
        print(f"Code:\n{code}")
        print(f"Labels: {labels}")

    # Test dataset generation
    dataset = generate_dataset(10)
    print(f"\nGenerated dataset with {len(dataset)} samples")

    # Test data loaders
    train_loader, val_loader = create_data_loaders(100, 50, 8)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Test one batch
    for batch in train_loader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Batch syntactic_label shape: {batch['syntactic_label'].shape}")
        break
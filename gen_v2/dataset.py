import datasets

def load_dataset(dataset_name: str)->datasets.Dataset:
    huggingface_name='KAIST-IC-LAB721/'
    dataset = ''
    if dataset_name == "iemocap":
        dataset = 'IEMOCAP-Conversation'
    elif dataset_name == "emobench":
        dataset = 'EmoBench-eu'
    elif dataset_name == "dreaddit":
        dataset = 'Dreaddit'
    elif dataset_name == "cssrs":
        dataset = 'CSSRS-Suicide'
    elif dataset_name == "sdcnl":
        dataset = 'SDCNL'
    elif dataset_name == "goemotion":
        dataset = 'GoEmotion-Single'
    return datasets.load_dataset(huggingface_name+dataset)

def preprocess_data(dataset_name: str, dataset: datasets.Dataset, max_rows=200):

    if len(dataset['train']) < max_rows:
        max_rows = len(dataset['train'])

    dataset = dataset['train'][:max_rows]

    data = {'context': [], 'label': [], 'label_text': [], 'conversations': [],
            'cause': [], 'cause_text': [],'subject': [], 'label_list': []}

    if dataset_name == "iemocap":
        num_label_info = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'angry', 4: 'excited', 5: 'frustrated'}
        # num_label_info = list(num_label_info.values())

        data['context'].extend([{i: j for i, j in enumerate(sample)} for sample in dataset['conversation']])
        data['label'].extend([{i: j for i, j in enumerate(sample)} for sample in dataset['label']])
        data['label_text'].extend([{i: j for i, j in enumerate(sample)} for sample in dataset['label_text']])
        data['label_list'].extend([num_label_info] * len(dataset['conversation']))  # 각 context에 대해 label_list 추가

    elif dataset_name == "emobench":
        data['context'].extend(dataset['scenario'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend(dataset['choices'])  # 이미 각 context에 맞게 되어 있다고 가정
        data['subject'].extend(dataset['subject'])

    elif dataset_name == "emobench_cause":
        data['context'].extend(dataset['scenario'])
        data['label'].extend(dataset['cause_label'])
        data['label_text'].extend(dataset['cause_label_text'])
        data['label_list'].extend(dataset['cause_choices'])  # 이미 각 context에 맞게 되어 있다고 가정
        data['subject'].extend(dataset['subject'])

    elif dataset_name == "dreaddit":
        num_label_info = {0: 'yes', 1: 'no'}
        num_label_info = list(num_label_info.values())
        data['context'].extend(dataset['post'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend([num_label_info] * len(dataset['post']))  # 각 context에 대해 label_list 추가

    elif dataset_name == "cssrs":
        num_label_info = {0: 'supportive', 1: 'indicator', 2: 'ideation', 3: 'behavior', 4: 'attempt'}
        num_label_info = list(num_label_info.values())
        data['context'].extend(dataset['Post'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend([num_label_info] * len(dataset['Post']))  # 각 context에 대해 label_list 추가

    elif dataset_name == "sdcnl":
        num_label_info = {0: 'depression', 1: 'suicidal'}
        num_label_info = list(num_label_info.values())
        data['context'].extend(dataset['text'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend([num_label_info] * len(dataset['text']))  # 각 context에 대해 label_list 추가

    elif dataset_name == "goemotion":
        label_info = 'admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surpris, neutral'.replace(
            ' ', '')
        num_label_info = label_info.split(',')
        data['context'].extend(dataset['sentence'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend([num_label_info] * len(dataset['sentence']))  # 각 context에 대해 label_list 추가

    return data
from torch.utils.data import Dataset


class SWDADataset(Dataset):
    def __init__(self, swda, tokenizer, max_seq_length):
        self.swda = swda
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.swda)

    def __getitem__(self, index):
        # Get input text
        input_text = self.tokenizer.encode_plus(
            self.swda[index]['text'],
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Get speaker
        speaker = self.swda[index]['speaker']

        # Get context n-1
        if index > 0 and self.swda[index]['conversation_no'] == self.swda[index - 1]['conversation_no']:
            context = self.swda[index - 1]
            prev_speaker = self.swda[index - 1]['speaker']
        else:
            context = self.swda[index]
            prev_speaker = self.swda[index]['speaker']
        context = self.tokenizer.encode_plus(
            context['text'],
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get label
        label = self.swda[index]['label']

        return input_text, context, (prev_speaker, speaker), label

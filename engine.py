import importlib
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from custom_datasets import SWDADataset
from utils import clean_swda_utterance, map_speaker_to_int


def engine(model_name :str):
    # Load the SWDA dataset
    swda = load_dataset("swda")

    swda = swda.remove_columns(
        ['swda_filename', 'ptb_basename', 'transcript_index', 'act_tag', 'utterance_index',
         'subutterance_index', 'pos', 'trees', 'ptb_treenumbers', 'talk_day', 'length', 'topic_description', 'prompt',
         'from_caller', 'from_caller_sex', 'from_caller_education', 'from_caller_birth_year',
         'from_caller_dialect_area', 'to_caller', 'to_caller_sex', 'to_caller_education', 'to_caller_birth_year',
         'to_caller_dialect_area'])
    swda = swda.rename_column("damsl_act_tag", "label")
    swda = swda.rename_column("caller", "speaker")
    swda = swda.map(lambda x: clean_swda_utterance(x['text']))
    swda = swda.map(map_speaker_to_int)
    # Tokenize the input and context
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    swda = swda.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)
    # Create the train loader
    train_dataset = SWDADataset(swda['train'], tokenizer, max_seq_length=128)
    val_dataset = SWDADataset(swda['validation'], tokenizer, max_seq_length=128)
    test_dataset = SWDADataset(swda['test'], tokenizer, max_seq_length=128)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    module = importlib.import_module("models")
    model_class = getattr(module, model_name)
    model = model_class(hidden_size=256, num_classes=43)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    num_epochs = 5

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        training_loop(criterion, device, epoch, model, num_epochs, optimizer, train_accuracies, train_loader,
                      train_losses)

        val_loss = validation_loop(criterion, device, epoch, model, num_epochs, val_accuracies, val_loader, val_losses)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name+'.pt')


def training_loop(criterion, device, epoch, model, num_epochs, optimizer, train_accuracies, train_loader, train_losses):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i, (input_text, context, speakers, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        input_text = {
            'input_ids': input_text['input_ids'].squeeze(1),
            'attention_mask': input_text['attention_mask'].squeeze(1),
            'token_type_ids': input_text['token_type_ids'].squeeze(1)
        }
        context = {
            'input_ids': context['input_ids'].squeeze(1),
            'attention_mask': context['attention_mask'].squeeze(1),
            'token_type_ids': context['token_type_ids'].squeeze(1)
        }
        labels = labels.to(device)
        if model.__class__.__name__ == "GruEncoder":
            output = model(input_text, context)
        else:
            output = model(input_text, speakers, context)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * input_text["input_ids"].size(0)

        # Compute accuracy for this batch
        predicted_labels = output.argmax(dim=1)
        num_correct = (predicted_labels == labels).sum().item()
        acc = num_correct / len(labels)
        epoch_acc += acc

        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
    # Compute average loss and accuracy for epoch
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')




def validation_loop(criterion, device, epoch, model, num_epochs, val_accuracies, val_loader, val_losses):
    # Validation loop
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for input_text, context, speakers, labels in val_loader:
            input_text = {
                'input_ids': input_text['input_ids'].squeeze(1),
                'attention_mask': input_text['attention_mask'].squeeze(1),
                'token_type_ids': input_text['token_type_ids'].squeeze(1)
            }
            context = {
                'input_ids': context['input_ids'].squeeze(1),
                'attention_mask': context['attention_mask'].squeeze(1),
                'token_type_ids': context['token_type_ids'].squeeze(1)
            }
            labels = labels.to(device)
            if model.__class__.__name__ == "GruEncoder":
                output = model(input_text, context)
            else:
                output = model(input_text, speakers, context)
            loss = criterion(output, labels)

            val_loss += loss.item() * input_text["input_ids"].size(0)

            # Compute accuracy for this batch
            predicted_labels = output.argmax(dim=1)
            num_correct = (predicted_labels == labels).sum().item()
            acc = num_correct / len(labels)
            val_acc += acc
    # Compute average validation loss and accuracy
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_acc /= len(val_loader)
    val_accuracies.append(val_acc)
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    return val_loss

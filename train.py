import time 
import spacy
import random 
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm, trange

import torch 
import torch.nn as nn

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.data import train_test_split, GenerateProof, GeneratePairs, NumofEntities, NumofRelationships
from models.transformers import BertClassifier


def find_all_indices(lst, entity): 
    """ Return all indices of a given element in a list """
    ind = []
    if entity in lst: 
        for i,j in enumerate(lst):
            if j == entity: 
                ind.append(i)
    else: 
        ind = -1
    return ind  


def entities_to_numbers(pairs, name_list):
  """ Convert entities to numbers to prevent the model from inferring the relationship from the entity names themselves """
  for i in trange(len(pairs)): 
      pair = pairs[i]
      D = {} # dictionary mapping entities to numbers 
      D_count = {} # dictionary to keep track of entities that already appeared when reading the sentence
      
      doc0 = nlp(pair[0])
      doc1 = nlp(pair[1])
      
      count = 0 # number of entities per story
      for word in doc0: 
          if word.text in name_list: 
              if word.text not in D_count:
                  D_count[word.text] = 0
                  count += 1
              
      for word in doc0:
          if word.text in name_list: 
              if word.text not in D: 
                  num = random.choice(np.arange(count))
                  while num in D.values():
                      num = random.choice(np.arange(count)) # prevent from assigning different entities to a same number
                  
                  D[word.text] = num  

                  pair[0] = pair[0].replace('['+word.text+']', str(num)) # convert entity to a random number 

      for word in doc1: 
          if word.text in name_list:
              pair[1] = pair[1].replace(word.text, str(D[word.text])) # assign the same entity to an identical number 


def define_variables(data): 
    """ Define variables necessary for data preprocessing """
    # train/validation/test set split 
    train_two = data.iloc[0:5096] 
    train_three = data.iloc[5096:]

    train_two, valid_two = train_test_split(train_two)
    train_three, valid_three = train_test_split(train_three) 

    train = pd.concat([train_two,train_three])
    valid = pd.concat([valid_two,valid_three])

    train.index = [i for i in range(len(train))] 
    valid.index = [i for i in range(len(valid))]

    # generate proofs/pairs
    proof = GenerateProof(train)
    pairs = GeneratePairs(train)
    valid_proof = GenerateProof(valid)
    valid_pairs = GeneratePairs(valid)

    name_list, num_list = NumofEntities(data)
    rel_list = NumofRelationships(data)

    entities_to_numbers(pairs, name_list)
    entities_to_numbers(valid_pairs, name_list)

    max_length = 0 # compute maximum story length 
    for row in range(train["story"].size):
        doc = nlp(train["story"][row])
        max_length = max(max_length, len(doc))

    num_to_string_list = list(map(str, num_list)) # convert to strings 

    rel_dic = {} # define model output labels
    rel_dic["empty"] = 0
    for i, rel in enumerate(rel_list): 
        rel_dic[rel] = i+1

    return pairs, valid_pairs, max_length, num_to_string_list, rel_dic


def preprocessing_for_bert(data, MAX_LENGTH):
    """ Perform required preprocessing steps for pretrained BERT
    @param    data (np.array): Array of texts to be processed

    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which tokens should be attended to by the model
    @return   entity_ids (torch.Tensor): Tensor of token ids indicating a pair of entities of which relationship the model predicts 
    @return   output_ids (torch.Tensor): Tensor of token ids the model should output 
    @return   story_ids (torch.Tensor): Tensor of story ids which each input data corresponds to 
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = [] 
    attention_masks = []
    entity_ids = []
    output_ids = []
    story_ids = []

    story_count = 1 

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text = sent[0], 
            add_special_tokens = True,        # add [CLS] and [SEP] tokens to the start and end of the sentence 
            truncation = True,
            max_length = MAX_LENGTH,          
            pad_to_max_length = True,         # max length padding
            return_attention_mask = True      # return attention mask
            )
        
        entities = {} # track entities so that they appear only once 
        tokens = tokenizer.tokenize(text = sent[0])
        for ind, token in enumerate(tokens):
            if token in num_to_string_list: 
                if token not in entities: 
                    entities[token] = 1

        # update input_ids, attention_masks, entity_ids, story_ids 
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                input_ids.append(encoded_sent.get('input_ids'))
                attention_masks.append(encoded_sent.get('attention_mask'))
                entity_ids.append([i,j])
                story_ids.append([story_count])

        story_count += 1

        # update output_ids 
        entities_list = list(entities.keys())
        tokens = tokenizer.tokenize(text = sent[1])
        # find relationship given two entities and a proof sequence 
        for i in range(len(entities_list)):
            for j in range(i+1, len(entities_list)):
                indices_i = find_all_indices(tokens, entities_list[i])
                indices_j = find_all_indices(tokens, entities_list[j])

                if (indices_i == -1) or (indices_j == -1): 
                    output_ids.append([rel_dic["empty"]])
                
                else:
                    num = 0 # track entities that appear in a proof sequence but not as a pair 
                    for ind_i in indices_i:
                        if tokens[max(0, ind_i-2)] == entities_list[j]:
                            ind_j = ind_i-2
                            rel_ind = int((ind_i + ind_j)/2)
                            if tokens[rel_ind] != "sep":
                                output_ids.append([rel_dic[tokens[rel_ind]]])
                                break
                        elif tokens[min(ind_i+2, len(tokens)-1)] == entities_list[j]:
                            ind_j = ind_i+2 
                            rel_ind = int((ind_i + ind_j)/2)
                            if tokens[rel_ind] != "sep":
                                output_ids.append([rel_dic[tokens[rel_ind]]])
                                break
                        num += 1
                        if num == len(indices_i): 
                            output_ids.append([rel_dic["empty"]]) 

    # convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    entity_ids = torch.tensor(entity_ids)
    output_ids = torch.tensor(output_ids)
    story_ids = torch.tensor(story_ids)

    return input_ids, attention_masks, entity_ids, output_ids, story_ids


def train(model, criterion, optimizer, scheduler, train_dataloader, valid_dataloader=None, epochs=4, evaluation=False):
    """ Train the model """ 
    best_valid_loss = float('inf') 
    
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Acc Per Story':^16} | {'Elapsed':^9}")
        print("-"*90)
        
        t0_epoch, t0_batch = time.time(), time.time()
        
        # initialization 
        total_loss, batch_loss, batch_counts = 0,0,0
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_entity_ids, b_outputs, b_story_ids = tuple(t.to(device) for t in batch)
            
            model.zero_grad()
            
            ind1 = b_entity_ids[:,0]
            ind2 = b_entity_ids[:,1]
            logits = model(b_input_ids, b_attn_mask, ind1, ind2)
            
            targets = b_outputs.view(-1)
            loss = criterion(logits, targets)
            batch_loss += loss.item()
            total_loss += loss.item()
                    
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients 
            
            # update parameters and learning rate 
            optimizer.step()
            scheduler.step()
            
            # print loss values and elapsed time for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} |  {'-':^16} | {time_elapsed:^9.2f}")

                # initialization
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
                
        # calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*90)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            valid_loss, valid_accuracy, valid_accuracy_per_story = valid(model, criterion, valid_dataloader)
            
            # save the model with best performance on the validation set 
            if valid_loss < best_valid_loss: 
                torch.save(model.state_dict(),'./model.pt')
                best_valid_loss = valid_loss 
                
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {valid_loss:^10.6f} | {valid_accuracy:^9.2f} | {valid_accuracy_per_story:^16.2f} | {time_elapsed:^9.2f}")
            print("-"*90)
        print("\n")
    
    print("Training complete!")


def valid(model, criterion, val_dataloader):
    """ Evaluate the model's performance on the validation set """
    model.eval()

    valid_accuracy = [] # validation accuracy
    valid_loss = [] # validation loss
    preds_total = [] # store predictions with story labels 
    
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_entity_ids, b_outputs, b_story_ids = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            ind1 = b_entity_ids[:,0]
            ind2 = b_entity_ids[:,1]
            logits = model(b_input_ids, b_attn_mask, ind1, ind2)
            
        targets = b_outputs.view(-1)
        loss = criterion(logits, targets)
                      
        valid_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()
        
        accuracy = (preds == b_outputs.view(-1)).cpu().numpy().mean() * 100
        valid_accuracy.append(accuracy)
    
        preds_bool = (preds == b_outputs.view(-1)) * 1 # convert booleans to integers
        preds_total.append(torch.cat((preds_bool.unsqueeze(1), b_story_ids.view(-1).unsqueeze(1)), dim=1))
        
    preds_total = torch.cat(preds_total, dim=0)
    
    # convert to dataframe 
    preds_total_df = pd.DataFrame(preds_total).astype('int')
    preds_total_df.columns = ["Accuracy","Story"] 
    
    # calculate accuracy per story 
    valid_accuracy_per_story = preds_total_df.groupby(preds_total_df["Story"]).prod().mean()[0] * 100
    
    # compute average accuracy and loss over the entire validation set
    valid_loss = np.mean(valid_loss)
    valid_accuracy = np.mean(valid_accuracy)
    
    return valid_loss, valid_accuracy, valid_accuracy_per_story


if __name__ == "__main__":
    data = pd.read_csv('./datasets/1.2,1.3_train.csv')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define variables required for data preprocessing 
    pairs, valid_pairs, max_length, num_to_string_list, rel_dic = define_variables(data)

    # customised datasets/dataloaders
    batch_size = 16
    train_inputs, train_masks, train_entities, train_outputs, train_story_ids = preprocessing_for_bert(pairs, max_length)
    valid_inputs, valid_masks, valid_entities, valid_outputs, valid_story_ids = preprocessing_for_bert(valid_pairs, max_length)

    datasets = {
        'train' : torch.utils.data.TensorDataset(train_inputs, train_masks, train_entities, train_outputs, train_story_ids),
        'validation' : torch.utils.data.TensorDataset(valid_inputs, valid_masks, valid_entities, valid_outputs, valid_story_ids)
    }
    dataloaders = {x : torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'validation']}

    #define hyperparameters
    bert_classifier = BertClassifier(freeze_bert=False)
    bert_classifier.to(device)

    epochs = 4
    total_steps = len(dataloaders["train"]) * epochs 
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # set seed value for reproducibility
    seed_val = 42 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # model training
    train(bert_classifier, criterion, optimizer, scheduler, dataloaders["train"], dataloaders["validation"], epochs=epochs, evaluation=True)
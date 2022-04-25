import spacy
import random
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

from tqdm.notebook import tqdm, trange 

import torch 
import torch.nn.functional as F

from transformers import BertTokenizer

from utils.data import findFiles, GenerateProof, GeneratePairs, NumofEntities, NumofRelationships
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
    # generate proofs/pairs
    proof = GenerateProof(data)
    pairs = GeneratePairs(data)

    entities_to_numbers(pairs, name_list)

    max_length = 0 # compute maximum story length 
    for row in range(data["story"].size):
        doc = nlp(data["story"][row])
        max_length = max(max_length, len(doc))

    num_to_string_list = list(map(str, num_list)) # convert to strings 

    rel_dic = {} # define model output labels
    rel_dic["empty"] = 0
    for i, rel in enumerate(rel_list): 
        rel_dic[rel] = i+1

    return pairs, max_length, num_to_string_list, rel_dic


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


def evaluate(model, test_dataloader):
    """ Return predicted probabilities for each class label """
    model.eval()

    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_entity_ids, _, _ = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            ind1 = b_entity_ids[:,0]
            ind2 = b_entity_ids[:,1]
            logits = model(b_input_ids, b_attn_mask, ind1, ind2)

        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def EvaluateTest(test):
    """ Evaluate the model's performance on data of which format is unseen from the training data """ 
    test_inputs, test_masks, test_entities, test_outputs, test_story_ids = preprocessing_for_bert(test_pairs, test_max_length)

    # create customized datasets/dataloaders
    batch_size = 16
    test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_entities, test_outputs, test_story_ids)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # compute predicted probabilities on the test set
    test_probs = evaluate(bert_classifier, test_dataloader)
    test_accuracy = (torch.argmax(torch.tensor(test_probs),1) == torch.tensor(test_outputs).view(-1)).cpu().numpy().mean() * 100
    
    # convert booleans to integers
    preds_bool = ((torch.argmax(torch.tensor(test_probs),1) == torch.tensor(test_outputs).view(-1))) * 1 
    preds_total = torch.cat((preds_bool.unsqueeze(1), test_story_ids.view(-1).unsqueeze(1)), dim=1)

    # convert to dataframe 
    preds_total_df = pd.DataFrame(preds_total).astype('int')
    preds_total_df.columns = ["Accuracy","Story"] 

    # calculate accuracy per story 
    test_accuracy_per_story = preds_total_df.groupby(preds_total_df["Story"]).prod().mean()[0] * 100
                  
    return test_accuracy, test_accuracy_per_story    


if __name__ == "__main__": 
    train = pd.read_csv('./datasets/1.2,1.3_train.csv')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # global variables
    name_list, num_list = NumofEntities(train)
    rel_list = NumofRelationships(train)

    # load the model  
    bert_classifier = BertClassifier(freeze_bert=False).to(device)
    bert_classifier.load_state_dict(torch.load('./model.pt'))
    bert_classifier.eval()

    # evaluate model performance on a test set 
    test_accuracies = [] 
    test_accuracies_per_story = []
    for filename in findFiles('./datasets/1.*_test.csv'):
        test = pd.read_csv(filename)
        test_pairs, test_max_length, num_to_string_list, rel_dic = define_variables(test)
        test_accuracy, test_accuracy_per_story = EvaluateTest(test)
        
        test_accuracies.append(test_accuracy)
        test_accuracies_per_story.append(test_accuracy_per_story)

    test_accuracies_rearrange = np.zeros(9)
    for i,filename in enumerate(findFiles('./datasets/1.*_test.csv')):
        ind = filename.split('.')[1].split('_')[0]
        test_accuracies_rearrange[int(ind)-2] = test_accuracies[i]

    test_accuracies_per_story_rearrange = np.zeros(9)
    for i,filename in enumerate(findFiles('./datasets/1.*_test.csv')):
        ind = filename.split('.')[1].split('_')[0]
        test_accuracies_per_story_rearrange[int(ind)-2] = test_accuracies_per_story[i]

    plt.plot(np.arange(2,11), test_accuracies_rearrange)
    plt.plot(np.arange(2,11), test_accuracies_per_story_rearrange)
    plt.legend(["Acc : Per Pair","Acc : Per Story"])
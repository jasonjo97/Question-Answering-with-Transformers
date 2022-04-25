import torch 
import torch.nn as nn 
from transformers import BertModel 
 
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False): 
        super(BertClassifier, self).__init__()
        
        rel_types = 14
        D_in, H, D_out = 768, 50, rel_types+1 # add one for empty relationship
        
        self.bert = BertModel.from_pretrained('bert-base-uncased') 
        self.classifier = nn.Sequential(
            nn.Linear(D_in*3, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask, ind1, ind2):
        """ Classify relationship given two entities
        @param    data (np.array): Array of texts to be processed
                  input_ids (torch.Tensor): Tensor of token ids to be fed to a model
                  attention_masks (torch.Tensor): Tensor of indices specifying which tokens should be attended to by the model
                  ind1 (torch.Tensor):
                  ind2 (torch.Tensor): 
        """
        outputs = self.bert(input_ids = input_ids,
                 attention_mask = attention_mask)
        
        # extract the last hidden state of the token '[CLS]', '[Ent1]', '[Ent2]'         
        cls, ent1, ent2 = outputs[0][:,0,:], outputs[0][torch.arange(len(ind1)),ind1], outputs[0][torch.arange(len(ind2)),ind2]
        
        logits = self.classifier(torch.cat((cls, ent1, ent2), dim=1))  
    
        return logits 
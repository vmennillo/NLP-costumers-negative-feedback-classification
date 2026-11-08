# NLP-costumers-negative-feedback-classification
Classifying danish negative customers reviews from different sources into six categories by reason of complaint.

The original dataset consisted of more than 44000 danish reviews, complete of title and star rating between 1 and 5. Only reviews with a rating equal or lower than 2 were selected. 
Annotation was performed manually on a subset of only 100 reviews.
The final dataset presented some imbalance in the classes:

Categories:
- Credibility: 29
- Product quality: 30
- Customer service: 25
- Pricing: 7
- Delivery: 7
- Return/refund issues: 2


Key challenge: reviews often involve more than one category of complaint. 

The models were sourced from HuggingFace, whose pipeline includes tokenization and lowercasing.
Frameworks and libraries: Pytorch, pandas, sklearn, transformers
Models:
1. Zero-shot/Few-shot classification:
   - xlm-roberta-xnli (Natural Language Inference)
   - Llama 3.1 (8B)
2. Fine-tuning:
   - Bert-base-multilingual 
   - Danish-bert-botxo

Applied oversampling to rebalance the classes.

Results:
                       Precision  Recall  F1 Score  Accuracy
NLI Model                 0.6392  0.4125    0.3753      0.49
LLM Zero-Shot             0.3825  0.3731    0.2821      0.42
LLM Few-Shot              0.7379  0.6702    0.6487      0.66
Bert Finetune English     0.3152  0.4540    0.3711      0.48
Bert Finetune Danish      0.4074  0.4471    0.3389      0.48

Key results: the dataset is too small to expect an effective fine-tuning. Bert multilingual unexpectedly performed better than Bert Danish.





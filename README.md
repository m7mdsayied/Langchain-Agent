# Langchain-Agent
Setting local environment LangChain-agent

# Dataset: 
NileULex
Nile University's Arabic sentiment Lexicon 
NilULex v0.27: dataset contains Egyptian Arabic and Modern Standard Arabic sentiment words and their polarity.

El-Beltagy, S. R. (2016, May). Nileulex: A phrase and word level sentiment lexicon for egyptian and modern standard arabic. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16) (pp. 2900-2905).
# Components: 
- LLM Wrappers(act as Chains for Sentiment and Topic Analysis performed by both models):
  1- bert-base-arabertv02
  2- Command R+
  
- Chains:
  1- arabert_sentiment_chain
  2- arabert_topic_chain
  3- command_r_plus_sentiment_chain
  4- command_r_plus_topic_chain
  
- Indexes:
  Acess any retrieved_data by it's index or group of indexes.

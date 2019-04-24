# COV-AI-CHALLENGE-NLP-

Project Description

The COVAI Challenge project aims at programming an algorithm in Python that is able to build a consistent Chitchat. For any given question,  it may pick the answer within a list of sentences that fits the best with a context based on an utterance and on people’s characteristics. This program is elaborated though cosine similarity. This kind of algorithm is a pretty useful to ensure chatbots performance. This report sum-up the creation process of the algorithm, the choices made for the programming part, the obstacles faced, the final results and the sources.

	Libraries used 

For basic manipulation:
-numpy  
-pandas  
-argparse   
-pickle  
-io
For Vectorizer process:
-sklearn.feature_extraction.text
-sklearn.preprocessing

	Class DialogueManager
	to get the best option

Definition of __init__: Initialization of the class by defining our vectorizer: TfidfVectorizer.

Definition of stopnumber function: to remove the digit elements of the list.

Definition of load function: load of the path.

Definition of  save function: save the information. 

Definition of train function: fit the vectorizer to the sentences. 

Definition of best_with_utterance: Compute similarity between each option and utterance of a given dialogue and return the scores in a matrix.

Definition of best_with_yourpers: Compute similarity between each option and your persona sentences of a given dialogue and return the scores in a matrix.

Definition of best_with_parpers: Compute similarity between each option and partner persona sentences of a given dialogue and return the scores in a matrix.

Definition of findBest: Keep the 5 options for which the similarity with the personas are the best and from that 5 options return the one whose similarity with utterance is the best. 

Parameters:   
Analyzer of the Tfidf vectorizer: ‘word’


	LoadData

param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)

return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue is a sequence of (utterance, answer, options)


	Results

without normalization of vectors: 43%
with normalisation of vectors: 47%



	External References

Personalizing Dialogue Agents: I have a dog, do you have a pet- Zhang, Dinan, Urbanek, Szlam, Kiela, Weston

An IR baseline  - Sordoni  

supervised embedding model, Starspace  - Wu

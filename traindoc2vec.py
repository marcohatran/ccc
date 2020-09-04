import gensim 
import pandas as pd
import tqdm
def get_corpus(documents):
    corpus = []
    
    for i in range(len(documents)):
        doc = documents[i]
    
        
        words = doc.split(' ')
        tagged_document = gensim.models.doc2vec.TaggedDocument(words, [i])
        
        corpus.append(tagged_document)
        
    return corpus


df = pd.read_csv('dataset.csv', encoding="utf8", index_col="id")


content = df['content']
print(len(content))
content_list = content.values.astype('U')

train_data = get_corpus(content_list)
model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
model.build_vocab(train_data)
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("d2v.model")
print("Model Saved")
# print(range(len(content_list)))


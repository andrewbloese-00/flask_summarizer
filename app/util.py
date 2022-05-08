#text rank based summary 
from cmath import sqrt
import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.tokenize import sent_tokenize
import re
nltk.download("punkt")

stop_words = [
    'himself', 'the', 'among', "n't", 'and', 'herein', 'hers', 'hereby', 'therein', 'along', 'seems', 'during', 'done', 'sometimes', 'off', 'until', 'without', '‘ve', 'per', 'above', 'other', 'on', 'noone', 'down', 'whole', 'anyway', 'please', 'fifty', "'ll", 'do', 'whereafter', 'mine', 'have', 'meanwhile', 'no', 'name', 'amongst', 'my', 'besides', 'became', 'to', 'somewhere', 'her', 'thereafter', 'against', 'give', 'always', 'empty', 'becoming', 'eight', '’ll', 'due', 'get', 'cannot', 'around', 'seem', 'if', 'thus', 'former', 'nine', 'ca', 'me', 'even', 'take', 'really', 'could', 'before', 'using', 'others', 'yours', 'however', 'who', 'but', 'those', 'myself', '‘d', 'while', 'she', 'up', 'call', 'in', 'everyone', '‘ll', 'make', 'though', 'nowhere', 'whatever', 'side', 'hence', 'third', 'only', 'go', 'has', 'may', 'many', 'become', '‘m', 'an', 'either', 'most', 'seemed', 'namely', 'say', '’m', 'yourselves', 'not', 'much', 'would', 'out', 'whereas', 'wherein', 'we', 'afterwards', 'with', 'were', 'hereafter', 'how', 'something', 'formerly', 'it', 'very', 'hundred', 'someone', 'will', '’re', 'sixty', 'herself', 'whether', 'except', 'next', 'same', 'still', 'so', 'n’t', 'becomes', 'might', 'into', '‘re', 'anywhere', 'whence', 'more', 'within', "'re", 'because', 'quite', 'just', 'also', 'bottom', 'should', 'perhaps', 'yourself', 'thru', 're', 'otherwise', 'twelve', 'i', 'few', 'nothing', 'had', 'beyond', 'he', 'onto', 'else', 'first', 'part', 'never', 'us', 'full', 'thereby', 'serious', 'three', 'whoever', 'upon', 'him', 'a', 'your', 'doing', 'further', 'often', 'too', 'you', 'all', 'front', 'whose', 'when', '’d', 'ours', 'his', 'after', 'from', 'am', 'wherever', 'its', 'see', 'somehow', 'together', 'alone', 'almost', 'least', 'be', 'between', 'five', "'ve", 'already', 'regarding', 'four', 'via', 'last', 'is', 'behind', 'everything', 'itself', 'beforehand', 'nor', 'what', 'none', 'thence', 'did', 'they', 'anyone', 'under', 'twenty', '‘s', 'any', 'why', 'anything', 'move', 'show', "'m", 'that', 'anyhow', 'six', 'whom', 'are', 'these', 'seeming', 'whenever', 'where', 'yet', 'now', 'was', 'another', 'therefore', 'whereby', 'once', '’ve', 'neither', 'both', 'top', 'our', 'below', 'through', 'one', 'whereupon', 'throughout', 'such', '’s', 'been', 'own', 'again', 'some', 'unless', 'nevertheless', 'latterly', 'since', 'at', 'of', 'although', 'themselves', 'fifteen', 'every', 'there', 'rather', 'mostly', 'eleven', 'beside', 'various', 'than', 'ten', 'ourselves', 'put', 'them', 'towards', 'thereupon', "'s", 'as', 'back', 'whither', 'about', 'this', 'made', 'each', 'elsewhere', "'d", 'can', 'enough', 'across', 'here', 'which', 'n‘t', 'less', 'does', 'latter', 'moreover', 'indeed', 'sometime', 'amount', 'then', 'for', 'two', 'or', 'ever', 'everywhere', 'keep', 'over', 'forty', 'several', 'their', 'being', 'hereupon', 'toward', 'by', 'well', 'used', 'must', 'nobody'
]

def summarize_text( text , reduce_to ): 
    from zipfile import ZipFile
    with ZipFile("glove.6B.100d.txt.zip") as ref: 
        ref.extractall()
    #split into sentences
    sentences = sent_tokenize(text)


    #text preprocessing 
    clean_sentences = pd.Series(sentences).str.replace('[^a-zA-Z]', " ", regex=True)
    #make_lower
    clean_sentences = [s.lower() for s in clean_sentences]


    def remove_stopwords( sen ):
        result = []
        wrds = str(sen).split()
        for wrd in wrds: 
            if not wrd in stop_words:
                result.append(wrd)
        
        return " ".join(result)

    clean_sentences = [ remove_stopwords(r.split) for r in clean_sentences]


    


    #create word vectors
    word_vectors = {}
    
    

    with open('glove.6b.100d.txt',encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            word_vectors[word] = coefs


    sentence_vectors = [ ]
    for sent in clean_sentences: 
        if len(sent) != 0:
            v = sum([ word_vectors.get( word, np.zeros((100,))) for word in sent.split()])
        else:
            v = np.zeros((100,))
        
        sentence_vectors.append(v)



    from sklearn.metrics.pairwise import cosine_similarity   

    #create similarity matrix
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range( len(sentences)):
            if( i != j ):
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
            
            


    #graph 
    import networkx as nx
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    #rank sentences
    ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    summary  = []
    #select length
    n = max(int( reduce_to * len(sentences) ) , 1)
    for i in range(n):
       summary.append(ranked[i][1])

    #return summary
    return " ".join(summary)




    

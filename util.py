# coding: utf-8
import numpy as np
import MeCab 
import codecs
from gensim.models.word2vec import Word2Vec

def padding(document_list, max_len):
    
    new_document_list = []
    for doc in document_list:
        pad_line = ['pad' for i in range(max_len - len(doc))] #全ての文書の単語数を合わせる
        new_document_list.append(doc + pad_line)
    return new_document_list

def load_data(fname):
    model_path = './latest-ja-word2vec-gensim-model/word2vec.gensim.model'
    model = Word2Vec.load(model_path)


    target = [] #ラベル
    source = [] #文書ベクトル

    #文書リストを作成
    document_list = []
    target = [] #ラベル
    source = [] #文書ベクトル
    fname = './profile/profile_171216.txt'
    with codecs.open(fname, 'r', 'utf-8') as file:
    	lines = file.readlines()
    	for line in lines:
             	sample = line.strip().split(' ',  1)
             	label = sample[0]
             	target.append(label) #ラベル
             	document_list.append(sample[1].split())#文書ごとの単語リスト
    	len(document_list)

    max_len = 0
    rev_document_list = [] #未知語処理後のdocument list
    for doc in document_list: #文書リスト
        rev_doc = []
        for word in doc: #文書
                m = MeCab.Tagger('C:\Program Files (x86)\MeCab\dic\ipadic -Owakati')
                
                encoded_text = word.encode('utf-8') 
                wakati = m.parse(word)
                watati = wakati.split()
                for i in wakati: #分かち書き文書
                        try:
                           word_vec = np.array(model[i])
                           rev_doc.append(i)
                        except KeyError:
                           rev_doc.append('unknown') #未知語
        rev_document_list.append(rev_doc)
        if len(rev_doc) > max_len:
    	    max_len = len(rev_doc)
    
    #文書長をpaddingにより合わせる
    rev_document_list = padding(rev_document_list, max_len)

    print('document_list len', len(document_list))
    print('rev_document_list len', len(rev_document_list))
    
    width = 0 #各単語の次元数
    #文書の特徴ベクトル化
    for doc in rev_document_list:
        doc_vec = []
        for word in doc:
            try:
                #vec = model[word.decode('utf-8')]
                vec = model[word]
            except KeyError:
                vec = model.seeded_vector(word)
                #vec = model.word_vec(word, True)
                #vec = rand(300,1)*2-1
            doc_vec.extend(vec)
            if len(vec) > width:
                width = len(vec)
        source.append(doc_vec)

    dataset = {}
    dataset['target'] = np.array(target)    
    dataset['source'] = np.array(source)    
    
    return dataset, max_len, width


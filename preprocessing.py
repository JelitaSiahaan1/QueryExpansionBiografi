from nltk import tokenize, word_tokenize
import re
import string
from nltk.tag import CRFTagger
from nltk.corpus import stopwords

import PyPDF2
import os

import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# LIBRARY FOR RANKED RETRIEVAL
import math
from collections import OrderedDict

def allFile(location):
    document = []
    for doc in os.walk(location):
        document = doc[2]
    return document

def extractPDF(location):
    documents = allFile(location)
    allText = []
    for doc in documents:
        file = open(location+'/'+doc, 'rb')
        fileReader = PyPDF2.PdfFileReader(file)
        
        docs = ''
        pages = fileReader.numPages
        for page in range(pages):
            obj = fileReader.getPage(page)
            docs = docs + obj.extractText()
        allText.append(docs)
    return allText

def generateDocNumber(filename):
    docNum = []
    for file in filename:
        docNum.append(str(filename.index(file)))
    return docNum

# PREPROCESSING

def removePunctuation(textList):
    for i in range(len(textList)):
        for punct in string.punctuation:
            textList[i] = textList[i].replace(punct, " ")
            textList[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', textList[i], flags=re.MULTILINE)
    
    return textList

def removePunct(textList):
    for punct in string.punctuation:
        textList = textList.replace(punct, " ")
        textList = re.sub(r'^https?:\/\/.*[\r\n]*', '', textList, flags=re.MULTILINE)
    
    return textList

def caseFolding(textList):
    text = []
    for i in range(len(textList)):
        text.append(textList[i].lower())
    return text

def caseFold(textList):
    text = []
    for i in range(len(textList)):
        text.append(textList.lower())
    return text

def tokenization(list):
    trans = str.maketrans('','',string.punctuation)
    datas = []
    for kalimat in list:
        line = ''.join([i for i in kalimat if not i.isdigit()])
        line = re.sub('r ^ https ?:\/\/.*[\r\n]*','',kalimat,flags = re.MULTILINE)
        line = line.translate(trans)
        line = word_tokenize(line)
        datas.append(line)
    return datas

def token(sentence):
    token = []
    for word in CountVectorizer().build_tokenizer()(sentence):
        token.append(word)
    return token

def tokenize(textList):
    tokens = []
    for i in range(len(textList)):
        tokens.append(token(textList[i]))
    return tokens

def checkStopword(sentence, stop_words):
    sentence = [w for w in sentence if not w in stop_words]
    return sentence
   
def stopwordRemove(textList):
    stop_words = set(stopwords.words('indonesian'))
    text = []
    for i in range(len(textList)):
        text.append(checkStopword(textList[i], stop_words))
    return text

def numberRemove(textList):
    text = []
    for i in range(len(textList)):
        text.append([w for w in textList[i] if not any(j.isdigit() for j in w)])
    return text

def stemming(textList):
    stemmer = PorterStemmer()
    text = textList
    for i in range(len(textList)):
        for j in range(len(textList[i])):
            text[i][j] = stemmer.stem(text[i][j])
    return text

def sorting(textList):
    for i in range(len(textList)):
        textList[i] = sorted(textList[i])
    return textList

def getAllTerms(textList):
    terms = []
    for i in range(len(textList)):
        for j in range(len(textList[i])):
            terms.append(textList[i][j])
    return sorted(set(terms))

# INDEXING FUNCTION
def createIndex(textList, docno):
    terms = getAllTerms(textList)
    proximity = {}
    for term in terms:
        position = {}
        for n in range(len(textList)):
            if(term in textList[n]):
                position[docno[n]] = []
                for i in range(len(textList[n])):
                    if(term == textList[n][i]):
                        position[docno[n]].append(i)
        proximity[str(term)] = position
    return proximity

def exportIndex(index, filename):
    file = open(filename,'w')
    for n in index:
        file.write(str(n)+'\n')
        for o in index[n]:
            file.write('\t'+o+': ')
            for p in range(len(index[n][o])):
                file.write(str(index[n][o][p]))
                if(p<len(index[n][o])-1):
                    file.write(', ')
                else:
                    file.write('\n')
    file.close()
    return "Index's file has been successfully created."

# RANKED RETRIEVAL FUNCTION
def queryInIndex(query, index):
    result = []
    for word in query:
        if word in index:
            result.append(word)
    return result

def df(query, index):
    docFreq = {}
    for word in query:
        if word in index:
            docFreq[word] = len(index[word])
    return docFreq

def idf(df, N):
    inv = {}
    for word in df:
        inv[word] = math.log10(N/df[word])
    return inv

def tf(query, index):
    termFreq = {}
    for word in query:
        freq = {}
        if word in index:
            for i in index[word]:
                freq[i] = len(index[word][i])
        termFreq[word] = freq
    return termFreq

def tfidf(tf, idf):
    w = {}
    for word in tf:
        wtd = {}
        for doc in tf[word]:
            wtd[doc] = (1+(math.log10(tf[word][doc])))*idf[word]
        w[word] = wtd
    return w
    
def score(TFIDF):
    res = {}
    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = 0
    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = res[j]+TFIDF[i][j]
    sorted_dict = sorted(res, key=res.get, reverse=True)
    return sorted_dict
    


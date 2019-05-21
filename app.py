from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from form import MyForm
from flask import jsonify
from logging import FileHandler, WARNING
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

import preprocessing
app = Flask(__name__)

file_handler = FileHandler('errorlog.txt') #file_handler logs errors
file_handler.setLevel(WARNING)
app.config.from_object(__name__) # Config for Flask-Session

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 #max upload size is 1MB
app.config['SECRET_KEY'] = 'secret'
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])#restricts file extensions to the .txt extension
app.config['SESSION_TYPE'] = 'filesystem' #config for Flask Session, indicates will store session data in a filesystem folder
app.logger.addHandler(file_handler)
Session(app) #create Session instance

def allowed_file(filename):
    '''Checks uploaded file to make sure it is.txt'''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/cari', methods=['GET', 'POST'])
def cari():
    cari = request.form['search_input']
        
    location = 'static/biografi'
    filename = preprocessing.allFile(location) #jalan
    extracted= preprocessing.extractPDF(location) #jalan
    totalDoc = len(filename)
    documentNumber = preprocessing.generateDocNumber(filename)

    for i in range(len(filename)):
        extracted[i] = str(extracted[i].encode("utf-8"))

    # # # PREPROCESSING
    text = preprocessing.removePunctuation(extracted)
    
    text = preprocessing.caseFolding(text)
    text = preprocessing.tokenize(text)
    text = preprocessing.stopwordRemove(text)
    text = preprocessing.numberRemove(text)
    text = preprocessing.stemming(text)

    # # # GET ALL TERMS IN COLLECTION
    terms = preprocessing.getAllTerms(text)

    # # # INDEXING
    # # # index = createIndex(text,documentNumber, terms)
    index = preprocessing.createIndex(text, documentNumber)
    
    
    query = preprocessing.removePunct(cari)
    query = preprocessing.caseFold(query)
    query = preprocessing.tokenization(query)
    query = preprocessing.stopwordRemove(query)
    query = preprocessing.numberRemove(query)
    query = preprocessing.stemming(query)
    query = query[0]
    
    # Check Query In Index

    query = preprocessing.queryInIndex(query, index)

    # RANKED RETRIEVAL
    N               = totalDoc
    tfidf_list      = []

    docFrequency    = preprocessing.df(query, index)
    invDocFrequency = preprocessing.idf(docFrequency, N)
    termFrequency   = preprocessing.tf(query, index)
    TFIDF           = preprocessing.tfidf(termFrequency, invDocFrequency)
    sc              = preprocessing.score(TFIDF)

    print(len(sc))
    relevanceDocNumber = []
    count = 0

    print('Query: ', cari,'\n\n')
    print('Result: \n')
    # for i in range(5):
    #     a = documentNumber.index(sc[i])
    #     print('Document Number: ',sc[i])
    #     print(filename[a])
    #     print('-------------------------------------------\n')

    result = []
    for i in range(len(sc)):
        relevanceDocNumber.append(int(sc[i]))
        a = documentNumber.index(sc[i])
        print()
        print('==========================================================================\n')
        print('| Filename: ',filename[a],' | Document ID: ',documentNumber[a],'|','\n')
        print(extracted[a][0:1000])
        print('\n==========================================================================')
        print('\n\n\n')
        # count = count + 1
        result.append((filename[a], documentNumber[a]))
        if(i >= 5):
            break

    return render_template('home.html', a=documentNumber[a], qs=cari, fl=filename[a], hasil = result, sc=sc)

if __name__ == '__main__':
   app.run()

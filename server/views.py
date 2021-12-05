from django.shortcuts import render
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import re 
import pandas as pd

wnl = WordNetLemmatizer()
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_preprocess(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and not(word.isdigit()))
    return text

def stem_ori(sentence):
    lemmas_sent = []
    tagged_sent  = pos_tag(sentence)
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos = wordnet_pos))
    return lemmas_sent

def stem(word):
    tag  = pos_tag([word])[0]
    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
    word = wnl.lemmatize(tag[0], pos = wordnet_pos) + '_' + tag[1].lower()
    return word

def home(request):
    if 'search' in request.POST:
        mode = 'tfidf'
        search = request.POST['search']
        search_pos = stem(text_preprocess(search))
        search_ori = stem_ori([text_preprocess(search)])[0]
        df_tfidf = pd.read_csv('tfidf.csv')
        ori_df_tfidf = pd.read_csv('ori_df_tfidf.csv')
        df_docs = pd.read_csv('document.csv')
        if search_ori in ori_df_tfidf:
            df = ori_df_tfidf.sort_values(by = [search_ori], ascending = False)[:10][search_ori]
            values = df.values
            index = df.index
            docs = []
            for i in range(10):
                if values[i] == 0: break
                docs.append([df_docs[str(index[i])][0], round(values[i], 3)])
        if search_pos in df_tfidf:
            df = df_tfidf.sort_values(by = [search_pos], ascending = False)[:10][search_pos]
            values = df.values
            index = df.index
            pos_docs = []
            for i in range(10):
                if values[i] == 0: break
                pos_docs.append([df_docs[str(index[i])][1], round(values[i], 3)])
    elif 'index' in request.GET:
        mode = request.GET['index'].split('%')[0]
        search = request.GET['index'].split('%')[1]
        if not search: return render(request, 'home.html', locals())
        search_pos = stem(text_preprocess(search))
        search_ori = stem_ori([text_preprocess(search)])[0]
        if mode == 'tfidf':
            df_tf = pd.read_csv('tfidf.csv')
            ori_tf = pd.read_csv('ori_df_tfidf.csv')
            df_data = pd.read_csv('document.csv')
        elif mode == 'tfisf':
            df_tf = pd.read_csv('tfisf.csv')
            ori_tf = pd.read_csv('ori_df_tfisf.csv')
            df_data = pd.read_csv('sentences.csv')
        elif mode == 'tficf':
            df_tf = pd.read_csv('tficf.csv')
            ori_tf = pd.read_csv('ori_df_tficf.csv')
            df_data = pd.read_csv('cates.csv')
        if search_ori in ori_tf:
            df = ori_tf.sort_values(by = [search_ori], ascending = False)[:10][search_ori]
            values = df.values
            index = df.index
            docs = []
            for i in range(10):
                try:
                    if values[i] == 0: break
                    docs.append([df_data[str(index[i])][0], round(values[i], 3)])
                except:
                    continue
        if search_pos in df_tf:
            df = df_tf.sort_values(by = [search_pos], ascending = False)[search_pos]
            values = df.values
            index = df.index
            pos_docs = []
            for i in range(10):
                try:
                    if values[i] == 0: break
                    pos_docs.append([df_data[str(index[i])][1], round(values[i], 3)])
                except:
                    continue
    return render(request, 'home.html', locals())

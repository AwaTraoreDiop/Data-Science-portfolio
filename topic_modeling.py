
# %%
import os
import random
import pyodbc #to connect to sql server 
#ecriture SQL Server
import sqlalchemy
from sqlalchemy.engine import url
from sqlalchemy.engine.url import URL
from pathlib import Path
import pandas as pd
import numpy as np
import nest_asyncio #to avoid runtime issue
nest_asyncio.apply()
import re
#preprocessing
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#from nltk.stem import SnowballStemmer
#from nltk.util import ngrams
#from nltk.tokenize import word_tokenize
#nltk.download('punkt') #contents a pre-trained Punkt tokenizer
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS
import emoji
#from deep_translator import GoogleTranslator as GT # translate other languages in french
#from deep_translator import exceptions as excp
#translate foreign languages (other than french)
#! pip install -U deep-translator
import time
from googletrans import Translator
#visualisation
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
sns.set(style='darkgrid')
sns.set(font_scale=1.3)
from unidecode import unidecode
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from PIL import Image
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
#sentimental analysis
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # helps using the GPU much more efficiently than simply ignoring the info.
import spacy
import spacy_udpipe
#Topic modeling
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel# spaCy for preprocessing
from gensim.utils import simple_preprocess
import pyLDAvis.gensim_models
import pyLDAvis
import pyLDAvis.gensim_models
#from sklearn.datasets import fetch_20newsgroups
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pprint import pprint
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pyLDAvis.lda_model
# %% [markdown]
# **CONNEXION TO DWH**
# %%
# %%

# # **Data Overview**
# %%
df = pd.read_excel(r"verbatims1000.xlsx")
df.head()
# %%
df.shape
# %%
df.info()
# %% [markdown]
# **Data preprocessing**
# %%
#convert comments into str and answers into float
df['Comments'] = df['Comments'].astype(str)
df['answer'] = df['answer'].astype(float)
# %%
#create word list by splitting every comments by word
liste_mots = [re.split('\W+',doc) for doc in df['Comments']]
# %%
#create a new feature to account the number of words of each comment
df['comments_length'] = [len(doc) for doc in liste_mots]
# %%
df.head()
# %%
# Here we use a column with categorical data
fig = px.histogram(df, x="comments_length")
fig.update_layout(bargap=0.5)
fig.show()
# %%
#-------------------most usefull terms in short comments------------------#
text = str(df['Comments'].unique())
wordcloud = WordCloud().generate(text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=100, max_words=50, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 
# %%
#transform the comments in list
corpus = list(df['Comments'])
corpus[0:10]
# %%
type(corpus)
# %%
len(corpus)
# %%
#count the number of words in every doc within the corpus
i = 1
for doc in corpus:
    print(f"Le document {i} est de longueur : {len(doc)}")
    i+=1
# %%
if doc.isdigit() in corpus :
    print(doc)
else : 
    print("Tous les documents sont de type string")
# %%
#Tokenization
def comments(corpus):
    i = 1
    for doc in corpus:
        #liste_mots= word_tokenize(doc)
        liste_mots = re.split('\W+',doc)
        #comment= re.findall(r"#(\w+)", doc)
        print(f"Le document {i} contient {len(liste_mots)} mots")
        print(f"La liste de mots du document {i} : {liste_mots} \n")
        i+=1
    return liste_mots
# print(comments(corpus))
# %%
df.head()
# %%
#from spacy
stopWords = ['a','abord', 'absolument', 'afin', 'ah', 'ai', 'aie', 'ailleurs', 'ainsi', 'ait', 'allaient', 'allo', 'allons',
             'allô', 'alors', 'anterieur', 'anterieure', 'anterieures', 'apres', 'après', 'as', 'assez', 'attendu', 'au',
             'aucun', 'aucune', 'aujourd', "aujourd'hui", 'aupres', 'auquel', 'aura', 'auraient', 'aurait', 'auront', 'aussi',
             'autre', 'autrefois', 'autrement', 'autres', 'autrui', 'aux', 'auxquelles', 'auxquels', 'avaient', 'avais', 'avait',
             'avant', 'avec', 'avoir', 'avons', 'ayant', 'bah', 'bas', 'basee', 'bat', 'beau', 'beaucoup', 'bigre', 'boum',
             'bravo', 'brrr', 'car', 'ce', 'ceci', 'cela', 'celle', 'celle-ci', 'celle-là', 'celles', 'celles-ci', 'celles-là',
             'celui', 'celui-ci', 'celui-là', 'cent', 'cependant', 'certain', 'certaine', 'certaines', 'certains', 'certes', 'ces',
             'cet', 'cette', 'ceux', 'ceux-ci', 'ceux-là', 'chacun', 'chacune', 'chaque', 'cher', 'chers', 'chez', 'chiche', 'chut',
             'chère', 'chères', 'ci', 'cinq', 'cinquantaine', 'cinquante', 'cinquantième', 'cinquième', 'clac', 'clic', 'combien',
             'comme', 'comment', 'comparable', 'comparables', 'compris', 'concernant', 'contre', 'couic', 'crac', "c'", "d'", 'da',
             'dans', 'de', 'debout', 'dedans', 'dehors', 'deja', 'delà', 'depuis', 'dernier', 'derniere', 'derriere', 'derrière',
             'des', 'desormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'deux', 'deuxième', 'deuxièmement', 'devant', 'devers',
             'devra', 'different', 'differentes', 'differents', 'différent', 'différente', 'différentes', 'différents', 'dire',
             'directe', 'directement', 'dit', 'dite', 'dits', 'divers', 'diverse', 'diverses', 'dix', 'dix-huit', 'dix-neuf',
             'dix-sept', 'dixième', 'doit', 'doivent', 'donc', 'dont', 'douze', 'douzième', 'dring', 'du', 'duquel', 'durant', 'dès',
             'désormais', "d'", 'effet', 'egale', 'egalement', 'egales', 'eh', 'elle', 'elle-même', 'elles', 'elles-mêmes', 'en',
             'encore', 'enfin', 'entre', 'envers', 'environ', 'es', 'est', 'et', 'etaient', 'etais', 'etait', 'etant', 'etc', 'etre',
             'eu', 'euh', 'eux', 'eux-mêmes', 'exactement', 'excepté', 'extenso', 'exterieur', 'fais', 'faisaient', 'faisant', 'fait',
             'façon', 'feront', 'fi', 'flac', 'floc', 'font', 'gens', 'ha', 'hein', 'hem', 'hep', 'hi', 'ho', 'holà', 'hop', 'hormis',
             'hors', 'hou', 'houp', 'hue', 'hui', 'huit', 'huitième', 'hum', 'hurrah', 'hé', 'hélas', 'i', 'il', 'ils', 'importe',
             'je', 'jusqu', 'jusque', 'juste', "j'","J", "l'","L'", 'la', 'laisser', 'laquelle', 'las', 'le', 'lequel', 'les',
             'lesquelles', 'lesquels', 'leur', 'leurs', 'longtemps', 'lors', 'lorsque', 'lui', 'lui-meme', 'lui-même', 'là', 'lès', "l'",
             "m'", 'ma', 'maint', 'maintenant', 'mais', 'malgre', 'malgré', 'maximale', 'me', 'meme', 'memes', 'mes', 'mien', 'mienne',
             'miennes', 'miens', 'mille', 'mince', 'minimale', 'moi', 'moi-meme', 'moi-même', 'moindres', 'moins', 'mon',
             'moyennant', 'même', 'mêmes', "m'",'na', 'naturel', 'naturelle', 'naturelles', 'neanmoins', 'necessaire',
             'necessairement', 'neuf', 'neuvième', 'ni', 'nombreuses', 'nombreux', 'non', 'nos', 'notamment', 'notre', 'nous', 'nous-mêmes',
             'nouveau', 'nul', 'néanmoins', 'nôtre', 'nôtres', "n'", 'o', 'oh', 'ohé', 'ollé', 'olé', 'on', 'ont', 'onze', 'onzième', 'ore',
             'ou', 'ouf', 'ouias', 'oust', 'ouste', 'outre', 'ouvert', 'ouverte', 'ouverts', 'où', 'paf', 'pan', 'par', 'parce', 'parfois',
             'parle', 'parlent', 'parler', 'parmi', 'parseme', 'partant', 'particulier', 'particulière', 'particulièrement', 'passé',
             'pendant', 'pense', 'permet', 'personne', 'peut', 'peuvent', 'peux', 'pff', 'pfft', 'pfut', 'pif', 'pire', 'plein', 'plouf',
             'plus', 'plusieurs', 'plutôt', 'possessif', 'possessifs', 'possible', 'possibles', 'pouah', 'pour', 'pourquoi', 'pourrais', 'pourrait',
             'pouvait', 'prealable', 'precisement', 'premier', 'première', 'premièrement', 'pres', 'probable', 'probante', 'procedant', 'proche',
             'près', 'psitt', 'pu', 'puis', 'puisque', 'pur', 'pure', "qu'", 'quand', 'quant', 'quant-à-soi', 'quanta', 'quarante', 'quatorze',
             'quatre', 'quatre-vingt', 'quatrième', 'quatrièmement', 'que', 'quel', 'quelconque', 'quelle', 'quelles', "quelqu'un", 'quelque',
             'quelques', 'quels', 'qui', 'quiconque', 'quinze', 'quoi', 'quoique', "qu'", 'rare', 'rarement', 'rares', 'relative', 'relativement',
             'remarquable', 'rend', 'rendre', 'restant', 'reste', 'restent', 'restrictif', 'retour', 'revoici', 'revoilà', 'rien', "s'", 'sa',
             'sacrebleu', 'sait', 'sans', 'sapristi', 'sauf', 'se', 'sein', 'seize', 'selon', 'semblable', 'semblaient', 'semble', 'semblent',
             'sent', 'sept', 'septième', 'sera', 'seraient', 'serait', 'seront', 'ses', 'seul', 'seule', 'seulement', 'si', 'sien', 'sienne',
             'siennes', 'siens', 'sinon', 'six', 'sixième', 'soi', 'soi-même', 'soit', 'soixante', 'son', 'sont', 'sous', 'souvent', 'specifique',
             'specifiques', 'speculatif', 'stop', 'strictement', 'subtiles', 'suffisant', 'suffisante', 'suffit', 'suis', 'suit', 'suivant',
             'suivante', 'suivantes', 'suivants', 'suivre', 'superpose', 'sur', 'surtout', "s'", "t'", 'ta', 'tac', 'tant', 'tardive', 'te',
             'tel', 'telle', 'tellement', 'telles', 'tels', 'tenant', 'tend', 'tenir', 'tente', 'tes', 'tic', 'tien', 'tienne', 'tiennes',
             'tiens', 'toc', 'toi', 'toi-même', 'ton', 'touchant', 'toujours', 'tous', 'tout', 'toute', 'toutefois', 'toutes', 'treize', 'trente',
             'trois', 'troisième', 'troisièmement', 'trop', 'tsoin', 'tsouin', 'tu', 'té', "t'", 'un', 'une', 'unes',
             'uniformement', 'unique', 'uniques', 'uns', 'va', 'vais', 'vas', 'vers', 'via', 'vif', 'vifs', 'vingt', 'vivat', 'vive', 'vives',
             'vlan', 'voici', 'voilà', 'vont', 'vos', 'votre', 'vous', 'vous-mêmes', 'vu', 'vé', 'vôtre', 'vôtres', 'zut', 'à', 'â', 'ça',
             'étaient', 'étais', 'était', 'étant', 'été', 'être', 'ô','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
             'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'grrrrrrr','mdrrrrrr', 'mdr', 'qu','grosso','modo',"c'est","s'est","en gros",'lol'
             ,'ptdr', 'svp','stp','qu', 'tr', "fois","vivement","vraiment", "mr.", "vraiment"
             , 'mdm', 'm.', 'mme']
stopWords = [unidecode(sw) for sw in stopWords]
# %%
#-------------------------Creation of  functions for cleanning process-----------------------#
#remove username et @username
def remove_users(comments):
    comments = re.sub('@[A-Za-z]+[A-Za-z0-9-_]+', '', comments)# remove @ before an alpha numerical word
    # remove tweeted at
    return comments
#remove links
def remove_links(comments):
    comments = re.sub(r'http\S+', '', comments) # remove http links
    comments = re.sub(r'bit.ly/\S+', '', comments) # remove bitly links
    comments = comments.strip('[link]') # remove [links]
    return comments
#lower case
def lower(comments):
    return comments.lower()
#remove stopwords
def removeStopWords(Comments):
    stops=stopWords
    new_comments = ' '.join([word for word in Comments.split() if word not in stops])
    return new_comments
#remove blank space and special caracters
def punct(comments):
    token=RegexpTokenizer(r'\w+')#regex
    comments = token.tokenize(comments)
    comments= " ".join(comments)
    return comments
#remove digits
def remove_digits(comments):
    pattern = r'[^a-zA-zÀ-ú.,!?/:;\"\'\s]'#when ^ inside [] >> don't want to remove these chars
    return re.sub(pattern, '', comments)
#remove spécial characters
def remove_special_characters(comments):
    #define the pattern to keep
    pat = r'[^a-zA-z0-9-zÀ-ú0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', comments)
def remove_(comments):
    comments = re.sub('([_]+)', ' ',comments)
    return comments
#remove french accent
def accent(comments):
    #Comments = re.sub('[^A-zÀ-ú]+', ' ', Comments) #keep french accents
    comments = unidecode(comments)
    return comments
#transform years
def translate_year(comments):
    comments = re.sub(r'[0-9]{4}', 'annee', comments) 
    return comments
# %%
#translate emojis
def translate_emoji(comments) :
    emo = emoji.demojize(comments, language='fr')
    return emo
# %%
def translate_comments(comments) :
    time.sleep(0.01)
    translator = Translator()
    #translated = GT(source='auto', target='french').translate(text=comments)
    translated = translator.translate(src='auto', dest='french', text=comments)
    return translated
# %%
## apply one part of the functions to the dataframe
df['new_comments'] = df.Comments.apply(func = translate_emoji)
#df['new_comments'] = df.new_comments.apply(func = translate_comments)
df['new_comments'] = df.new_comments.apply(func = translate_year)
df['new_comments'] = df.new_comments.apply(func = lower)
# %%
sample = "Hello"
test = lower(sample)
print(test)
# %%
df.head()
# %% [markdown]
# **Normalisation**
# %%
#Retrieve the comments in a str object
df_str = df['new_comments'].str.split('_', expand=True) #create a df with only df['new_comments] as a column
arr = df_str[0].unique() #retrieve only the comments --output : np.array
my_str = ",".join(str(x) for x in arr) # transform the array into str
# %%
len(my_str)
# %%
nlp = spacy.load("fr_core_news_md", disable = ["parser","ner"])
#add a language detector
nlp.max_length = len(my_str)
#comments = str(df['new_comments'].unique())
doc = nlp(my_str)
tagged_tokens = [(token.text, token.pos_) for token in doc]
print(tagged_tokens)
# %%
doc = nlp(my_str)
spacy.displacy.render(doc,style='dep',jupyter=True)
# %%
#change the tag of specific words
def get_pos(tok):
    return tok.tag_ if (str(tok) != "top") else "ADJ"
# %%
#change the tag of specific words
#0. change the ruler name to avoid error
from spacy.pipeline import EntityRuler
ruler_comments = EntityRuler(nlp, overwrite_ents=True)
words = ["top"]
for w in words :
    ruler_comments.add_patterns([{"label": "adjectif", "pattern": w}])
ruler_comments.name = 'ruler_comments'
# %%
from spacy.language import Language
from spacy_langdetect import LanguageDetector
#add a language detector
def get_lang_detector(nlp, name):
    return LanguageDetector()
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)
# %% [markdown]
# ## 1. add a ruler 
# #ruler = nlp.add_pipe("attribute_ruler")
# ruler = nlp.add_pipe('ruler_comments')
# patterns = [[{"ORTH": "Family"}]]
# attrs = {"TAG": "PROPN", "POS": "ADJ"}
# ruler.add(patterns=patterns, attrs=attrs)
# 
# ## 2. change the variable doc
# 
# doc2 = nlp(my_str)
# %% [markdown]
# ruler = nlp.add_pipe("ruler_comments")
# patterns = [[{"TAG": "ADJ"}]]
# attrs = {"POS": "ADJ"}
# ruler.add(patterns=patterns, attrs=attrs)
# %%
def lemmatize(docs, allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]):
  '''
  Performs lemmization of input documents.
  Args:
    - docs: list of strings with input documents
    - allowed_postags: list of accepted Part of Speech (POS) types
  Output:
    - list of strings with lemmatized input
  '''
  nlp = spacy.load("fr_core_news_md", disable = ["parser","ner"])
  lemmatized_docs = []
  for doc in docs:
    doc = nlp(doc)
    tokens = []
    for token in doc:
      if token.pos_ in allowed_postags:
        tokens.append(token.lemma_)
    lemmatized_docs.append(" ".join(tokens))
  return (lemmatized_docs)
# %%
def tokenize(docs):
  '''
  Performs tokenization of input documents.
  Args:
    - docs: list of strings with input documents
  Output:
    - list of strings with tokenized input
  '''
  tokenized_docs = []
  for doc in docs:
    tokens = gensim.utils.simple_preprocess(doc, deacc=True)
    tokenized_docs.append(tokens)
  return (tokenized_docs)
# %%
docs = list(df['new_comments']) #use the original comments
#docs = pd.Series(docs)
# %%
len(docs)
# %%
print(docs)
# %%
lemmatized_docs = lemmatize(docs)
tokenized_docs = tokenize(lemmatized_docs)
# %%
df['lemmatized_docs'] = lemmatized_docs
# %%
df['Comments'][13]
# %%
df['lemmatized_docs'][13]
# %%
df.head()
# %%
### apply the functions to the dataframe
#df['new_comments'] = df.lemmatized_docs.apply(func = translate_emoji)
#df['new_comments'] = df.new_comments.apply(func = translate_comments)
df['cleaned_comments'] = df.lemmatized_docs.apply(func = accent)
#df['new_comments'] = df.new_comments.apply(func = remove_users)
#df['new_comments'] = df.new_comments.apply(func = remove_links)
df['cleaned_comments'] = df.cleaned_comments.apply(func = lower)
df['cleaned_comments'] = df.cleaned_comments.apply(func = punct)
#df['new_comments'] = df.new_comments.apply(func = translate_year)
#df['new_comments'] = df.new_comments.apply(func = remove_digits)
#df['new_comments'] = df.new_comments.apply(func = remove_special_characters)
#df['new_comments'] = df.new_comments.apply(func = remove_)
df['cleaned_comments'] = df.cleaned_comments.apply(func = removeStopWords) #at the end in order to have a better cleaning
# %%
word_list = [re.split('\W+',doc) for doc in df['cleaned_comments']]
# %%
df['comments_length_2'] = [len(doc) for doc in word_list]
# %%
df.head()
# %%
data2 = df[df['comments_length_2']==1]
display (
    data2['cleaned_comments'].count()
    , data2['cleaned_comments'].unique()
    )
# %%
# Here we use a column with categorical data
fig = px.histogram(df, x="comments_length_2")
fig.update_layout(bargap=0.2)
fig.show()
# %%
df['Comments'][1]
# %%
df['lemmatized_docs'][1]
# %%
df['cleaned_comments'][1]
# %%
df['comments_length'][13]
# %% [markdown]
# **Ngram Tokenization**
# %%
# counting term frequencies
from collections import Counter
# accessing items in table
from operator import itemgetter
# flatten list to string
from nltk.lm.preprocessing import flatten
# create trigram (ngram approach)
from nltk.util import ngrams
# %%
set_excl_stopwords=set(['ne',"n'",'pas','jamais','non'])
# %%
def remove_row_separator(text):
  return text.replace('┇',' ')
def remove_set_excl_stopwords(text):
  text_filtered=[word for word in text.split() if word not in set_excl_stopwords]
  return ' '.join(text_filtered)
# %% [markdown]
# https://blog.devgenius.io/working-with-nltk-ngram-analysis-on-airline-sentiment-twitter-dataset-65bf8d568b3c
# %%
#create list from ngram generator
# gram1 (unigram) does not include row_separator
df['gram1']=[list(ngrams(remove_row_separator(text).split(),1))for text in df['cleaned_comments']]
df[['gram1']][0:10].style.set_properties(**{'vertical-align': 'top'})
# %%
#create list from ngram generator
# gram2 (bigram) does not include row_separator
df['gram2']=[list(ngrams(remove_row_separator(text).split(),2))for text in df['cleaned_comments']]
df[['gram2']][0:10].style.set_properties(**{'vertical-align': 'top'})
# %%
#create list from ngram generator
# gram3 (trigram) does not include row_separator
df['gram3']=[list(ngrams(remove_row_separator(text).split(),3))for text in df['cleaned_comments']]
df[['gram3']][0:10].style.set_properties(**{'vertical-align': 'top'})
# %%
# count gram1 tokens
count_gram1 = Counter (flatten(list_item for list_item in df['gram1']))
print('len count_gram1:',len(count_gram1))
list_gram1 = sorted(list(count_gram1.items()),key=itemgetter(1),reverse=True)
print('len list_gram1:', len(list_gram1))
#limiting to top 300
print(list_gram1[0:300])
# %%
# count gram2 tokens
count_gram2 = Counter (flatten(list_item for list_item in df['gram2']))
print('len count_gram2:',len(count_gram2))
list_gram2 = sorted(list(count_gram2.items()),key=itemgetter(1),reverse=True)
print('len list_gram2:', len(list_gram2))
#limiting to top 300
print(list_gram2[0:300])
# %%
# count gram 3 tokens
count_gram3 = Counter (flatten(list_item for list_item in df['gram3']))
print('len count_gram3:',len(count_gram3))
list_gram3 = sorted(list(count_gram3.items()),key=itemgetter(1),reverse=True)
print('len list_gram3:', len(list_gram3))
#limiting to top 300
print(list_gram3[0:300])
# %%
# create a list for gram1 that occurs more than 1 time
list_gram1_word_gt_1=[gram[0] for gram in list_gram1 if int(gram[1])>1]
print(list_gram1_word_gt_1)
# %%
# create a list for gram2 that occurs more than 1 time
list_gram2_word_gt_1=[gram[0] for gram in list_gram2 if int(gram[1])>1]
print(list_gram2_word_gt_1)
# %%
# create a list for gram3 that occurs more than 1 time
list_gram3_word_gt_1=[gram[0] for gram in list_gram3 if int(gram[1])>1]
print(list_gram3_word_gt_1)
# %%
df.columns
# %%
# filter gram1 word containing word that occurs more than 1 time only
df['gram1_word_gt_1']=[', '.join('_'.join([gram[0]]) for gram in list_gram if gram in list_gram1_word_gt_1) for list_gram in df['gram1']]
# %%
# filter gram2 word containing word that occurs more than 1 time only
df['gram2_word_gt_1']=[', '.join('_'.join([gram[0],gram[1]]) for gram in list_gram if gram in list_gram2_word_gt_1) for list_gram in df['gram2']]
# %%
df['gram3_word_gt_1']=[', '.join('_'.join([gram[0],gram[1],gram[2]]) for gram in list_gram if gram in list_gram3_word_gt_1) for list_gram in df['gram3']]
# %%
df[['Id','Date_Id#Evaluation','answer', 'Comments', 'new_comments', 'lemmatized_docs',
       'cleaned_comments','comments_length', 'gram1','gram1_word_gt_1'
       , 'gram2','gram2_word_gt_1', 'gram3','gram3_word_gt_1'
       ]].head() #style.set_properties(**{'vertical-align': 'top'})[10]
# %% [markdown]
# **Visualisation**
# %% [markdown]
# # Word Frequency
# longtext= my_str
# nlp = spacy.load("fr_core_news_sm", disable = ["ner"])
# long_text=nlp(longtext)
# 
# 
# list_of_tokens=[token.text for token in long_text if not token.is_stop and not token.is_punct]
# token_frequency=Counter(list_of_tokens)
# #print(token_frequency)
# #printing 6 most coomon words using most_common() method of Counter
# 
# most_frequent_tokens=token_frequency.most_common(10)
# # print(most_frequent_tokens)
# %%
df.columns
# %%
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(1, 1), stop_words=stopWords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['gram1_word_gt_1'], 10)
df3 = pd.DataFrame(common_words, columns = ['onegram' , 'count'])
fig = go.Figure([go.Bar(x=df3['onegram'], y=df3['count'])])
fig.update_layout(title=go.layout.Title(text="Top 10 onegram in the comments after removing stop words and lemmatization"))
fig.show()
# %%
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stopWords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['cleaned_comments'], 20)
df3 = pd.DataFrame(common_words, columns = ['trigram' , 'count'])
fig = go.Figure([go.Bar(x=df3['trigram'], y=df3['count'])])
fig.update_layout(title=go.layout.Title(text="Top 20 trigrams in the comments after removing stop words and lemmatization"))
fig.show()
# %%
all_words = ''.join([word for word in df['cleaned_comments'][0:10]])
#all_words = df_plot[0]
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("les mots fréquemment utilisés dans les commentaires", weight='bold', fontsize=14)
plt.show()
# %%
#-------------------most usefull hashtags------------------#
text = str(df['cleaned_comments'].unique())
wordcloud = WordCloud().generate(text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=80, max_words=60, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 
# %% [markdown]
# **Topic Modeling**
# %%
#BIGRAMS & TRIGRAMS
# %%
#TF-IDF >> to remove the frequent non essentialwords 
# %%
# Mapping from word IDs to words
id2word = corpora.Dictionary(tokenized_docs)
# %%
# Prepare Document-Term Matrix
corpus = []
for doc in tokenized_docs:
    corpus.append(id2word.doc2bow(doc))
# %%
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, random_state = 100, passes = 100)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity = model.log_perplexity(corpus)
    return model_list, coherence_values
# %%
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=tokenized_docs, start=2, limit=40, step=6)
# %%
# Show graph
limit=40; start=2; step=6
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# %%
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of ", round(cv, 4))
# %%
# Select the model and print the perplexity
optimal_model = model_list[1]
print(f"optimal model : {optimal_model}")
print("Perplexity : ", optimal_model.log_perplexity(corpus)) 
# %% [markdown]
# - Coherence score and perplexity provide a convinent way to measure how good a given topic model is.
# - Lower the perplexity better the model.
# - Higher the topic coherence, more human interpretable the topic is .
# %%
# Visualize with pyLDAvis: See [2] for more details
pyLDAvis.enable_notebook()
visualization = pyLDAvis.gensim_models.prepare(
    optimal_model, 
    corpus,
    id2word,
    mds = "mmds", 
    R = 30)
# %% [markdown]
# visualization
# %%
num_topics = 8
num_words = 10
# %%
#display the topics and the most significant ones
model_topics = optimal_model.show_topics(formatted=False)
significant_topics = optimal_model.print_topics(num_topics=num_topics, num_words=num_words) #Get the most significant topics (alias for show_topics() method)
significant_topics.sort() 
significant_topics
# %%
# Get the most representative words in each topic
topics = optimal_model.show_topics(num_topics=num_topics,
                               num_words=num_words,
                               formatted=False)
# Assign a name to each topic based on the most representative words
topic_names = ['Topic {}'.format(i) for i in range(num_topics)]
for i, topic in enumerate(topics):
  words = [word for word, _ in topic[1]]
  #  topic_names[i] = ' '.join(words)
  if i == 0 :
    topic_names[i] = 'Produit'
  else :
    topic_names[i] = 'Service'
# Attribute scores to each LDA topic
topic_label = []
for i, doc in enumerate(corpus):
    topic_scores = optimal_model.get_document_topics(doc)
    for topic, score in topic_scores:
        print('Document {}, {}: {}'.format(i, topic_names[topic], score))
        topic_label.append(topic_names[topic])
        topic_label.append(score)
# %%
# Rename topics
## Get the most representative words in each topic
topics = optimal_model.show_topics(num_topics=num_topics,
                               num_words=num_words,
                               formatted=False)

topic_names = ['topic {}'.format(i) for i in range(num_topics)]
for i, topic in enumerate(topics):
  words = [word for word, _ in topic[1]]
  #  topic_names[i] = ' '.join(words)
  if i == 0 :
    topic_names[i] = '0_personnel'
  elif i == 1 :
    topic_names[i] = '1_service'
  elif i == 2 :
    topic_names[i] = '2_remerciement'
  elif i == 3 :
    topic_names[i] = '3_accueil'
  elif i == 4 : 
    topic_names[i] = '4_accueil'
  elif i == 5 : 
    topic_names[i] = '5_accueil'
  elif i == 6 : 
    topic_names[i] = '6_personnel'
  else :
    topic_names[i] = '7_efficacite'

#Create a new dataframe with the topic name and score as columns
##Get the topic distribution for each document in the corpus
doc_topics = optimal_model.get_document_topics(corpus)

df1 = pd.DataFrame(columns=['Document'] + topic_names)
for i, doc in enumerate(doc_topics):
  row = [i] + [0] * num_topics
  for topic, score in doc:
      row[topic + 1] = score
  df1.loc[i] = row
#Rename the 'Document' column to 'Document ID'
df1 = df1.rename(columns={'Document': 'topic'})
# View the resulting dataframe
df1.head()
# %%
#join the original df to df1
df2 = df.join(df1)
# %%
df2.head()
# %%
#topic for all docs
#df['topic'] = [max(optimal_model.get_document_topics(i), key=lambda x: x[1]) for i in corpus]
#df['topic'] = [(optimal_model.get_document_topics(i)) for i in corpus]

# %%
df2.columns
# %%
#create the last table
df2= df[['Date_Id#Evaluation',
       'answer', 'Comments', "topic"
       'comments_length',
       'gram1_word_gt_1',
       'gram2_word_gt_1',
       'gram3_word_gt_1'
       ]]

# %%
df3[df3['Comments']== 'Top'].head()
# %% [markdown]
# # CONNEXION A SQL SERVER : table DWH
# %%


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


import os


# In[3]:


import os

path = r'C:\Users\arsha\OneDrive\Desktop\search_engine\project\files'
os.chdir(path)


# In[4]:


os.getcwd()


# In[5]:


import os
directory = r"C:\Users\arsha\OneDrive\Desktop\search_engine\project\files"
compressed_file_name = "eng_subtitles_database.db.gz"
decompressed_file_name = "eng_subtitles_database_decompressed.db"
compressed_database_path = os.path.join(directory, compressed_file_name)
decompressed_database_path = os.path.join(directory, decompressed_file_name)

print("Compressed database file path:", compressed_database_path)
print("Decompressed database file path:", decompressed_database_path)


# In[6]:


import sqlite3
import pandas as pd
import zipfile
import io
import re

database_path = r"C:\Users\arsha\OneDrive\Desktop\search_engine\project\files\eng_subtitles_database.db"
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT num, name, content FROM zipfiles")
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['subtitle_id', 'subtitle_name','subtitle_content'])

    print("Subtitle files fetched, decoded, cleaned, and stored in DataFrame successfully.")
    
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    cursor.close()
    conn.close()


# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.drop_duplicates(inplace = True)


# In[10]:


df.isna().sum()


# In[11]:


import zipfile
import io

def extract_content(content):
    with io.BytesIO(content) as bio:
        with zipfile.ZipFile(bio, "r") as zipf:
            file_list = zipf.namelist()
            for file_name in file_list:
                with zipf.open(file_name) as file:
                    content = file.read().decode("latin-1")
                    return content

df['subtitle_content'] = df['subtitle_content'].apply(extract_content)


# In[12]:


new_df = df.sample(frac = 0.3, random_state = 42)


# In[13]:


new_df.shape


# In[14]:


new_df.head()


# In[15]:


new_df.reset_index(drop = True, inplace = True)


# In[16]:


new_df['subtitle_content'][0]


# In[17]:


def clean_text(text):
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\r\n', '', text)
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
new_df['subtitle_content'] = new_df['subtitle_content'].apply(clean_text)


# In[18]:


new_df['subtitle_content'][0]


# In[19]:


def partitioning_srt_file_data_into_chunks(row, window_size=512, overlapping=100):
    chunks = []
    i = 0
    while i < len(row['subtitle_content']):
        chunk_end = min(i + window_size, len(row['subtitle_content']))
        chunks.append(row['subtitle_content'][i:chunk_end])
        i += window_size - overlapping
    return chunks

new_df['subtitle_content'] = new_df.apply(partitioning_srt_file_data_into_chunks, axis=1)


# In[20]:


new_df['subtitle_content'][0]


# In[21]:


new_df['subtitle_name'] = new_df['subtitle_name'].apply(lambda x: re.sub('\.', ' ', x))


# In[22]:


new_df.head()


# In[23]:


new_df = new_df.explode('subtitle_content',ignore_index = False)


# In[24]:


new_df.shape


# In[25]:


df_1 = new_df.sample(frac = 0.2, random_state = 42, axis = 0)


# In[26]:


df_1.shape


# In[27]:


#df_1.to_csv('search_engine.csv', index = False)


# In[28]:


df_1.head()


# In[29]:


df_1.shape


# In[30]:


df_1.reset_index(drop = True, inplace = True)


# In[31]:


df_1['subtitle_content'] = df_1['subtitle_content'].apply(lambda x : x.lower())


# In[32]:


df_1.head()


# In[33]:


df_1['subtitle_content'][1]


# In[34]:


df_1['subtitle_id'].value_counts()


# In[35]:


from wordcloud import WordCloud

wc = WordCloud(background_color='black',
               width=2500,
               height=1000).generate(' '.join(df_1['subtitle_content']))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[36]:


pip install chromadb


# In[37]:


import chromadb


# In[38]:


import os
path = os.getcwd()


# In[39]:


path


# In[40]:


client = chromadb.PersistentClient(path="/path/to/save/to")


# In[41]:


client.heartbeat() 


# In[42]:


from chromadb.utils import embedding_functions


# In[43]:


pip install sentence-transformers


# In[44]:


from sentence_transformers import SentenceTransformer,util


# In[45]:


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# In[47]:


collection = client.create_collection(
        name="subtitle_semioo",
        metadata={"hnsw:space": "cosine"},
        embedding_function=sentence_transformer_ef
    )


# In[48]:


content = df_1['subtitle_content'].tolist()
metadatas = [{'subtitle_name': name, 'subtitle_id': id} for name, id in zip(df_1['subtitle_name'], df_1['subtitle_id'])] 
ids = [str(i) for i in range(len(df_1))]


# In[50]:


import multiprocessing

# Define a function to process a batch and add it to the collection
def process_batch(batch_content, batch_metadatas, batch_ids, collection):
    collection.add(documents=batch_content, metadatas=batch_metadatas, ids=batch_ids)

def main():
    batch_size = 5000  
    num_batches = (len(content) + batch_size - 1) // batch_size 

    # Create a multiprocessing Pool
    pool = multiprocessing.Pool()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(content))
        
        batch_content = content[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        
        # Process the batch in parallel
        pool.apply_async(process_batch, args=(batch_content, batch_metadatas, batch_ids, collection))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()


# In[49]:


batch_size = 5000  
num_batches = (len(content) + batch_size - 1) // batch_size 

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(content))
    
    batch_content = content[start_idx:end_idx]
    batch_metadatas = metadatas[start_idx:end_idx]
    batch_ids = ids[start_idx:end_idx]
    
    collection.add(
        documents=batch_content,
        metadatas=batch_metadatas,
        ids=batch_ids
    )  


# In[51]:


get_ipython().run_cell_magic('timeit', '', '')


# In[52]:


query_text = 'They found traces of Marthas blood in the trunk of your car' 


# In[53]:


result = collection.query(
    query_texts = query_text,
    include=["metadatas", 'distances'],
    n_results=10
)


# In[54]:


ids = result['ids'][0]
distances = result['distances'][0] 
metadatas = result['metadatas'][0] 
zipped_data = zip(ids, distances, metadatas)
sorted_data = sorted(zipped_data, key=lambda x: x[1], reverse=True)
for _, distance, metadata in sorted_data:
    subtitle_name = metadata['subtitle_name']
    print(f"Subtitle Name: {subtitle_name.upper()}")


# In[55]:


print(f"Subtitle Name: {subtitle_name.upper()}")


# In[58]:


# Assuming sorted_data contains the sorted query results
subtitle_names = [item[2]['subtitle_name'] for item in sorted_data]

for subtitle_name in subtitle_names:
    print(subtitle_name.upper())


# In[56]:


subtitle_names = [item['subtitle_name'] for sublist in results for item in sublist]

for subtitle_name in subtitle_names:
    print(subtitle_name.upper())


# In[61]:


from chromadb.utils import embedding_functions

# Choose a pre-trained SentenceTransformer model from the Hugging Face Model Hub
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Define the embedding function
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# Now you can create or get the collection using emb_fn
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)



# In[59]:


collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)


# In[63]:


# Retrieving the collection if it exists
try:
    collection = client.get_collection(name="test")
    print("Collection 'test' exists.")
except ValueError:
    print("Collection 'test' does not exist.")

# Creating or retrieving the collection
collection = client.get_or_create_collection(name="test")


# In[64]:


#retriving or creating
collection = client.get_collection(name="test") 
collection = client.get_or_create_collection(name="test") 
client.delete_collection(name="my_collection") 


# In[65]:


collection = client.get_collection(name="subtitle_sem")


# In[67]:


# Assuming client is your ChromaDB client
collection_name = "subtitle_sem"
collection = client.get_collection(name=collection_name)

# Now you can use the collection object as needed


# In[ ]:


print(subtitle_sem)


# In[69]:


import torch
print(torch.__version__)


# In[70]:


from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer,util


# In[71]:


model = SentenceTransformer("all-MiniLM-L6-v2")


# In[81]:


def encoding_content(x):
    return model.encode(x, normalize_embeddings=True)

df_2 = df_2.copy()  # Create a copy of the DataFrame
df_2.loc[:, 'subtitle_content_encoded'] = df_2['subtitle_content'].apply(encoding_content)


# In[78]:


def encode_batch(batch):
    # Assuming model.encode is a function that encodes a single sample
    return [model.encode(sample, normalize_embeddings=True) for sample in batch]

# Split the DataFrame into batches
batch_size = 100  # Adjust batch size as needed
batches = [df_2[i:i+batch_size] for i in range(0, len(df_2), batch_size)]

# Encode each batch and concatenate the results
encoded_batches = [encode_batch(batch['subtitle_content'].tolist()) for batch in batches]
encoded_data = np.concatenate(encoded_batches)

# Assign the encoded data to the DataFrame
df_2['subtitle_content_encoded'] = encoded_data



# In[ ]:


df_2.head()


# In[79]:


import chromadb
from chromadb.utils import embedding_functions


# In[80]:


chroma_client = chromadb.HttpClient(host='localhost', port=8000)


# In[82]:


chroma_client = chromadb.Client()


# In[83]:


collection = chroma_client.create_collection(name="my_collection")


# In[ ]:


import numpy as np

documents = df_2['subtitle_content'].tolist()
#embedding_val = df_2['subtitle_content_encoded'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).tolist()
metadatas = [{'subtitle_name': name, 'subtitle_id': _id} for name, _id in zip(df_2['subtitle_name'], df_2['subtitle_id'])]
ids = [str(i) for i in range(len(df_2))]
collection.add(documents=documents, metadatas=metadatas, ids=ids)


# In[ ]:


db = chromadb.connect('my_collection')
for index, row in df_2.iterrows():
    id_val = row['subtitle_id']
    embedding_val = row['subtitle_content_encoded']
    document_val = row['subtitle_content']
    metadata_val = row['subtitle_name']

    # Add the values to ChromaDB
    db.add_document(embeddings=[embedding_val], documents=[document_val], metadatas={"subtitle_id": id_val, "name": metadata_val}, ids=[index])


# In[ ]:


chroma_client = chromadb.Client()


# In[ ]:


collection = chroma_client.create_collection(name="mycollection")


# In[ ]:


chroma_client.delete_collection(name="my_collection")


# In[ ]:


documents = df_2['subtitle_content'].tolist()
metadatas = [{'subtitle_name': name, 'subtitle_id': _id} for name, _id in zip(df_2['subtitle_name'], df_2['subtitle_id'])]
ids = [str(i) for i in range(len(df_2))]
collection.add(documents=documents, metadatas=metadatas, ids=ids)


# In[ ]:


results = collection.query(
    query_texts=["looking at the eye chart onto which he hallucinated a bat"],
    n_results=5
)


# In[ ]:


'''
import sqlite3
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Import data from database
def importing_data_from_database(x_name_database, table_name):
    conn = sqlite3.connect(x_name_database + ".db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Partition text data into chunks
def partitioning_srt_file_data_into_chunks(rows, n=20):
    chunks = []
    ids = []
    for idx, row in enumerate(rows):
        text = row[1]
        for i in range(0, len(text), n):
            chunk = text[i:i+n]
            chunks.append(chunk)
            ids.append(row[0])
    df = pd.DataFrame({'id': ids, 'chunk': chunks})
    return df

# Custom transformer to clean text
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemm = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_text = []
        for x in X:
            x = re.sub(r'[^a-zA-Z\s]', '', x)
            x = re.sub(r'[^a-zA-Z0-9\s]', '', x)
            words = x.split()
            words_without_stopwords = [word for word in words if word.lower() not in self.stop_words]
            cleaned_text.append(' '.join([self.lemm.lemmatize(word) for word in words_without_stopwords]))
        return cleaned_text


class TextToVector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)

    def fit(self, X, y=None):
        self.model.build_vocab(X)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, X):
        return [self.model.wv[word] for word in X]

# Save vectors to Chrom database
class SaveToChromDatabase(BaseEstimator, TransformerMixin):
    def __init__(self, dbname):
        self.conn = sqlite3.connect(dbname + ".db")
        self.cursor = self.conn.cursor()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for vector in X:
            self.cursor.execute("INSERT INTO vectors (vector) VALUES (?)", (vector,))
        self.conn.commit()
        return X


pipeline = Pipeline([
    ('import_data', importing_data_from_database),
    ('partition_chunks', partitioning_srt_file_data_into_chunks),
    ('clean_text', TextCleaner()),
    ('text_to_vector', TextToVector()),
    ('save_to_chrom', SaveToChromDatabase("your_chrom_database_name"))
])

x_name_database = 'your_database'
table_name = 'your_table'
data = pipeline.fit_transform(x_name_database, table_name)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





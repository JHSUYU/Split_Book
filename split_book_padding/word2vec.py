import pandas as pd
import gensim.models.word2vec as word2vec
def read_dat_data(filepath, columns):
    f = open(filepath, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.strip().split('\t')
        sentimentlist.append(s)
    f.close()
    res_pd = pd.DataFrame(sentimentlist, columns=columns)
    return res_pd

users = read_dat_data('book_crossing/users_info.dat', ['user', 'location', 'age'])
users['city'] = users['location'].map(lambda x: x.split(',')[0])
users['state'] = users['location'].map(lambda x: x.split(',')[1])
users['country'] = users['location'].map(lambda x: x.split(',')[2])

city=list()
for name in users['city']:
    if not name in city:
        city.append(name)

model=word2vec.Word2Vec([city],min_count=1,size=7,window=2)
print(model.similarity('berkeley','austin'))

# 构建模型

import pandas as pd
import numpy as np
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


def prepare_bookrating_data(watch_vec_size, book_rating):

    max_user_id = 2945
    user_watch = np.zeros((max_user_id, watch_vec_size))
    user_labels = np.zeros((max_user_id, 1))

    MOVIE_EMBED_TAB_HEIGHT = 17384
    MOVIE_EMBED_TAB_WIDTH = watch_vec_size


    np.random.seed(1)

    MOVIE_EMBED_TAB = np.zeros((MOVIE_EMBED_TAB_HEIGHT, MOVIE_EMBED_TAB_WIDTH))


    for i in range(MOVIE_EMBED_TAB_HEIGHT):
        MOVIE_EMBED_TAB[i][hash(i) % MOVIE_EMBED_TAB_WIDTH] = 1
        MOVIE_EMBED_TAB[i][hash(hash(i)) % MOVIE_EMBED_TAB_WIDTH] = 1


    pre_uid = 0
    movie_count = 0
    for index, row in book_rating.iterrows():
        uid = int(row['user']) - 1
        mov_id = int(row['item']) - 1
        user_watch[uid, :] += MOVIE_EMBED_TAB[mov_id, :]
        user_labels[uid, :] = mov_id
        if uid == pre_uid:
            movie_count += 1
        else:
            user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count
            movie_count = 1
            pre_uid = uid
    user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count

    return user_watch, user_labels


def prepare_bookhistory_data(search_vec_size, book_history):
    """
    prepare movielens data
    """
    max_user_id = 2945
    user_search = np.zeros((max_user_id, search_vec_size))

    MOVIE_EMBED_TAB_HEIGHT = 17384
    MOVIE_EMBED_TAB_WIDTH = search_vec_size

    np.random.seed(1)

    MOVIE_EMBED_TAB = np.zeros((MOVIE_EMBED_TAB_HEIGHT, MOVIE_EMBED_TAB_WIDTH))

    for i in range(MOVIE_EMBED_TAB_HEIGHT):
        MOVIE_EMBED_TAB[i][hash(i) % MOVIE_EMBED_TAB_WIDTH] = 1
        MOVIE_EMBED_TAB[i][hash(hash(i)) % MOVIE_EMBED_TAB_WIDTH] = 1

    pre_uid = 0
    movie_count = 0
    for index, row in book_history.iterrows():
        uid = int(row['user']) - 1
        mov_id = int(row['item']) - 1
        user_search[uid, :] += MOVIE_EMBED_TAB[mov_id, :]
        if uid == pre_uid:
            movie_count += 1
        else:
            user_search[pre_uid, :] = user_search[pre_uid, :] / movie_count
            movie_count = 1
            pre_uid = uid
    user_search[pre_uid, :] = user_search[pre_uid, :] / movie_count
    return user_search

def prepare_userinfo_data(other_feat_size, user_info, city, state ,country):
    model_city = word2vec.Word2Vec([city], min_count=1, size=15, window=2)
    model_state = word2vec.Word2Vec([state], min_count=1, size=8, window=2)
    model_country = word2vec.Word2Vec([country], min_count=1, size=2, window=2)
    age_table=[1, 18, 25, 35, 45, 50, 60]

    max_user_id = 2945
    user_feat = np.zeros((max_user_id, other_feat_size))

    AGE_EMBED_TAB_HEIGHT = len(age_table)
    AGE_EMBED_TAB_WIDTH = len(age_table)


    np.random.seed(1)

    AGE_EMBED_TAB = np.zeros((AGE_EMBED_TAB_HEIGHT, AGE_EMBED_TAB_WIDTH))


    for i in range(AGE_EMBED_TAB_HEIGHT):
        AGE_EMBED_TAB[i][i] = 1


    for index, row in user_info.iterrows():
        uid = int(row['user']) - 1
        if (row['age']==None):
            age=35
        else:
            age = int(row['age'])
        age_id=get_age_id(age,age_table)
        city=model_city[row['city']]
        state=model_state[row['state']]
        country=model_country[row['country']]
        user_feat[uid, :] = np.concatenate((city,state,country,
                                            AGE_EMBED_TAB[age_id, :]))

    return user_feat

def get_age_id(age,age_table):
    if age >= age_table[len(age_table) - 1]:
        age_id = len(age_table) - 1
        return age_id
    for k in range(1, len(age_table)):
        if (age >= age_table[k - 1] and age < age_table[k]):
            age_id = k - 1
            return age_id
    print("false")
    return 0

if __name__ == "__main__":
    users = read_dat_data('book_crossing/users_info.dat', ['user', 'location', 'age'])
    book_ratings = read_dat_data('book_crossing/book_ratings.dat', ['user', 'item', 'rating'])
    book_history = read_dat_data('book_crossing/book_history.dat', ['user', 'item', 'accessed'])
    users['city'] = users['location'].map(lambda x: x.split(',')[0])
    users['state'] = users['location'].map(lambda x: x.split(',')[1])
    users['country'] = users['location'].map(lambda x: x.split(',')[2])
    print(len(np.unique(users['city'])))
    print(len(np.unique(users['state'])))
    print(len(np.unique(users['country'])))
    city = list()
    for name in users['city']:
        if not name in city:
            city.append(name)
    state=list()
    for name in users['state']:
        if not name in state:
            state.append(name)
    country = list()
    for name in users['country']:
        if not name in country:
            country.append(name)
    # users.drop('location', axis=1, inplace=True)
    # df_history = pd.merge(book_history, users, on='user')
    # df_ratings = pd.merge(book_ratings, users, on='user')
    # user_watch, user_labels = prepare_bookrating_data(64, book_rating=book_ratings)
    # user_search=prepare_bookhistory_data(64,book_history=book_history)
    user_feat=prepare_userinfo_data(32,user_info=users,city=city,state=state,country=country)
    # np.save('user_watch', user_watch)
    # np.save('user_search', user_search)
    np.save('user_feat', user_feat)
    # np.save('user_labels', user_labels)
    print("1")

    # berkeley,california,usa --> [1665772542386832815, 7849071561198217636, -543696986365761034]  -->经过torch.nn.embedding层，然后这个embedding层和网络一起训练

    # location_list=user.Location.tolist()
    # location_id=0
    # location_name=[]
    # location_name_simple=[]
    # location_id_name=dict()
    # for i in range(1, len(location_list)):
    #     index=str(location_list[i]).index(',')
    #     location
    #     if not location_list[i] in location_id_name.keys():
    #         location_id_name[location_list[i]]=location_id
    #         location_id+=1
    # print(location_id_name)

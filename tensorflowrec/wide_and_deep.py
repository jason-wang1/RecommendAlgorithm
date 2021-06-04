import tensorflow as tf
import os

# ---------------------- get data ------------------------------
parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))

train_data_file = tf.keras.utils.get_file("trainingSamples.csv", "file://" + parent_dir + "/data/trainingSamples.csv")
test_data_file = tf.keras.utils.get_file("testSamples.csv", "file://" + parent_dir + "/data/testSamples.csv")


def get_dataset(data_path):
    dataset = tf.data.experimental.make_csv_dataset(
        data_path,
        batch_size=12,
        label_name='label',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


train_dataset = get_dataset(train_data_file)
test_dataset = get_dataset(test_data_file)

# ---------------------- input categorical features and embedding layer on deep model ------------------------------
categorical_columns = []

# 电影类别、用户喜欢的电影类别
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
genre_features = {
    "movieGenre1": genre_vocab,
    "movieGenre2": genre_vocab,
    "movieGenre3": genre_vocab,
    "userGenre1": genre_vocab,
    "userGenre2": genre_vocab,
    "userGenre3": genre_vocab,
    "userGenre4": genre_vocab,
    "userGenre5": genre_vocab
}
for feature, vocab in genre_features.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(categorical_column=cat_col, dimension=10)
    categorical_columns.append(emb_col)

# 电影id、用户id
movie_col = tf.feature_column.categorical_column_with_identity(key="movieId", num_buckets=1000, default_value=0)
movie_emb_col = tf.feature_column.embedding_column(categorical_column=movie_col, dimension=10)
categorical_columns.append(movie_emb_col)

user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30000, default_value=0)
user_emb_col = tf.feature_column.embedding_column(categorical_column=user_col, dimension=10)
categorical_columns.append(user_emb_col)

# ---------------------- input numerical features on deep model ------------------------------
numerical_columns = [
    tf.feature_column.numeric_column(key='releaseYear'),
    tf.feature_column.numeric_column(key='movieRatingCount'),
    tf.feature_column.numeric_column(key='movieAvgRating'),
    tf.feature_column.numeric_column(key='movieRatingStddev'),
    tf.feature_column.numeric_column(key='userRatingCount'),
    tf.feature_column.numeric_column(key='userAvgRating'),
    tf.feature_column.numeric_column(key='userRatingStddev')
]

# ---------------------- input feature on wide model ------------------------------
# 使用 movieId 与 userRatedMovie1 进行交叉积计算
rated_movie = tf.feature_column.categorical_column_with_identity(key="userRatedMovie1", num_buckets=1000, default_value=0)
crossed_col = tf.feature_column.indicator_column(tf.feature_column.crossed_column(keys=[rated_movie, movie_col], hash_bucket_size=10000))

# ---------------------- build model ------------------------------
inputs = {
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
}

deep = tf.keras.layers.DenseFeatures(feature_columns=categorical_columns + numerical_columns)(inputs)
deep = tf.keras.layers.Dense(units=128, activation='relu')(deep)
deep = tf.keras.layers.Dense(units=128, activation='relu')(deep)
wide = tf.keras.layers.DenseFeatures(feature_columns=crossed_col)(inputs)
both = tf.keras.layers.concatenate(inputs=[deep, wide])
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='ROC')]
)

model.fit(train_dataset, epochs=5)

# ---------------------- evaluation ------------------------------
test_loss, test_auc = model.evaluate(test_dataset)
print("\nLoss: {}; Auc: {}".format(test_loss, test_auc))

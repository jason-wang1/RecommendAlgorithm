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

# ---------------------- input categorical features and embedding layer ------------------------------
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
    emb_col = tf.feature_column.embedding_column(cat_col, 10)
    categorical_columns.append(emb_col)

# 电影id、用户id
movie_col = tf.feature_column.categorical_column_with_identity(key="movieId", num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(categorical_column=movie_col, dimension=10)
categorical_columns.append(movie_emb_col)

user_col = tf.feature_column.categorical_column_with_identity(key="userId", num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(categorical_column=user_col, dimension=10)
categorical_columns.append(user_emb_col)

# ---------------------- input numerical features ------------------------------
numerical_columns = [
    tf.feature_column.numeric_column(key='releaseYear'),
    tf.feature_column.numeric_column(key='movieRatingCount'),
    tf.feature_column.numeric_column(key='movieAvgRating'),
    tf.feature_column.numeric_column(key='movieRatingStddev'),
    tf.feature_column.numeric_column(key='userRatingCount'),
    tf.feature_column.numeric_column(key='userAvgRating'),
    tf.feature_column.numeric_column(key='userRatingStddev')
]

# ---------------------- build model ------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns=categorical_columns + numerical_columns),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='ROC')]
)

model.fit(train_dataset, epochs=5)

# ---------------------- evaluate ------------------------------
test_loss, test_auc = model.evaluate(test_dataset)
print("\nLoss: {}; Auc: {}".format(test_loss, test_auc))


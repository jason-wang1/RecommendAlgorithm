import os
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


# ---------------------- get gbdt input data ------------------------------
numerical_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev', 'userRatingCount',
                      'userAvgRating', 'userRatingStddev']
category_features = ['movieGenre1', 'movieGenre2', 'movieGenre3', 'userGenre1', 'userGenre2', 'userGenre3',
                     'userGenre4', 'userGenre5']
label = ['label']


def get_data(path, flag):
    parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    df = pd.read_csv(parent_dir + path)
    df = df[numerical_features + category_features + label]
    df['flag'] = flag
    return df


gbdt_train_df = get_data("/data/trainingSamples.csv", "train")
gbdt_test_df = get_data("/data/testSamples.csv", "test")

gbdt_df = pd.concat([gbdt_train_df, gbdt_test_df])
gbdt_df = pd.get_dummies(data=gbdt_df, columns=category_features)

x_gbdt_train_df = gbdt_df[gbdt_df['flag'] == "train"]
x_gbdt_train_df.pop("flag")
y_train = x_gbdt_train_df.pop(*label)

x_gbdt_test_df = gbdt_df[gbdt_df['flag'] == "test"]
x_gbdt_test_df.pop("flag")
y_test = x_gbdt_test_df.pop(*label)


# ---------------------- build gbdt ------------------------------
n_estimators = 20
gbdt_model = lgb.LGBMClassifier(objective='binary', min_child_weight=0.5, colsample_bytree=0.7,
                                num_leaves=35, learning_rate=0.01, n_estimators=n_estimators, random_state=2021)
gbdt_model.fit(x_gbdt_train_df, y_train)


# ---------------------- get lr input data ------------------------------
def get_gbdt_pred(x_df):
    """
    :param x_df: 输入到gbdt的数据特征
    :return: 输入到lr的数据特征
    """
    # 拿到每条样本落到了每棵树的哪个叶子节点上
    x_lr_list = gbdt_model.predict(x_df, pred_leaf=True)
    lr_feats_name = ['gbdt_leaf_' + str(i) for i in range(n_estimators)]
    x_lr_df = pd.DataFrame(x_lr_list, columns=lr_feats_name)
    # 对叶子节点进行one-hot编码
    x_lr_df = pd.get_dummies(x_lr_df, columns=lr_feats_name)
    return x_lr_df


x_lr_train_df = get_gbdt_pred(x_gbdt_train_df)
x_lr_test_df = get_gbdt_pred(x_gbdt_test_df)

# ---------------------- build lr ------------------------------
lr_model = LogisticRegression()
lr_model.fit(x_lr_train_df, y_train)

# ---------------------- evaluate ------------------------------
train_pred_score = lr_model.predict_proba(x_lr_train_df)[:, 1]
train_auc = roc_auc_score(y_train, train_pred_score)
train_log_loss = log_loss(y_train, train_pred_score)
print("train_log_loss: ", train_log_loss)
print("train_auc: ", train_auc)

test_pred_score = lr_model.predict_proba(x_lr_test_df)[:, 1]
test_auc = roc_auc_score(y_test, test_pred_score)
test_log_loss = log_loss(y_test, test_pred_score)
print("test_log_loss: ", test_log_loss)
print("test_auc: ", test_auc)

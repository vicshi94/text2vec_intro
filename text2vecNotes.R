## text2vec 简单介绍
# 首先加载text2vec包
suppressMessages(library(text2vec))
# supreeMessages()函数是为了不显示加载包时的信息

# 读取数据 --------------------------------
suppressMessages(library(data.table))
data("movie_review")
setDT(movie_review) # 将数据转换为data.table格式
setkey(movie_review, id) # 设置主键

set.seed(20240223) # 设置随机seed以便结果可重复
train_idx <- sample(nrow(movie_review), 0.8 * nrow(movie_review)) # 生成训练集索引
train_set <- movie_review[train_idx] # 生成训练集
test_set <- movie_review[-train_idx] # 生成测试集
print(head(train_set)) # 查看训练集前几行

# 总览 --------------------------------
## 1. 预处理--------------------------------
# 创建一个迭代器，用于将文本转换为小写并分词
# An iterator is an object that traverses a container. A list is iterable.
it_train = itoken(
  train_set$review, # 语料
  preprocessor = tolower, # 将文本转换为小写
  tokenizer = word_tokenizer, # 分词器
  ids = train_set$id, # documents的ID，不设置会自动分配
  progressbar = TRUE # 是否显示进度条
  )

## 2. 创建词汇表 --------------------------------
# 词汇表是所有文档中独特单词的列表
vocab = create_vocabulary(it_train)
# print(vocab)

## 3. 向量化词汇表 --------------------------------
# 创建一个后续可用于各种文本分析所需的矩阵分解的单词数据结构
vectorizer = vocab_vectorizer(vocab)
## 4. 使用迭代器和向量化词汇表，形成文本矩阵 --------------------------------
# 文档-词项矩阵（DTM）
dtm_train = create_dtm(it_train, vectorizer)
print(dim(as.matrix(dtm_train))) # 查看DTM的维度
# output: 4000 38350
# 4000是文档数，38350是词汇表的长度
## 5. N-gram --------------------------------
vocab = create_vocabulary(
  it_train,
  ngram = c(1, 2) # ngram_min = 1, ngram_max = 2
)
# print(vocab)

# 由于增加了N-gram，词汇表的长度会增加，因此需要剪枝
suppressMessages(library(glmnet))
vocab = prune_vocabulary(
  vocab, 
  term_count_min = 10, # 单词最小出现次数
  doc_proportion_max = 0.5 # 单词在所有文档中最大出现比例
  )
# print(vocab)

ngram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, ngram_vectorizer)
NFOLDS = 5
res = cv.glmnet(
  x = dtm_train, 
  y = train_set$sentiment, 
  family = "binomial", # 使用logit回归
  alpha = 1, # 1 for lasso, 0 for 岭回归
  type.measure = "auc", # 评价指标
  nfolds = NFOLDS, # 交叉验证折数, 默认为10, 最小为3
  keep = TRUE, # 是否保留交叉验证结果
  thresh = 1e-3, # 停止迭代阈值
  maxit = 1e3 # 最大迭代次数
  )
)
# more details about cv.glmnet: https://glmnet.stanford.edu/articles/glmnet.html

plot(res)
# 横轴是log(λ)，纵轴是交叉验证的AUC值。λ是正则化参数，λ越大，正则化效果越强，系数越小。交叉验证的AUC值越大，模型效果越好。

# 查看cv.glmnet的结果中包含的内容
# print(names(res)) 
# "lambda"是正则化参数λ的取值，"cvm"是交叉验证的均值，"cvsd"是交叉验证的标准差，"cvup"是交叉验证的上界，"cvlo"是交叉验证的下界，"nzero"是非零系数的个数，"name"是模型名称，"glmnet.fit"是glmnet模型，"lambda.min"是最小的λ，"lambda.1se"是1标准误的λ。
# 输出结果中，"lambda.min"和"lambda.1se"是两个重要的参数。"lambda.min"是交叉验证AUC最大的λ，"lambda.1se"是交叉验证AUC在最大值的1标准误内的λ。
print(paste("max AUC =", round(max(res$cvm), 4))) # 查看最大的交叉验证AUC值
print(res$cvsd[which.max(res$cvm)]) # 查看最大的交叉验证AUC值对应的标准差
print(res$lambda[which.max(res$cvm)]) # 查看最大的交叉验证AUC值对应的λ

# 测试集
it_test = itoken(
  test_set$review, 
  tolower, 
  word_tokenizer, 
  ids = test_set$id, 
  progressbar = FALSE
  )
dtm_test = create_dtm(it_test, ngram_vectorizer)
preds = predict(res, newx = dtm_test, type = "response")[,1] # 预测概率
glmnet:::auc(test_set$sentiment, preds) # 计算AUC值

## 6. 利用TF-IDF优化DTM --------------------------------
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)
dtm_test = create_dtm(it_test, vectorizer)

tfidf = TfIdf$new() # 创建一个TF-IDF对象
dtm_train_tfidf = fit_transform(dtm_train, tfidf) # 训练集TF-IDF
dtm_test_tfidf = transform(dtm_test, tfidf) # 测试集TF-IDF

res = cv.glmnet(x = dtm_train_tfidf, y = train_set[['sentiment']], 
                family = 'binomial', 
                alpha = 1,
                type.measure = "auc",
                nfolds = NFOLDS,
                thresh = 1e-3,
                maxit = 1e3)
print(paste("max AUC =", round(max(res$cvm), 4)))

## 7. 词嵌入 --------------------------------
suppressMessages(library(magrittr))
data("movie_review")
it = itoken(
  movie_review$review, 
  tolower, 
  word_tokenizer
  )
v = create_vocabulary(it) %>% prune_vocabulary(term_count_min=10)
vectorizer = vocab_vectorizer(v)
tcm = create_tcm(
  it, vectorizer, 
  grow_dtm = FALSE, # 生成TCM过程中不生成DTM
  skip_grams_window = 5L) # window size of context
print(dim(tcm))

# GloVe
glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(
  tcm, 
  n_iter = 25, 
  convergence_tol = 0.01, 
  n_threads = 8
  )
wv_context = glove$components
word_vectors = wv_main + t(wv_context)

#Pass: w=word, d=dist matrix, n=nomber of close words
findCloseWords = function(w,d,n) {
  words = rownames(d)
  i = which(words==w)
  if (length(i) > 0) {
    res = sort(d[i,])
    print(as.matrix(res[2:(n+1)]))
  } 
  else {
    print("Word not in corpus.")
  }
}

#Make distance matrix
d = dist2(word_vectors, method="cosine")  #Smaller values means closer
print(dim(d))

findCloseWords("woman",d,10)

# 8.LDA主题模型 --------------------------------
# 可以做，但不展示了，主题模型有很多其他包可以参考。
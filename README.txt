# It is a NLP text classify project

# Instruction
    1. Classification of hotel evaluation information based on tf-idf algorithm containing information entropy,
    which including data pre-processing: missing value processing, invalid data cleaning;
    2. Segment the comment and remove the stop words;
    3. Use word2vec to convert word vector;
    4. Calculate tf-idf weight and information entropy tf-idf weight;
    5. Use SVM to train the data;
    6. Performance evaluation comparison;

# Project Directory:
    > collection_data: the middle result or final result produced by the py files.
    > text_resource: NLP material collected on https://github.com/SophonPlus/ChineseNlpCorpus, thanks to the author.
    > data_preview.py: 数据预处理：填充缺失值，写入data,label文件
    > fea_extract.py: 特征提取
    > tf-idf.py: 基于tf-idf算法计算每个单词的权重，从而得到句子的权重
    > training.py: 对训练集进行训练(SVM)

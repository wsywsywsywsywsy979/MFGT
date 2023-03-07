## MFGT ##
1. 解释：mfa+fbank+gmm+tsne pipeline
2. sklearn_lib：1.1.3版本的scikit-learn 库，可使用pip install 安装，主要用于理解gmm的原理。 
3. basic_operator.py：放置了抽取常见特征fbank等需要的分步函数
4. basic_operator2.py：放置了抽取fbank特征，准备训练数据，训练GMM，可视化的函数
5. my_gmm.py：仿照sklearn中的GMM，手动实现的gmm代码
6. data：放置了“five”对应的
    1. 音频（five.wav）；
    2. 转录本（five.txt）；
    3. 使用mfa切分的对齐（five.TextGrid）；
    4. 使用basic_operator.py中的函数抽取的fbank特征（five_fbank.npy）及对应可视化图（five_fbank.png）
    5. five_xandy：根据对齐，fbank得到的用于训练gmm模型的x（five_x.npy）和y（five_y.npy）

##### note #####
如对代码有问题或建议，欢迎commit issue~
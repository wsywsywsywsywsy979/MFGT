import librosa
from basic_operator import pre_emphasis,framing,add_window,stft,mel_filter,log_pow,plot_spectrogram
import numpy as np
import textgrid
import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture as GMM # sklearn中的GMM
from my_gmm import GMM
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse

def extract_fbank(path):
    """
    function：调用basic_operator.py中的函数抽取fbank特征，并画图
    para:
        path: 要抽取特征的音频文件
    return: None
    """
    data,fs=librosa.load(path) # (22050)
    step1   =   pre_emphasis(data) # 预加重 (22050,)
    step2   =   framing(step1,fs) # 分帧 (99,551) 
    step3   =   add_window(step2,fs) # 加窗 (99,551)
    step4   =   stft(step3) # 幅值平方 (99,257) 此处得到的全是整数
    step5   =   mel_filter(step4, fs) # mel滤波 (99,80) 这个也全是整数
    fbank   =   log_pow(step5) # 对数功率 (99,80) 
    plot_spectrogram(fbank.T, ylabel='Filter Banks',png_name="fbank.png")  
    np.save("fbank", fbank.astype(np.float32), allow_pickle=False)

def make_x_and_y_accord_phone(fbank_path,textgrid_path):
    """
    function:根据对齐数据和fbank制作用于训练gmm的x和y数据
    para：
        fbank_path：抽取保存的fbank数据路径
        textgrid_path：对齐数据的路径
    return: None
    """
    fbank=np.load(fbank_path) # (99,80)
    tg = textgrid.TextGrid() 
    tg.read(textgrid_path)
    x=[]
    y=[]
    maxindex=int(tg.tiers[1].maxTime*100)-1
    for tmp in tg.tiers[1].intervals:
        if len(tmp.mark)==0: # silence也给一个标号 #
            term="#"
        else:
            term=tmp.mark
        start_time=tmp.minTime
        end_time=tmp.maxTime
        start_index=int(start_time*100) 
        end_index=int(end_time*100) # 25
        a=1
        if end_index>maxindex:
            end_index=maxindex
        for i in range(start_index,end_index):
            x.append(fbank[i]) # 此处可能需要拉平,不用拉平，就是一维的，(80,)
            y.append(term)
        # break
    x=np.array(x) # (99,80)
    y=np.array(y) # (99,)
    np.save("x",x.astype(np.float32), allow_pickle=False)
    np.save("y",y,allow_pickle=False)
    
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    function：绘制具有给定位置和协方差的椭圆
    parrotron：
        position：
        convariance：协方差
    return None
    """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
 
def GMM_and_plot_accord_phone(x_path,y_path,n_class,plt_c):
    """
    function：使用准备好的x和y训练sklearn中的GMM模型，并可视化
    para：
        x_path：准备的x数据 （npy类型）
        y_path：准备的y数据 （npy类型）
        n_class：进行聚类的类别数
        plt_c：选择画的图的类型：
            1：散点图（无gmm patch）
            2：文本图（无gmm patch）
            3. 文本gmm图
    return: None
    """
    x=np.load(x_path) # (99,80)
    y=np.load(y_path) # (99,)
    
    X_tsne = TSNE(n_components=2,perplexity=30).fit_transform(x)  # (99,2) 
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)  # (99,2) 
    
    #------sklearn 中的GMM-------------------------
    # gmm = GMM(n_components=n_class).fit(X_norm)
    # labels = gmm.predict(X_norm)
    #---------------------------------------------
    gmm = GMM(n_class=n_class)
    gmm.fit(X_norm)
    if plt_c==1:
        labels = gmm.pred(X_norm)    
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=40, cmap='viridis') 
        plt.savefig("phone_gmm_result_point.png")
        plt.show()
    elif plt_c==2:
        for i in range(len(X_norm)):    
            plt.text(X_norm[i,0], X_norm[i,1], s=y[i],fontdict={'weight': 'bold', 'size':9})
        plt.savefig("phone_gmm_result_text.png")
        plt.show()
    else:
        for i in range(len(X_norm)):    
            plt.text(X_norm[i,0], X_norm[i,1], s=y[i],fontdict={'weight': 'bold', 'size':9})
          
        #------sklearn 中的GMM----------------------------------------------------  
        # w_factor = 0.2 / gmm.weights_.max()
        # for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):   
        #--------------------------------------------------------------------------
        w_factor = 0.2 / gmm.class_prob.max()
        for pos, covar, w in zip(gmm.mus, gmm.vars, gmm.class_prob):
            draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.savefig("gmm_text.png")  
        plt.show()

def plot_gmm(gmm, X, y,label=True, ax=None):
    """
    function: 根据给定的gmm模型和二维数据画散点图
    para：
        gmm：训练好的gmm模型
        X：二维数据
    """
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.savefig("gmm_point.png")  
    plt.show()
                       
def plot_gmm_and_text(X_norm,y,gmm):
    """
    fanction:绘制带有GMM，text的图
    para:
        X_norm：降维并正则化后的数据
        gmm：使用X_norm训练的gmm模型
    """
    for i in range(len(X_norm)):    
        plt.text(X_norm[i,0], X_norm[i,1], s=y[i],fontdict={'weight': 'bold', 'size':9})
    w_factor = 0.2 / gmm.class_prob.max()
    from basic_operator2 import draw_ellipse
    for pos, covar, w in zip(gmm.mus, gmm.vars, gmm.class_prob):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.savefig("gmm_text.png")   
    plt.show()
    
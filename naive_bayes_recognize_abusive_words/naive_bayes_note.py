import numpy as np

'''
贝叶斯算法

A. 任务
给定一个句子 Sentence {word_i}
判定Sentence是否偏向Abusive or Netural?
P(Abusive|Sentence) vs P(Netural|Sentence)

B. 贝叶斯算法
P(Abusive|Sentence) = P(Sentence|Abusive) * P(Abusive) / P(Sentence)
P(Netural|Sentence) = P(Sentence|Netural) * P(Netural) / P(Sentence)

忽略相同分母P(Sentence)， P(Sentence|Abusive) = P(word_1|Abu) * P(word_2|Abu) * ...

C. 准备过程
P(Abusive) 和 P(Netural) 取决于历史句库；
历史库由historyData载入，将句库所有的单词制做成词库，并将句子分类为abusive 和 netural 两种
计算词库中所有单词来源于A类或N类句子的概率，制做成概率向量 P(words|Abusive)

以其中一个类别为例, 对于任意属于词库{words}的一个单词{w}
P(w|Abusive) 是单一单词来源于 Abusive 句子的概率，其计算方法是
P(w|Abusive) = Count(w in A)/Size(A)
Size(A) ：统计历史句库中所有 Abusive 句子的单词个数之和（不删除重复单词）
Count(w in A) ：统计 word_i 在 所有 Abusive 句子中的出现次数

D. 贝叶斯分类
log(P(Sentence|Abusive)) = Set(word_i) * log(P(words|Abusive))
词库{words}中未出现于 Sentence 的 Set 是零，相当于不被选取
如果 Sentence中有词库外的新词，只能舍去不予考虑

E. 优化
1.解决有些单词只在 Netural 句中出现而不是 Abusive 句中出现的方法是把所有单词的初始次数都设定为1
这样就不会出现零概率情况
2.将 Count(A)的初始值设定为 2（no idea）
3.概率连乘会出现非常小的数字乃至超出 Python的精读。解决办法就是用 log 代替连乘。

F. 延伸
贝叶斯算法可以用于识别一个文章的作者。由于每个作者的行文风格导致单词出现频率不同或者出现个性单词
，我们可以将所有备选作者的历史文章做成词库，并将待识别文章输入。该场景通常需要大量数据，有些优化
算法可以忽略。

另外一个重要运用是垃圾邮件过滤

G. 思考
当词库选取相等数量的 Abusive 和 Netural （每个作家的历史文章数相同）时。P(Abusive)=P(Netural)
贝叶斯算法本质上是在比较测试 Sentence 中每个单词来自不同类别句子的可能性，然后将可能性叠加比较

是否可以将不同单词做加权处理？比如某个单词比如（“FUCK”）的出现可以很大程度上确定测试句子的类型
此时选取适当句库（或者当句库非常大的时候，Size(A)>>1）,P("FUCK"|Abu)会相对较高，此时是否可以将
 P("FUCK"|Abu)适当加权（比如引入指数函数）？

P(Abusive)和P(Netural)的选择是佛会影响到算法准确度?是P(Abusive)=P(Netural)更好，还是
随机选取历史数据，让P(Abusive)与P(Netural)取决于大数据统计结果更好？

'''


def loadHistoryData():
    historyData = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVector = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return historyData, classVector

def createwordSetForHistory(historyData):
    wordSetofHistoryData = set([])  # create empty set
    for document in historyData:
        wordSetofHistoryData = wordSetofHistoryData | set(document)  # union of the two sets
    return list(wordSetofHistoryData)

def setSentencetoVector(vocabList, inputSet):
    returnVec = [0] * len(vocabList)# [0,0......]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            # This step is for testing sentence, not for history data.
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNaiveBayes(trainMatrix, classVector):

    numHistorySentence = len(trainMatrix)
    sizeHistoryWordSet = len(trainMatrix[0])
    pAbuSentence = np.sum(classVector) / float(numHistorySentence)
    # 构造单词出现次数列表
    # p0Num 正常的统计
    # pAbuWord 侮辱的统计
    pNeuWord = np.ones(sizeHistoryWordSet)#[0,0......]->[1,1,1,1,1.....]
    pAbuWord = np.ones(sizeHistoryWordSet)

    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0Denom 正常的统计
    # p1Denom 侮辱的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numHistorySentence):
        if classVector[i] == 1:
            # 累加辱骂词的频次
            pAbuWord += trainMatrix[i]
            # 对每篇文章的辱骂的频次 进行统计汇总
            p1Denom += np.sum(trainMatrix[i])
        else:
            pNeuWord += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = np.log(pAbuWord / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = np.log(pNeuWord / p0Denom)
    return p0Vect, p1Vect, pAbuSentence

def classifyNB(testVector, p0Vec, p1Vec, pAbu):
    """
    使用算法: 
        # 将乘法转换为加法
        乘法: P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法: P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param testVector: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pAbu: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 大家可能会发现，上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了，
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是: 这里的 testVector * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = np.sum(testVector * p1Vec) + np.log(pAbu) # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = np.sum(testVector * p0Vec) + np.log(1.0 - pAbu) # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    historyData, classVector = loadHistoryData()
    wordSetofHistory = createwordSetForHistory(historyData)
    trainingMatrix = []
    for sentence in historyData:
        trainingMatrix.append(setSentencetoVector(wordSetofHistory, sentence))
    # for x in trainingMatrix:
        # print(x)
    # quit()
    p0V, p1V, pAb = trainNaiveBayes(np.array(trainingMatrix), classVector)
    # print(p0V, p1V, pAb)
    # quit()

    testEntry = ['my','stupid','garbage','boulder']
    thisDoc = np.array(setSentencetoVector(wordSetofHistory, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # testEntry = ['stupid', 'garbage']
    # thisDoc = np.array(setSentencetoVector(wordSetofHistory, testEntry))
    # print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == "__main__":
    testingNB()


























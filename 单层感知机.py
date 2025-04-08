import re
import jieba
import numpy as np

#1. 首先准备数据集（中文邮件示例）
# 垃圾邮件样本
spam_emails = [
    "恭喜您获得100万奖金！点击链接领取：http://获奖.com",
    "双11限时特惠，所有商品1折起！",
    "您的银行账户存在风险，请立即验证",
    "快速赚钱秘籍，月入10万不是梦",
    "独家投资机会，年回报率500%"
]
# 正常邮件样本
ham_emails = [
    "项目会议安排在明天下午3点",
    "您的快递已到菜鸟驿站，取件码1234",
    "关于项目进度的报告已发到您的邮箱",
    "本周五部门聚餐，地点：公司餐厅",
    "请查收附件中的财务报表"
]

# 创建标签：1表示垃圾邮件，0表示正常邮件
labels = [1]*5 + [0]*5
emails = spam_emails + ham_emails

#2. 文本预处理函数
def preprocess_text(text):
    """清洗和预处理中文邮件文本"""
    # 去除特殊内容
    text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
    text = re.sub(r'http\S+', '', text)  # 去除网址
    text = re.sub(r'\S*@\S*\s?', '', text)  # 去除邮箱
    text = re.sub(r'[^\u4e00-\u9fa5\s]', ' ', text)  # 只保留中文和空格
    
    # 中文分词
    words = jieba.lcut(text)
    return ' '.join(words)
#3.特征提取
from collections import defaultdict

def build_features(emails, top_n=50):
    """构建词袋特征向量"""
    # 统计词频
    word_counts = defaultdict(int)
    for email in emails:
        for word in email.split():
            word_counts[word] += 1
    
    # 选择最高频的top_n个词作为特征
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]
    vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
    
    # 构建特征矩阵
    features = np.zeros((len(emails), len(vocab)))
    for i, email in enumerate(emails):
        for word in email.split():
            if word in vocab:
                features[i, vocab[word]] += 1
                
    return features, vocab
#4.单层感知机实现
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1  # 初始化权重
        self.bias = 0  # 初始化偏置
    
    def activate(self, x):
        """阶跃激活函数"""
        return 1 if x >= 0 else 0
    
    def train(self, X, y, lr=0.01, epochs=100):
        """训练模型"""
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # 计算预测值
                prediction = self.activate(np.dot(X[i], self.weights) + self.bias)
                
                # 计算误差
                error = y[i] - prediction
                total_error += abs(error)
                
                # 更新权重和偏置
                self.weights += lr * error * X[i]
                self.bias += lr * error
            
            print(f"第{epoch+1}轮训练，总误差：{total_error}")
            
            # 如果误差为0则提前停止
            if total_error == 0:
                break
    
    def predict(self, X):
        """预测新样本"""
        results = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            results.append(self.activate(linear_output))
        return np.array(results)
#5.训练和测试流程
# 预处理所有邮件
processed_emails = [preprocess_text(email) for email in emails]

# 构建特征矩阵
X, vocab = build_features(processed_emails)
y = np.array(labels)

# 初始化并训练感知机
perceptron = Perceptron(input_size=X.shape[1])
perceptron.train(X, y)

# 测试新邮件
test_emails = [
    "点击领取您的百万奖金",  # 垃圾邮件
    "明天上午10点开会",    # 正常邮件
    "限时特价最后一天",    # 垃圾邮件
    "您的快递已送达"       # 正常邮件
]

# 预处理测试邮件
processed_test = [preprocess_text(email) for email in test_emails]

# 转换为特征向量
X_test = np.zeros((len(test_emails), len(vocab)))
for i, email in enumerate(processed_test):
    for word in email.split():
        if word in vocab:
            X_test[i, vocab[word]] += 1

# 预测并显示结果
predictions = perceptron.predict(X_test)
for email, pred in zip(test_emails, predictions):
    print(f"邮件内容：{email}")
    print(f"预测结果：{'垃圾邮件' if pred == 1 else '正常邮件'}\n")
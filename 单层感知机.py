import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import re
import jieba
import numpy as np
from collections import defaultdict

# 1. 数据准备
spam_emails = [
    "恭喜您获得100万奖金！点击链接领取：http://获奖.com",
    "双11限时特惠，所有商品1折起！",
    "您的银行账户存在风险，请立即验证",
    "快速赚钱秘籍，月入10万不是梦",
    "独家投资机会，年回报率500%"
]

ham_emails = [
    "项目会议安排在明天下午3点",
    "您的快递已到菜鸟驿站，取件码1234",
    "关于项目进度的报告已发到您的邮箱",
    "本周五部门聚餐，地点：公司餐厅",
    "请查收附件中的财务报表"
]

# 2. 预处理函数
def preprocess_text(text):
    """清洗和分词中文文本"""
    text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
    text = re.sub(r'http\S+', '', text)  # 去除URL
    text = re.sub(r'\S*@\S*\s?', '', text)  # 去除邮箱
    text = re.sub(r'[^\u4e00-\u9fa5\s]', ' ', text)  # 只保留中文和空格
    words = jieba.lcut(text)
    return ' '.join(words)

# 3. 特征提取
def build_features(emails, top_n=50):
    """构建词袋特征矩阵"""
    word_counts = defaultdict(int)
    for email in emails:
        for word in email.split():
            word_counts[word] += 1
    
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]
    vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
    
    features = np.zeros((len(emails), len(vocab)))
    for i, email in enumerate(emails):
        for word in email.split():
            if word in vocab:
                features[i, vocab[word]] += 1
                
    return features, vocab

# 4. 感知机实现
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0
    
    def activate(self, x):
        return 1 if x >= 0 else 0
    
    def train(self, X, y, lr=0.01, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.activate(np.dot(X[i], self.weights) + self.bias)
                error = y[i] - prediction
                total_error += abs(error)
                self.weights += lr * error * X[i]
                self.bias += lr * error
            if total_error == 0:
                break
    
    def predict(self, X):
        return np.array([self.activate(np.dot(x, self.weights) + self.bias) for x in X])

# 5. 训练模型
processed_emails = [preprocess_text(email) for email in (spam_emails + ham_emails)]
labels = [1]*5 + [0]*5
X, vocab = build_features(processed_emails)
y = np.array(labels)
perceptron = Perceptron(input_size=X.shape[1])
perceptron.train(X, y)

# 6. Tkinter可视化界面
class EmailClassifierApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("垃圾邮件分类可视化系统")
        self.root.geometry("500x500")
        
        # 颜色配置
        bg_color = "#f0f0f0"
        button_color = "#4a7a8c"
        text_color = "#333333"
        
        self.root.configure(bg=bg_color)
        
        # 标题
        title_frame = tk.Frame(self.root, bg=bg_color)
        title_frame.pack(pady=20)
        
        tk.Label(
            title_frame, 
            text="邮件分类系统", 
            font=("微软雅黑", 20, "bold"), 
            fg="#2c3e50", 
            bg=bg_color
        ).pack()
        
        # 输入区域
        input_frame = tk.LabelFrame(
            self.root, 
            text=" 输入邮件内容 ", 
            font=("微软雅黑", 12), 
            bg=bg_color, 
            fg=text_color
        )
        input_frame.pack(pady=10, padx=20, fill="x")
        
        self.text_input = scrolledtext.ScrolledText(
            input_frame, 
            height=8, 
            font=("微软雅黑", 11), 
            wrap=tk.WORD
        )
        self.text_input.pack(pady=5, padx=10, fill="both", expand=True)
        
        # 按钮区域
        button_frame = tk.Frame(self.root, bg=bg_color)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame, 
            text="分类", 
            command=self.classify_email,
            bg=button_color,
            fg="white",
            font=("微软雅黑", 12),
            padx=20,
            relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame, 
            text="清空", 
            command=self.clear_input,
            bg="#e74c3c",
            fg="white",
            font=("微软雅黑", 12),
            padx=20,
            relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=10)
        
        # 测试案例
        test_frame = tk.Frame(self.root, bg=bg_color)
        test_frame.pack(pady=10)
        
        test_cases = [
            ("垃圾邮件示例1", "点击领取您的百万奖金"),
            ("正常邮件示例1", "明天上午10点开会"),
            ("垃圾邮件示例2", "限时特价最后一天"),
            ("正常邮件示例2", "您的快递已送达")
        ]
        
        for text, content in test_cases:
            tk.Button(
                test_frame, 
                text=text, 
                command=lambda c=content: self.load_test_case(c),
                bg="#3498db",
                fg="white",
                font=("微软雅黑", 10),
                relief=tk.FLAT
            ).pack(side=tk.LEFT, padx=5)
        
        # 结果区域
        result_frame = tk.LabelFrame(
            self.root, 
            text=" 分类结果 ", 
            font=("微软雅黑", 12), 
            bg=bg_color, 
            fg=text_color
        )
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            height=12, 
            font=("微软雅黑", 11), 
            wrap=tk.WORD,
            state="disabled"
        )
        self.result_text.pack(pady=5, padx=10, fill="both", expand=True)
    
    def classify_email(self):
        email_content = self.text_input.get("1.0", "end-1c").strip()
        if not email_content:
            messagebox.showwarning("提示", "请输入邮件内容！")
            return
        
        # 预处理和预测
        processed = preprocess_text(email_content)
        features = np.zeros((1, len(vocab)))
        for word in processed.split():
            if word in vocab:
                features[0, vocab[word]] += 1
        
        prediction = perceptron.predict(features)[0]
        result = "垃圾邮件" if prediction == 1 else "正常邮件"
        
        # 显示结果
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        
        # 结果样式
        self.result_text.tag_config("header", font=("微软雅黑", 12, "bold"))
        self.result_text.tag_config("spam", foreground="red", font=("微软雅黑", 12, "bold"))
        self.result_text.tag_config("ham", foreground="green", font=("微软雅黑", 12, "bold"))
        self.result_text.tag_config("keyword", foreground="blue")
        
        self.result_text.insert("end", "邮件内容:\n", "header")
        self.result_text.insert("end", f"{email_content}\n\n")
        
        self.result_text.insert("end", "分类结果: ", "header")
        self.result_text.insert("end", f"{result}\n\n", "spam" if prediction == 1 else "ham")
        
        self.result_text.insert("end", "关键词分析:\n", "header")
        
        # 关键词权重分析
        keywords = []
        for word in processed.split():
            if word in vocab:
                weight = perceptron.weights[vocab[word]]
                keywords.append((word, weight))
        
        keywords.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for word, weight in keywords[:10]:  # 显示前10个关键词
            weight_desc = "正向(垃圾)" if weight > 0 else "负向(正常)"
            self.result_text.insert("end", f"  '{word}': ", "keyword")
            self.result_text.insert("end", f"权重={weight:.3f} ({weight_desc})\n")
        
        self.result_text.config(state="disabled")
    
    def load_test_case(self, content):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("end", content)
        self.classify_email()
    
    def clear_input(self):
        self.text_input.delete("1.0", "end")
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()

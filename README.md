# machine-learning-nlp
北京邮电大学机器学习创新实践课程大作业，实现了类 jieba 库的中文分词与标记算法以及 TextRank 关键词提取算法
### 运行方法
`python main.py`
- 是否预测词性：`--with_tag`
- 是否采用有监督方法：`--supervised`
- 训练单元：`--atom`（只认可 `letter` 或 `word`）
- 重新训练：`--retrain`
- 是否进行测试：`--eval`
- 是否字使用预测模型进行测试：`--only_pred`
- 打印帮助：`-h, --help`
### 代码结构
- `data/`：所有的数据集、词典与模型参数
- `dic.py`：字典、词典的构造
- `main.py`：主程序
- `pred.py`：HMM 模型实现
- `extractor`：TextRank 算法实现
- `seg.py`：词典与预测结合的算法模型实现
- `ds.py`：数据集的构造
- `code.py`：语句-向量转换工具实现
- `utils.py`：一些常用工具函数与工具类的实现
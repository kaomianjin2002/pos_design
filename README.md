# 基于结构化平均感知机的中英文词性标注系统

## 一、项目简介

这是一个**本地运行**的中英文词性标注项目。

它的完整流程是：

1. 准备原始语料；
2. 把原始语料转换成标准 CoNLL 格式；
3. 使用**结构化平均感知机**训练词性标注模型；
4. 使用**维特比解码**完成整句预测；
5. 通过本地网页查看训练结果、预测结果、词性分布和混淆矩阵。

如果你是第一次接触自然语言处理，也可以直接照着本 README 操作。本文会尽量把每一步都讲清楚，让**完全没有经验的小白**也能上手。

---

## 二、这个项目能做什么

本项目已经实现以下功能：

### 1. 语料转换
- 支持中文 `词/词性` 原始格式。
- 支持英文 `word/词性` 原始格式。
- 支持把语料转换成标准 CoNLL 格式。
- 提供命令行脚本 `backend/data_converter.py`。

### 2. 本地训练
- 使用**结构化平均感知机**训练序列标注模型。
- 使用**维特比解码**进行全局最优路径搜索。
- 同时训练一个**最高频词性基线**作为对照模型。

### 3. 本地预测
- 输入一句中文或英文。
- 返回每个词对应的**中文词性名称**。
- 例如：名词、动词、形容词、限定词、专有名词、副词等。

### 4. 结果统计
- 验证集准确率。
- 测试集准确率。
- 最高频词性基线准确率。
- 词性分布。
- 混淆矩阵。

### 5. 网页交互
- 在浏览器中直接点击按钮训练。
- 在网页中输入句子做预测。
- 在网页中查看统计结果。

---

## 三、项目目录详解

下面是项目目录，以及每个文件/文件夹的用途。

```text
pos_design/
├── README.md
├── development.md
├── prd.md
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   ├── data_converter.py
│   ├── main.py
│   ├── data/
│   │   ├── raw/
│   │   │   ├── chinese_sample.txt
│   │   │   └── english_sample.txt
│   │   └── processed/
│   │       ├── chinese_sample.conll
│   │       └── english_sample.conll
│   ├── models/
│   │   ├── structured_perceptron.json
│   │   └── training_stats.json
│   └── tests/
│       ├── conftest.py
│       └── test_system.py
└── frontend/
    └── index.html
```

### 1. 根目录文件说明

#### `README.md`
就是你现在看的说明文档。

作用：
- 介绍项目；
- 告诉你怎么启动；
- 告诉你每个文件做什么；
- 告诉你怎么训练和预测；
- 帮你快速排错。

#### `development.md`
技术架构文档。

作用：
- 说明项目应该采用的技术路线；
- 说明推荐目录结构；
- 说明后端、前端、模型层的大致设计。

#### `prd.md`
产品需求文档。

作用：
- 说明项目最终想实现哪些功能；
- 说明训练、预测、统计、可视化这些需求；
- 说明为什么需要这个系统。

#### `requirements.txt`
Python 依赖清单。

作用：
- 安装项目运行需要的包；
- 保证不同环境尽量一致。

---

## 四、后端各文件详细说明

### 1. `backend/core.py`
这是**模型核心文件**。

它实现了三个最重要的部分：

#### （1）特征提取器
负责从句子中抽取训练和预测需要的特征，例如：
- 当前词；
- 前一个词；
- 后一个词；
- 英文前缀和后缀；
- 是否包含数字；
- 是否大写；
- 上一个词的词性到当前词性的转移特征。

#### （2）结构化平均感知机
这是本项目的主模型。

你可以把它理解成：
- 它会不断比较“真实词性序列”和“模型预测序列”；
- 如果预测错了，就调整参数；
- 训练很多轮以后，模型会逐渐学会哪类词更像名词、动词、形容词等。

#### （3）最高频词性基线
这是一个很简单的对照模型。

它的思想非常朴素：
- 一个词在训练里最常见的词性是什么；
- 预测时就优先使用那个最常见词性。

它的作用不是为了效果最好，而是为了做比较：
- 如果你的主模型还不如这个基线，说明主模型还有改进空间。

---

### 2. `backend/utils.py`
这是**工具函数文件**，相当于后端的数据工具箱。

它主要做这些事情：

#### （1）判断语言
自动判断输入文本更接近中文还是英文。

#### （2）原始语料转 CoNLL
把下面这种格式：

```text
中国/名词 政府/名词 宣布/动词
```

转换成标准 CoNLL：

```text
1	中国	_	名词	名词	_	_	_	_	_
2	政府	_	名词	名词	_	_	_	_	_
3	宣布	_	动词	动词	_	_	_	_	_
```

#### （3）读取 CoNLL 文件
把 CoNLL 文件读进 Python，供训练使用。

#### （4）切分数据集
把数据切成：
- 训练集；
- 验证集；
- 测试集。

比例是 `8:1:1`。

#### （5）低频词处理
对于出现次数很低的词，进行统一归一化，减少稀疏问题。

#### （6）词性中文化
如果你导入的是像 `NN`、`NNS`、`VBZ`、`DT` 这样的英文词性代码，系统会尽量映射成中文名称，例如：
- `NN` → 名词
- `VBZ` → 动词
- `NNS` → 名词（复数名词并入名词）
- `DT` → 限定词

这样在网页和接口里，用户看到的是中文。

---

### 3. `backend/data_converter.py`
这是**命令行转换脚本**。

它适合什么场景？

比如你拿到了一份原始语料：
- 中文文件；
- 英文文件；
- 每个词后面带词性；

但它还不是 CoNLL 标准格式。

这时你就可以用这个脚本把它转换成标准格式，方便后续训练。

---

### 4. `backend/main.py`
这是**项目服务入口**。

你运行下面的命令：

```bash
python -m backend.main
```

实际上就是启动这个文件。

它会负责：
- 启动本地 HTTP 服务；
- 提供训练接口；
- 提供预测接口；
- 提供统计接口；
- 提供网页首页。

这个文件里最重要的几个接口如下：

#### `/`
打开网页首页。

#### `/health`
检查服务是否正常。

#### `/train`
训练**结构化平均感知机**。

#### `/predict`
输入一句话，返回每个词的中文词性名称。

#### `/stats`
查看训练后的统计结果。

---

## 五、前端文件说明

### `frontend/index.html`
这是项目的网页界面。

网页里可以做三件事：

1. 训练默认模型；
2. 输入句子并预测词性；
3. 查看训练统计结果。

你在页面上会看到：
- 输入框；
- 训练按钮；
- 预测按钮；
- 词性结果表格；
- 词性分布图；
- 混淆矩阵。

整个页面不依赖云服务，访问本地地址就能用。

---

## 六、样例数据文件说明

### 1. `backend/data/raw/chinese_sample.txt`
中文原始样例语料。

特点：
- 使用 `词/词性` 格式；
- 词性名称已经是中文；
- 适合直接给转换脚本使用。

### 2. `backend/data/raw/english_sample.txt`
英文原始样例语料。

特点：
- 单词是英文；
- 词性名称仍然使用中文；
- 便于统一展示。

### 3. `backend/data/processed/chinese_sample.conll`
中文 CoNLL 样例文件。

### 4. `backend/data/processed/english_sample.conll`
英文 CoNLL 样例文件。

这两个文件是处理后的标准数据，可以直接用于训练。

---

## 七、训练后生成的文件说明

训练完成后，会在 `backend/models/` 下生成两个重要文件。

### 1. `structured_perceptron.json`
这是**结构化平均感知机模型文件**。

作用：
- 保存训练后的参数；
- 预测时会读取这个文件；
- 不需要每次预测都重新训练。

### 2. `training_stats.json`
这是**训练统计结果文件**。

里面会保存：
- 训练轮数；
- 验证集准确率；
- 测试集准确率；
- 最高频词性基线准确率；
- 词性集合；
- 词性分布；
- 混淆矩阵。

---

## 八、测试文件说明

### 1. `backend/tests/conftest.py`
测试时的环境准备文件。

### 2. `backend/tests/test_system.py`
自动化测试文件。

当前测试覆盖：
- 原始语料转 CoNLL；
- CoNLL 文件加载；
- 启动服务；
- 调用训练接口；
- 调用预测接口；
- 调用统计接口。

也就是说，这个测试已经覆盖了项目的主流程。

---

## 九、环境准备

## 1. Python 版本
建议使用：

```text
Python 3.10 或更高版本
```

## 2. 安装依赖
在项目根目录执行：

```bash
pip install -r requirements.txt
```

如果你所在环境无法联网，项目中的一部分逻辑仍然有回退处理，但最推荐的方式依然是先安装依赖。

---

## 十、最简单上手流程（适合小白）

如果你完全没接触过这个项目，请严格按照下面步骤来。

### 第 1 步：进入项目目录

```bash
cd /workspace/pos_design
```

### 第 2 步：创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate
```

如果你不知道虚拟环境是什么，也没关系。
它的作用只是把项目依赖和系统环境隔离开，避免冲突。

### 第 3 步：安装依赖

```bash
pip install -r requirements.txt
```

### 第 4 步：启动服务

```bash
python -m backend.main
```

如果启动成功，终端里会看到类似下面的提示：

```text
词性标注系统已启动：http://127.0.0.1:8000
```

### 第 5 步：打开浏览器

访问：

```text
http://127.0.0.1:8000
```

### 第 6 步：点击训练按钮

页面中点击：

```text
训练默认模型（结构化平均感知机）
```

点击之后，系统会自动：
- 读取样例语料；
- 切分训练集/验证集/测试集；
- 训练结构化平均感知机；
- 训练最高频词性基线；
- 生成统计结果；
- 写入模型文件。

### 第 7 步：输入一句话做预测

例如英文：

```text
The student studies NLP carefully
```

例如中文：

```text
中国政府宣布新政策
```

然后点击：

```text
开始标注
```

页面会返回每个词对应的**中文词性名称**。

---

## 十一、命令行使用方法

如果你不想通过网页操作，也可以直接在命令行中使用。

### 1. 手动转换语料

#### 转换中文语料

```bash
python backend/data_converter.py --input backend/data/raw/chinese_sample.txt --output backend/data/processed/chinese_sample.conll --lang zh
```

#### 转换英文语料

```bash
python backend/data_converter.py --input backend/data/raw/english_sample.txt --output backend/data/processed/english_sample.conll --lang en
```

### 2. 命令行训练模型

先启动服务：

```bash
python -m backend.main
```

再打开另一个终端执行：

```bash
python -c "import json; from urllib.request import Request, urlopen; req=Request('http://127.0.0.1:8000/train', data=json.dumps({'iterations': 5}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(urlopen(req).read().decode('utf-8'))"
```

### 3. 命令行预测

```bash
python -c "import json; from urllib.request import Request, urlopen; req=Request('http://127.0.0.1:8000/predict', data=json.dumps({'text':'The cat sits quietly','tokenizer':'whitespace'}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(urlopen(req).read().decode('utf-8'))"
```

### 4. 命令行查看统计

```bash
python -c "from urllib.request import urlopen; print(urlopen('http://127.0.0.1:8000/stats').read().decode('utf-8'))"
```

---

## 十二、接口返回内容说明

### 1. `/train`
返回训练后的统计信息，例如：
- 模型名称；
- 基线模型名称；
- 准确率；
- 词表大小；
- 中文词性集合；
- 中文混淆矩阵；
- 中文词性分布。

### 2. `/predict`
返回：
- 输入的词；
- 原始词性标签；
- 中文词性名称。

网页里优先展示中文词性名称。

### 3. `/stats`
返回最近一次训练保存下来的统计结果。

---

## 十三、词性名称说明

本项目根据你的要求，**模型名称和词性名称统一使用中文展示**。

常见词性示例（粗粒度）：
- 名词（含单复数、专有名词）
- 动词（含时态和人称变化）
- 形容词
- 副词
- 代词
- 介词
- 连词
- 限定词

即使导入的是标准英文词性代码，系统也会尽量把它翻译成中文后再展示。

---

## 十四、常见问题

### 问题 1：预测时报“模型尚未训练”怎么办？
先训练，再预测。

### 问题 2：为什么网页上有两个模型名字？
因为：
- **结构化平均感知机**是主模型；
- **最高频词性基线**是对照模型。

### 问题 3：我可以换成自己的语料吗？
可以。
建议做法是：
1. 先把原始语料整理成 `词/词性` 格式；
2. 再用 `backend/data_converter.py` 转成 CoNLL；
3. 放到 `backend/data/processed/` 目录；
4. 然后训练。

### 问题 4：为什么这个项目不依赖云服务？
因为项目设计目标就是**本地优先**，适合课程设计、实验演示和离线使用。

### 问题 5：我不懂模型，也能用吗？
可以。
你只需要记住三步：
1. 启动服务；
2. 点击训练；
3. 输入句子预测。

---

## 十五、建议的新手使用顺序

如果你现在就想最快跑通，请按下面顺序：

1. `cd /workspace/pos_design`
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python -m backend.main`
6. 打开浏览器访问 `http://127.0.0.1:8000`
7. 点击“训练默认模型（结构化平均感知机）”
8. 输入一句话点击“开始标注”

做到这里，你就已经完整跑通这个项目了。

---

## 十六、一句话总结

> 这个项目是一套本地运行的中英文词性标注系统：先准备或转换语料，再训练结构化平均感知机，最后在网页中用中文查看模型名称、词性名称和统计结果。

---

## 十七、如何使用你自己的语料（重点）

这一节专门回答你提到的“语料文件应该放在哪”。

### 1. 你必须先区分两类目录

项目里和语料相关的目录有两个：

1. `backend/data/raw/`
   - 放**原始语料**。
   - 原始语料就是你自己手上还没转换的文本，比如 `词/词性` 格式。

2. `backend/data/processed/`
   - 放**转换后的 CoNLL 语料**。
   - 训练时系统默认读取这里的 `.conll` 文件。

> 一句话：**原始放 raw，训练放 processed**。

### 2. 推荐的文件命名方式

你可以按下面方式命名，便于后续管理：

- 中文原始语料：`backend/data/raw/my_zh_raw.txt`
- 英文原始语料：`backend/data/raw/my_en_raw.txt`
- 中文转换后语料：`backend/data/processed/my_zh.conll`
- 英文转换后语料：`backend/data/processed/my_en.conll`

### 3. 原始语料格式怎么写

#### 3.1 中文示例（raw）

放到 `backend/data/raw/my_zh_raw.txt`：

```text
中国/名词 政府/名词 发布/动词 新/形容词 方案/名词
学生/名词 正在/副词 学习/动词 算法/名词
```

#### 3.2 英文示例（raw）

放到 `backend/data/raw/my_en_raw.txt`：

```text
The/限定词 students/名词 are/动词 learning/动词 quickly/副词
He/代词 works/动词 in/介词 school/名词
```

### 4. 把原始语料转换成 CoNLL

#### 4.1 转换中文

```bash
python backend/data_converter.py \
  --input backend/data/raw/my_zh_raw.txt \
  --output backend/data/processed/my_zh.conll \
  --lang zh
```

#### 4.2 转换英文

```bash
python backend/data_converter.py \
  --input backend/data/raw/my_en_raw.txt \
  --output backend/data/processed/my_en.conll \
  --lang en
```

### 5. 转换后如何确认文件是对的

你可以先看前几行：

```bash
sed -n '1,20p' backend/data/processed/my_zh.conll
sed -n '1,20p' backend/data/processed/my_en.conll
```

只要看到类似下面结构，说明格式基本正确：

```text
1	The	_	限定词	限定词	_	_	_	_	_
2	students	_	名词	名词	_	_	_	_	_
```

### 6. 训练时会读取哪些文件

默认情况下，调用 `/train` 时会自动读取：

- `backend/data/processed/` 下所有 `.conll` 文件。

也就是说，你只要把你的语料转换后放到这个目录里，直接训练就会被加载。

### 7. 如果你只想训练某几个文件

可以在 `/train` 请求里明确传 `dataset_paths`，例如（命令行示意）：

```bash
python -c "import json; from urllib.request import Request, urlopen; req=Request('http://127.0.0.1:8000/train', data=json.dumps({'iterations': 8, 'dataset_paths': ['backend/data/processed/my_zh.conll', 'backend/data/processed/my_en.conll']}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(urlopen(req).read().decode('utf-8'))"
```

### 8. 你可以直接照着走的“自定义语料完整流程”

1. 把原始文件放到 `backend/data/raw/`。
2. 用 `data_converter.py` 转成 CoNLL 到 `backend/data/processed/`。
3. 启动服务：`python -m backend.main`。
4. 调 `/train` 训练模型。
5. 调 `/predict` 做预测。
6. 调 `/stats` 查看效果。

### 9. 常见错误与对应处理

#### 错误 A：`模型尚未训练，请先执行训练。`
先调用 `/train`，再调用 `/predict`。

#### 错误 B：`没有找到可用于训练的语料文件。`
说明 `backend/data/processed/` 下没有可读的 `.conll` 文件。

检查：

```bash
rg --files backend/data/processed
```

#### 错误 C：训练准确率异常低
通常是语料标签不一致导致的：
- 同一个词类出现多种写法；
- 原始语料里夹杂非法 token；
- 中英文数据混用时标签体系不统一。

建议先用小样本语料调通流程，再逐步加大数据量。

### 10. 语料规模建议（给新手）

- 首次联调：10~50 句（快速验证流程）
- 小规模实验：500~2000 句
- 正式训练：按你机器性能逐步增加

先保证格式正确，再追求规模和指标，这是最稳妥的方式。

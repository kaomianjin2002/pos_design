

# 技术架构文档：基于感知机的中英文词性标注系统 (Local-First 版)

## 1. 项目定位与核心目标
* [cite_start]**项目名称**：基于感知机的词性标注系统设计与实现 [cite: 1, 5]
* [cite_start]**核心算法**：结构化平均感知机 (Structured Averaged Perceptron) + Viterbi 全局解码 [cite: 1, 5]
* **工程原则**：**No Cloud, No Hand-rolling**。优先调用成熟库（如 `conllu` 处理数据，`FastAPI` 构建后端，`Element Plus` 构建 UI），所有服务本地化部署。

## 2. 本地开发环境与技术栈
为了保证项目的成熟度和 Cursor 的生成效率，建议采用以下技术组合：

| 维度 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **语言环境** | Python 3.10+ | [cite_start]核心逻辑实现语言 [cite: 1, 5] |
| **后端框架** | **FastAPI** | 比 Flask 更现代，自带自动文档 (Swagger)，适合 Cursor 快速生成接口 |
| **前端框架** | **Vue 3 (Vite) + Element Plus** | [cite_start]使用成熟的 UI 组件库实现"精美页面"要求 [cite: 5] |
| **数据解析** | `conllu` / `nltk` | [cite_start]专门用于解析 CoNLL 标准格式语料 [cite: 1] |
| **可视化** | **ECharts / v-charts** | [cite_start]本地运行，用于展示词性分布和性能指标 [cite: 1, 5] |
| **数据存储** | 本地 JSON / SQLite | 无需配置数据库服务器，完全本地化 |

---

## 3. 项目目录结构 (Cursor 参考)
请让 Cursor 按照以下结构初始化项目：
```text
/pos-tagger-project
├── /backend                # FastAPI 后端
│   ├── /data
│   │   ├── /raw            # 原始语料（PKU/WSJ 等原始格式）
│   │   └── /processed      # 转换后的标准 CoNLL 文件
│   ├── /models             # 存放训练好的感知机模型权重 (Pickle/JSON)
│   ├── data_converter.py   # 【新增】中英文语料 → CoNLL 格式转换脚本
│   ├── core.py             # 结构化感知机核心算法
│   ├── utils.py            # 数据预处理、特征提取、convert_to_conll()
│   └── main.py             # API 路由与服务启动
├── /frontend               # Vue 3 前端
│   ├── /src
│   │   ├── /components     # UI 组件 (输入框、标注结果表格)
│   │   └── /views          # 主页面与可视化看板
│   └── package.json
└── README.md
```

---

## 4. 核心功能实现逻辑

### 4.1 数据与预处理 (Data Layer)

#### 4.1.0 语料格式统一转换（新增，优先级最高）
在加载语料之前，需先通过转换脚本将原始语料规范化为 CoNLL 格式。

* **新增文件**：`backend/data_converter.py`（独立脚本，可命令行调用）
* **新增函数**：`utils.py` 中的 `convert_to_conll(input_path, output_path, lang)`

**支持的输入格式**：

| 语言 | 输入格式 | 示例 |
| :--- | :--- | :--- |
| 中文 | PKU/MSR（`词/词性` 空格分隔） | `中国/NR 政府/NN 宣布/VV` |
| 英文 | NLTK/WSJ（`word/TAG` 空格分隔） | `The/DT cat/NN sat/VBD` |
| 英文 | Penn Treebank 括号格式 | `(NNP Pierre)(NNP Vinken)` |

**输出格式**（标准 CoNLL，每行字段：`ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC`）：
```text
1	中国	_	NR	_	_	_	_	_	_
2	政府	_	NN	_	_	_	_	_	_

1	The	_	DT	_	_	_	_	_	_
2	cat	_	NN	_	_	_	_	_	_
```

**命令行用法**：
```bash
# 转换中文语料
python data_converter.py --input data/raw/chinese.txt --output data/processed/chinese.conll --lang zh

# 转换英文语料
python data_converter.py --input data/raw/english.txt --output data/processed/english.conll --lang en
```

#### 4.1.1 标准数据加载
* [cite_start]**成熟方案**：使用 `conllu` 库读取标准语料 [cite: 1]。
* **任务**：
    1.  加载转换后的本地 `.conll` 文件。
    2.  [cite_start]实现 8:1:1 的本地数据集切分 [cite: 1]。
    3.  构建词典 (Vocabulary) 和标签集 (Tagset)。

### 4.2 特征工程 (Feature Engineering)
* **成熟方案**：定义一个 Python 类 `FeatureExtractor`，利用 Python 字典实现高维稀疏特征映射。
* **特征模板**：
    * [cite_start]**基础特征**：$w_{i-1}, w_i, w_{i+1}$ (上下文词) [cite: 1]。
    * [cite_start]**形态特征**：针对英文，提取前缀、后缀、数字、大小写 [cite: 1]。
    * [cite_start]**转移特征**：$y_{i-1} \rightarrow y_i$ (标签转移) [cite: 1]。

### 4.3 模型训练与推理 (Model Layer)
* **感知机核心**：
    * [cite_start]实现 **Averaged Weights** (平均权重) 以防止过拟合并提高稳定性 [cite: 1]。
    * [cite_start]集成 **Viterbi 算法** 实现全局最优路径搜索 [cite: 1]。
* **持久化**：训练完成后，将模型参数保存为本地二进制文件。

### 4.4 Web 交互接口 (API Layer)
* [cite_start]**Endpoint 1: `/train`**：触发本地训练逻辑，返回进度和初步 Accuracy [cite: 1]。
* [cite_start]**Endpoint 2: `/predict`**：接收用户输入字符串，返回词性标注序列 [cite: 5]。
* [cite_start]**Endpoint 3: `/stats`**：返回模型在测试集上的详细评估数据（混淆矩阵、词性占比） [cite: 5]。

---

## 5. 给 Cursor 的 Prompt 指令建议

在使用 Cursor 开发时，请直接发送类似以下的指令：

> **指令 A (语料转换)**:
> "实现一个 `data_converter.py` 脚本，支持将中文 PKU/MSR 格式（`词/词性` 空格分隔）和英文 NLTK/WSJ 格式（`word/TAG` 空格分隔）的语料文件，统一转换为标准 CoNLL 格式（每行一个词，制表符分隔 ID/FORM/LEMMA/UPOS 等字段，句间空行分隔）。脚本需支持命令行参数 `--input`、`--output`、`--lang`，同时在 `utils.py` 中暴露 `convert_to_conll()` 函数。"
>
> **指令 B (初始化后端)**:
> "使用 FastAPI 创建一个后端项目，参考我的开题报告，实现一个 `StructuredPerceptron` 类。要求使用 `conllu` 库来加载本地数据。模型需要支持平均权重更新（Averaged Weights）和 Viterbi 解码。请确保所有逻辑都在本地运行，不调用外部 API。"
>
> **指令 C (构建 UI)**:
> "使用 Vue 3 和 Element Plus 创建一个前端页面。需要一个文本输入框（用于输入待标注句子）、一个动态表格（显示词性结果，不同词性使用不同颜色的 Tag 标记）以及一个 ECharts 饼图（展示词性分布）。数据通过 Axios 从本地 FastAPI 获取。"

---

## 6. 关键技术指标要求 [cite: 5]
1.  **词性标注准确率**：在不同语料库测试中需达到良好效果。
2.  **可视化展示**：Web 页面需具备交互性，支持直观展示标注结果。
3.  **代码规范**：模块化设计，符合吉林大学本科毕业论文撰写要求。


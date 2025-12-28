## 使用说明 & 项目简介

LexRich 是一个基于预训练词向量的语义场丰富度分析器：给定英文文本，按照配置好的语义场与分组策略，输出覆盖度、类型数、簇熵、支配度等可插拔指标，支持阈值扫网格调参和 debug 导出。

快速运行（生成 Markdown 报告）：

- `python3.14 -m lexrich.cli texts/psy.txt --config config.yaml --markdown > outputs/psy.md`

核心配置入口：`config.yaml`（或同结构的 JSON），聚类方式、语义场归类方式、指标集合及参数都可在此切换。

### 示例：Tess 文本的测量结果

最新一次在文本 `texts/tess.txt` 上的运行输出见 [outputs/tess.md](outputs/tess.md)。主要指标概览：

- emotion：覆盖度约 43/10k，type 42，簇熵 ~3.81，支配度 0.27，说明情绪词分布较分散。
- character_evaluation：覆盖度约 8.39/10k，type 13，熵 2.58，支配度 0.47，评价词集中度中等。
- social_class：覆盖度约 11.36/10k，type 16，熵 3.17，支配度 0.33，阶层相关词有一定多样性。
- environment：覆盖度约 5058/10k，type 534，但支配度高达 0.97（熵 0.31），表明环境相关词大量集中在少数簇，需关注是否因词表扩展或文本主题导致的偏置。

### 方法核心：seeds 决定统计效果

本方法的灵敏度和可解释性高度取决于 seed 词表的设计：

- 覆盖面：seed 过窄会导致 FieldCoverage 偏低、Dominance 偏高；seed 过宽或语义漂移会让 ClusterEntropy 偏低且解释力下降。
- 粒度与平衡：field 内子语义若覆盖不均，簇分布会失衡；跨 field 的模糊种子会造成“抢词”，使 assign_mode=cluster 时归类不稳定。

建议：

- 先人工审阅与精简 [lexrich/resources/seed_fields.yaml](lexrich/resources/seed_fields.yaml)，每个 field 确保 3–5 个高置信度、互补的种子。
- 开启 debug 检查扩展 lexicon 与高频簇，观察语义漂移或场域串扰，再迭代 seeds。

AI Prompt for project design:

下面是一份“实现向导级”的设计文档（偏工程规格书），我需要你照着搭骨架逐步填充实现。为了满足我说的“三个决策都实现并放进 config 可调”，你需要把**聚类方式 / 语义场归类方式 / 指标集合**都做成可切换策略，并且所有关键参数都集中在 `config.py`（或 YAML/JSON 配置文件）。

---

# LexRich：基于 Word Embedding 的语义场“丰富度”分析器设计文档

## 0. 目标与非目标

### 0.1 目标

实现一个可复用工具：

* 输入任意英文文本 `text`
* 输出一组“丰富度”指标（可多语义场）

  * 语义场例：emotion / evaluation / social_class / environment …
* 丰富度计算核心流程：

  1. 词 → embedding 向量
  2. 使用阈值 τ 划分“相似词集合”（簇/邻域）
  3. 在语义场内统计丰富度（多指标、可插拔）

### 0.2 非目标（暂不做 / 可扩展）

* 不实现 BERT 等 contextual embedding 的完整版本（但预留接口）
* 不做复杂统计推导（论文也不需要）

---

## 1. 总体项目结构

```
lexrich/
  __init__.py

  config.py
  cli.py

  preprocess.py
  extractor.py

  embedding.py
  neighbors.py
  grouping.py

  fields.py

  metrics/
    __init__.py
    base.py
    cluster_metrics.py
    field_metrics.py

  analyzer.py
  report.py

  resources/
    seed_fields.yaml
    stopwords.txt

tests/
  test_preprocess.py
  test_grouping.py
  test_metrics.py
```

---

## 2. 数据流与管线（Pipeline）

### 2.1 核心数据流（Analyzer 内部）

1. `TextPreprocessor`：text → tokens（lemma/pos/offset）
2. `VocabularyExtractor`：tokens → `CorpusStats`（词频、总词数等）
3. `EmbeddingModel`：lemma → vector（查表 / OOV 策略）
4. `NeighborIndex`：建立近邻索引
5. `SimilarityGrouper`：按 config 策略 → clusters 或 neighborhoods
6. `FieldLexiconBuilder / FieldAssigner`：构建语义场词表并归类（word-level 或 cluster-level）
7. `MetricRunner`：对每个语义场计算配置中启用的 metrics
8. `ReportFormatter`：输出 JSON / Markdown 表格 / 文本报告

### 2.2 输出结果结构（建议）

* `overall`：全局指标（可选）
* `fields`：按语义场分组的指标
* `debug`：可选（簇规模、top词、扩展词表等）

---

## 3. 配置系统设计（Config-first）

你希望三类决策可自由调整：

* 聚类方式（cluster vs neighborhood）
* 语义场归类方式（word-level vs cluster-level）
* 指标集合（metrics 可插拔）

因此建议采用：

* `AnalysisConfig`（dataclass）
* 支持从 YAML/JSON 加载（可选）
* 所有策略都用字符串枚举 + 参数块

### 3.1 `config.py`：配置对象规范

```python
@dataclass
class PreprocessConfig:
    language: str = "en"
    lowercase: bool = True
    lemmatize: bool = True
    pos_tag: bool = True
    keep_pos: tuple[str, ...] = ("NOUN", "VERB", "ADJ", "ADV")
    remove_stopwords: bool = True
    min_token_len: int = 2

@dataclass
class EmbeddingConfig:
    model_type: str = "glove"  # "glove" | "word2vec" | "fasttext"
    model_path: str = "path/to/vectors"
    vector_dim: int = 300
    oov_strategy: str = "skip"  # "skip" | "lowercase_fallback" | "subword" | "zero"
    normalize_vectors: bool = True

@dataclass
class NeighborConfig:
    index_type: str = "exact"  # "exact" | "ann"
    top_k: int = 30            # 用于 neighborhood 或构图候选边
    similarity: str = "cosine" # 目前只做 cosine 即可

@dataclass
class GroupingConfig:
    mode: str = "threshold_graph"  # "threshold_graph" | "topk_neighborhood"
    threshold: float = 0.55        # τ：相似度阈值
    max_vocab: int = 20000         # 限制参与 embedding 的词表规模（按频率截断）
    min_freq: int = 2              # 低频词可不参与聚类
    graph_build: str = "topk_then_threshold"
    # graph_build:
    #   "all_pairs_threshold"（小语料才可）
    #   "topk_then_threshold"（推荐：先取 top_k 再筛阈值）

@dataclass
class FieldConfig:
    seed_path: str = "lexrich/resources/seed_fields.yaml"
    expand_fields: bool = True
    expand_top_k: int = 50
    expand_threshold: float = 0.55
    assign_mode: str = "word"    # "word" | "cluster"
    # word: 词级别分到 field
    # cluster: 簇级别分到 field（再把簇内词视为该 field）

@dataclass
class MetricsConfig:
    enabled: list[str] = field(default_factory=lambda: [
        "FieldCoveragePer10k",
        "FieldTypeCount",
        "ClusterEntropy",
        "Dominance",
    ])
    params: dict[str, dict] = field(default_factory=dict)
    # params 示例：
    # {"Dominance": {"top_n": 1}}

@dataclass
class AnalysisConfig:
    preprocess: PreprocessConfig
    embedding: EmbeddingConfig
    neighbor: NeighborConfig
    grouping: GroupingConfig
    fields: FieldConfig
    metrics: MetricsConfig
    debug: bool = False
    threshold_grid: list[float] | None = None  # 可选：做阈值敏感性分析
```

> 你可以做到：只改 `mode / assign_mode / enabled` 就切换整个策略组合，代码不用动。

---

## 4. 核心模块规格与接口

下面按文件说明 “需要实现哪些类 / 关键方法 / 输入输出”。

---

## 4.1 `preprocess.py`

### 4.1.1 数据结构

```python
@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str | None
    start: int | None
    end: int | None
```

### 4.1.2 类：`TextPreprocessor`

**职责**：清洗、分词、（可选）词形还原、词性标注、过滤

**接口**

```python
class TextPreprocessor:
    def __init__(self, cfg: PreprocessConfig): ...
    def process(self, text: str) -> list[Token]:
        """text -> filtered tokens"""
```

**实现要点**

* `lowercase`、`remove_stopwords`
* `keep_pos`：只保留内容词（论文友好）
* 提供稳定输出（同输入同输出）

---

## 4.2 `extractor.py`

### 4.2.1 数据结构：`CorpusStats`

```python
@dataclass
class CorpusStats:
    counts: dict[str, int]                  # lemma -> freq
    pos_counts: dict[str, dict[str, int]]   # lemma -> pos -> freq
    total_tokens: int
    total_types: int
```

### 4.2.2 类：`VocabularyExtractor`

```python
class VocabularyExtractor:
    def __init__(self, cfg: GroupingConfig): ...
    def extract(self, tokens: list[Token]) -> CorpusStats: ...
```

**实现要点**

* 统计 lemma 频次
* 为 embedding 选词表：按频率排序，受 `min_freq / max_vocab` 控制

---

## 4.3 `embedding.py`

### 4.3.1 接口：`EmbeddingModel`

```python
class EmbeddingModel(Protocol):
    def get_vector(self, lemma: str) -> np.ndarray | None: ...
    def has_word(self, lemma: str) -> bool: ...
    @property
    def dim(self) -> int: ...
```

### 4.3.2 实现类（至少一个）

* `KeyedVectorsEmbedding`（加载预训练向量并查表）
* `OOVHandler`：按 `oov_strategy` 处理

**实现要点**

* `normalize_vectors=True` 时，向量 L2 归一化（cosine 更稳定）
* OOV 策略：

  * `skip`：直接丢弃该词（推荐）
  * `lowercase_fallback`
  * `zero`：返回 0 向量（不推荐，但可选）

---

## 4.4 `neighbors.py`

### 4.4.1 接口：`NeighborIndex`

```python
class NeighborIndex(Protocol):
    def build(self, vectors: dict[str, np.ndarray]) -> None: ...
    def query(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
```

### 4.4.2 实现

* `ExactNeighborIndex`

  * 存储矩阵 `V` 和词表列表
  * `query` 做 `V @ vec` 得到 cosine（若已归一化）

---

## 4.5 `grouping.py`（你说的“阈值划分相似词”的核心）

### 4.5.1 输出结构

```python
@dataclass
class Cluster:
    id: int
    members: set[str]

@dataclass
class GroupResult:
    clusters: list[Cluster]                  # threshold_graph 模式有意义
    neighborhoods: dict[str, set[str]]       # topk_neighborhood 模式有意义
    word_to_cluster: dict[str, int]          # cluster 模式映射
```

### 4.5.2 主类：`SimilarityGrouper`

```python
class SimilarityGrouper:
    def __init__(self, cfg: GroupingConfig, neighbor_cfg: NeighborConfig): ...

    def group(
        self,
        vocab: list[str],
        vectors: dict[str, np.ndarray],
        index: NeighborIndex
    ) -> GroupResult:
        """
        根据 cfg.mode:
          - threshold_graph: 构图 -> union-find -> clusters
          - topk_neighborhood: 每词取 top_k 且 sim>=tau -> neighborhoods
        """
```

### 4.5.3 两种 mode 行为定义

#### A) `threshold_graph`

* 构边策略 `graph_build`：

  * 推荐：`topk_then_threshold`

    * 每个词 query top_k
    * 筛选 sim >= τ 的边
    * 用 Union-Find 合并连通分量
* 输出：

  * `clusters` + `word_to_cluster`

#### B) `topk_neighborhood`

* 每个词得到邻域集合（包含自己可选）
* 输出：

  * `neighborhoods`（每词一组相似词）
* 注意：这时“簇指标”需要按邻域改写或在 metric 内兼容

> 建议实现时：metric 接收 `GroupResult`，由 metric 决定使用 clusters 还是 neighborhoods（或在 `FieldContext` 里统一暴露 `get_group_id(word)` 和 `iter_groups()`）。

---

## 4.6 `fields.py`（语义场构建与归类）

### 4.6.1 语义场定义

```python
@dataclass
class SemanticField:
    name: str
    seeds: list[str]
```

### 4.6.2 Builder：`FieldLexiconBuilder`

**职责**：从 seed 扩展 field lexicon（可关）

```python
class FieldLexiconBuilder:
    def __init__(self, cfg: FieldConfig, embed: EmbeddingModel, index: NeighborIndex): ...
    def build(self) -> dict[str, set[str]]:
        """field_name -> lexicon(set<lemma>)"""
```

扩展策略（建议）：

* 每个 seed 查 top_k
* 合并所有相似词
* 过滤 sim < expand_threshold
* 最终再 union seeds 本身

### 4.6.3 Assigner：`FieldAssigner`

**职责**：把词/簇分配到 field（由 config 决定）

```python
class FieldAssigner:
    def __init__(self, cfg: FieldConfig, field_lexicons: dict[str, set[str]],
                 embed: EmbeddingModel): ...

    def assign_words(self, vocab: list[str]) -> dict[str, str | None]:
        """word -> field_name or None"""

    def assign_clusters(self, clusters: list[Cluster]) -> dict[int, str | None]:
        """cluster_id -> field_name or None"""
```

#### `assign_mode="word"`

* 规则：word ∈ field_lexicon → 属于该 field
* 若命中多个 field：可用相似度最高/优先级（配置可加）

#### `assign_mode="cluster"`

* 规则：簇内命中某 field 词比例最高 → 分配给该 field
* 优点：更符合“相似词集合”思想，且对小说风格更稳定

> 这一步最好在 debug 输出：每个 field 的 top clusters/代表词，便于论文举例。

---

## 4.7 `metrics/`：丰富度度量系统（可插拔）

### 4.7.1 `metrics/base.py`

```python
@dataclass
class FieldContext:
    field_name: str
    field_words: set[str]          # 属于该 field 的词（最终用于统计）
    counts: dict[str, int]         # 全局词频
    total_tokens: int

    group_result: GroupResult
    word_to_cluster: dict[str, int] | None   # 若 cluster 模式
    neighborhoods: dict[str, set[str]] | None

@dataclass
class MetricResult:
    name: str
    value: float | dict
    details: dict | None = None

class RichnessMetric(ABC):
    name: str
    def __init__(self, params: dict | None = None): ...
    @abstractmethod
    def compute(self, ctx: FieldContext) -> MetricResult: ...
```

### 4.7.2 你最少需要实现的一组 Metrics（论文够用 + 可解释）

#### A) 语义场覆盖密度：`FieldCoveragePer10k`

* 定义：field token 总数 / 总 token × 10000

#### B) 语义场 type 数：`FieldTypeCount`

* 定义：field 内出现的不同 lemma 个数（可同时输出 per10k）

#### C) 簇熵：`ClusterEntropy`

* 基于 field 内 token 在“簇”上的分布
* 越均匀 → 表达更分散、更丰富

#### D) 支配度：`Dominance`

* top cluster share（最大簇占 field token 的比例）
* 越高 → 说明表达集中、丰富性差

> 注意兼容两种 grouping：

* 如果是 `threshold_graph`：簇就是 cluster
* 如果是 `topk_neighborhood`：可把“每个词的邻域代表簇”转换成 pseudo-group（建议简单起见：只在 cluster 模式下启用 ClusterEntropy/Dominance；或在 metric 内实现 neighborhood 的替代定义）

### 4.7.3 `metrics/cluster_metrics.py`（建议实现）

* `ClusterEntropyMetric`
* `DominanceMetric`
* （可选）`ClusterTypeTokenRatioMetric`：簇数量 / field token

### 4.7.4 `metrics/field_metrics.py`（建议实现）

* `FieldCoveragePer10kMetric`
* `FieldTypeCountMetric`
* （可选）`TopNLemmaShareMetric`：top n lemma 占比，衡量“避免重复”

---

## 4.8 `analyzer.py`：总控

### 4.8.1 输出结构

```python
@dataclass
class AnalysisResult:
    total_tokens: int
    total_types: int
    per_field: dict[str, list[MetricResult]]
    overall: list[MetricResult] | None = None
    debug: dict | None = None
```

### 4.8.2 `RichnessAnalyzer`

```python
class RichnessAnalyzer:
    def __init__(self, cfg: AnalysisConfig): ...

    def analyze(self, text: str) -> AnalysisResult:
        # 1 preprocess
        # 2 extract
        # 3 embedding lookup
        # 4 neighbor index build
        # 5 grouping
        # 6 fields build + assign
        # 7 metrics compute
        # 8 return
```

### 4.8.3 阈值敏感性分析（可选但很加分）

如果 `cfg.threshold_grid` 不为空：

* 对每个 τ 运行 grouping + metrics（可以复用前面 embedding/index/fields）
* 输出一个额外 `debug["threshold_sweep"]`：

  * τ → 每个 field 的指标值
    这可以直接生成论文里的“稳健性”讨论。

---

## 4.9 `report.py`：结果可直接复制进论文

`ReportFormatter` 支持：

* `to_json()`
* `to_markdown_tables()`：

  * 语义场 × 指标 的表格
  * 以及每个语义场 top words / top clusters（可选）

---

## 5. 语义场与词表资源（resources/seed_fields.yaml）

建议把 seed 做成可读、可改、可扩展的 YAML：

```yaml
emotion:
  seeds: [happy, sad, angry, fear, delighted, ashamed, anxious]
evaluation:
  seeds: [good, bad, excellent, awful, fine, admirable, disgraceful]
social_class:
  seeds: [gentleman, lady, noble, servant, poor, wealthy, fortune]
environment:
  seeds: [rain, wind, river, forest, garden, sky, sunshine, landscape]
```

实现上允许：

* `expand_fields: true/false`
* 扩展出来的词也保存到 debug，便于你写 Appendix

---

## 6. “阈值划分相似词 → 丰富度”的定义建议（你论文会用）

为了写得清楚、老师能接受，建议你在论文里把核心操作定义成：

* 使用预训练词向量表示每个 lemma
* 使用 cosine similarity 衡量语义接近度
* 当 similarity ≥ τ 时，将词视为同一“semantic neighborhood / cluster”
* 通过 cluster 分布（熵、支配度）与 field 覆盖/多样性（type count、密度）来衡量丰富性

你实现时指标可更灵活，但论文表达要“干净”。

---

## 7. 实现顺序建议（避免卡死）

按这个顺序做，基本不会返工：

1. `preprocess.py` + `extractor.py`（先能得到 counts）
2. `embedding.py`（先支持 glove 查表 + skip OOV）
3. `neighbors.py`（exact top_k 查询）
4. `grouping.py`（先实现 threshold_graph + union-find）
5. `fields.py`（先实现 seed 不扩展、word assign；再加 expand；再加 cluster assign）
6. metrics（先 2 个 field metrics，再加 2 个 cluster metrics）
7. `analyzer.py` 串起来
8. `report.py` 输出 markdown 表格
9. 再加 `topk_neighborhood` mode + threshold_grid sweep

---

## 8. 最小可用版本（MVP）验收标准

给任意 text：

* 能输出至少 4 个指标（每个 field 一组）：

  * FieldCoveragePer10k
  * FieldTypeCount
  * ClusterEntropy（cluster 模式下）
  * Dominance（cluster 模式下）
* 能通过 config 切换：

  * `grouping.mode = threshold_graph / topk_neighborhood`
  * `fields.assign_mode = word / cluster`
  * `metrics.enabled` 任意组合

---

## 9. 你可能会想加的“Debug 输出”（强烈建议）

为了你做论文举例，debug 里建议包含：

* 每个 field 的：

  * top 20 lemma（按频率）
  * top clusters（簇大小/簇代表词）
  * 扩展 lexicon（若 expand_fields）

这能直接变成 Discussion 的例子来源（老师会觉得“内容充实”）。

---

## 10. 配置示例（YAML）

```yaml
preprocess:
  lowercase: true
  lemmatize: true
  pos_tag: true
  keep_pos: ["NOUN", "VERB", "ADJ", "ADV"]
  remove_stopwords: true
embedding:
  model_type: "glove"
  model_path: "data/glove.6B.300d.txt"
  vector_dim: 300
  oov_strategy: "skip"
  normalize_vectors: true
neighbor:
  index_type: "exact"
  top_k: 30
  similarity: "cosine"
grouping:
  mode: "threshold_graph"
  threshold: 0.55
  min_freq: 2
  max_vocab: 20000
  graph_build: "topk_then_threshold"
fields:
  seed_path: "lexrich/resources/seed_fields.yaml"
  expand_fields: true
  expand_top_k: 50
  expand_threshold: 0.55
  assign_mode: "cluster"
metrics:
  enabled:
    - "FieldCoveragePer10k"
    - "FieldTypeCount"
    - "ClusterEntropy"
    - "Dominance"
  params:
    Dominance:
      top_n: 1
debug: true
threshold_grid: [0.45, 0.55, 0.65]
```

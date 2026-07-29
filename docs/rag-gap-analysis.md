# RAG 改进措施差距分析

> 基于 8 篇行业文章的观点，对照 SparkSage 当前实现，识别**尚未考虑但值得借鉴**的改进措施。
>
> 文章来源（均为微信公众号）：
> 1. 《2026年构建RAG系统的核心策略：从60%到94%准确率》— 11种策略组合
> 2. 《RAG准确率翻倍，我做了这些优化》— 文档解析→切分→向量化→检索→重排→上下文全链路
> 3. 《高精度RAG的底层架构，准确率飙升29.6%》— 增强型实体页面 / 结构化数据物化
> 4. 《HyGRAG：打通知识图谱与文本检索》— 上下文+关系感知的层次化图检索
> 5. 《5个技巧让RAG准确率飙升》— 分片优化→查询处理→检索→重排
> 6. 《医疗AI答题准确率破60%，40组实验》— 检索Pipeline组件贡献拆解
> 7. 《RAG命中率从60%到78%》— 结构感知分块 / Small-to-Big / Context Cliff
> 8. 《让RAG准确率飙升至90%以上》— OCR/表格识别 / 混合检索+RRF / 多向量

---

## 一、SparkSage 已覆盖的能力（不需重复借鉴）

在分析差距前，先明确文章中大量提到但 **SparkSage 已通过 IdeaBlock 架构超越式覆盖** 的点，避免重复劳动：

| 文章提到的策略 | SparkSage 对应实现 | 备注 |
|---|---|---|
| 上下文感知分块（策略1/5/7） | **IdeaBlock 本身** — 问题对齐的结构化知识单元 | 文章讨论的递归切分/锚点切分/语义切分都是针对朴素文本切片的优化，IdeaBlock 从根本上绕过了"切断语义"问题 |
| 上下文检索 / Contextual Retrieval（策略2，Anthropic） | `embedding_text` = name + critical_question + trusted_answer | 每个块的嵌入文本天然自带"问题上下文"，不需额外 LLM 生成前缀 |
| 重排序（策略3/2/5/6/7） | `CrossEncoderReranker`（`[rerank]`）、`LLMReranker`、`IdentityReranker` | sigmoid 归一化的 cross-encoder 已是业界最优方案 |
| 查询扩展 / 多查询 RAG（策略4/5） | `LLMQueryExpander` + `QAEngine._multi_retrieve`（RRF 融合） | 多路召回→RRF→rerank 已闭环 |
| 查询改写 / 指代消解（2/5/8） | `LLMQueryRewriter` + `ConversationContext` | 多轮对话上下文已一等公民 |
| 自查询 / 元数据过滤（2/5） | `LLMSelfQueryParser` → `RetrievalFilter`（tags/entities/language/kb_id） | 标签值从 `Tag` enum 实时读取 |
| 混合检索 BM25+dense（2/6/8） | `BM25Retriever`（权重 `keywords` 字段）+ dense kNN + `reciprocal_rank_fusion` | 纯 stdlib，CJK unigram+bigram |
| Context Cliff / token 预算（7） | `trim_to_token_budget`（`reader/budget.py`），Reader `max_context_tokens` | 生成前和判分前都裁剪 |
| 拒答门 / "不知道就说不知道"（1/2/5） | `Reader` abstention gate（`min_faithfulness`）+ `QueryProcessor`（`min_confidence`） | 查询侧+答案侧双重门控 |
| 忠实度评判 | `LLMFaithfulnessJudge` | LLM-as-judge |
| 引用溯源（2/3/8） | `GeneratedAnswer.citations` 绑定 block id + `source.uri`/`source.locator` | 幻觉 id 会被丢弃 |
| 语义缓存 | `InMemorySemanticCache`（embedding-keyed，实现 `QACache`） | 近似查询短路整条管线 |
| 去重 | Distill 管线（迭代阈值精炼 + 层次化 LLM 合并） | 含 LSH 加速、异步 Job |
| 基准评估 / 评估集（2/6/7） | `BenchmarkRunner`（IdeaBlock vs RecursiveCharSplitter）、`QAEvaluator` | hit@k / MRR / token 效率 |
| 自动标签 | `tags/`：RAKE / TF-IDF / TextRank | CJK bigram tokenizer |
| 文档转换 | `MarkdownConverter`（markitdown 后端） | 多格式→Markdown |
| 文本清洗 | `TextCleaner` + 可组合 `CleaningRule` + 源感知路由 | |

**结论**：SparkSage 的 IdeaBlock 设计已在"分块策略"这一被文章反复强调的核心环节上实现了降维打击。以下差距分析聚焦于**检索后/答案前**和**知识组织结构**层面的空白。

---

## 二、值得借鉴的改进措施（按优先级排序）

### P0 — 高价值，架构契合度高，实现成本低

#### 1. 自反思检索循环（Self-Reflective / Iterative Retrieval）

**来源**：文章1（策略7）、文章2（7.1）、文章5

**现状差距**：
`QAEngine.ask()` 是**单次直通**管线：query → retrieve → answer。Reader 的 abstention 门控只在**答案侧**生效——当 faithfulness 低于阈值时直接拒答，**不会触发重新检索**。如果检索结果本身质量差（召回的 block 与查询不相关），系统没有自我修正机制，只会直接拒答或生成低质量答案。

**文章观点**：
- 文章1策略7：检索后评分相关性（1-5分），若评分低则用 LLM 优化查询并重新检索，最多迭代 2 次。
- 文章6：查询改写 + 重排贡献了 90% 的准确率增益，是医疗 RAG 的"必装组件"。

**建议**：
新增 `RetrievalGrader` 协议（对称于 `FaithfulnessJudge`），在 `QAEngine` 中增加可选的迭代循环：

```
query → retrieve → grade(relevance) → 若低于阈值 → refine query → re-retrieve
```

- 复用现有 `LLMClient`（lenient→strict 模式），评分后触发 `LLMQueryRewriter` 做查询精炼。
- 最大迭代次数（默认 2）作为配置项，避免延迟爆炸。
- 与 Reader abstention 形成对称：查询侧 → 检索侧 → 答案侧，三道门控。

**契合度**：极高。SparkSage 已有 `QueryProcessor`（查询门控）和 `Reader`（答案门控），缺的正是中间的检索门控——这是三段式门控的最后一块拼图。

---

#### 2. HyDE（假设性文档嵌入检索）

**来源**：文章2（7.3）

**现状差距**：
`query/expander.py` 的 docstring 提到 "HyDE-style variants"，但 `LLMQueryExpander` 实际实现的是**多查询改写**（生成 n 个同义改写），并非真正的 HyDE。真正的 HyDE 是：让 LLM 先根据查询生成一段**假设性答案**，再用这段假设答案去向量库检索。

**为什么对 IdeaBlock 特别有价值**：
IdeaBlock 的 `embedding_text` = name + critical_question + **trusted_answer**。标准查询是"问题→问题"的匹配，而 HyDE 是"问题→假设答案→真实答案"的匹配，恰好对齐 IdeaBlock 的问答结构。对于简短/模糊查询，假设答案比原始查询更能匹配 `trusted_answer` 的语义空间。

**建议**：
在 `QueryExpander` 协议下新增 `HyDEExpander` 实现：
- 用 LLM 生成一段假设答案（temperature=0.7 鼓励多样性）。
- 将假设答案作为额外的"变体"参与 RRF 多路召回。
- 仅在查询长度 < 阈值时启用（文章2建议 <5 词），避免长查询引入幻觉。

**契合度**：高。完美契合 `QueryExpander` 协议和 `_multi_retrieve` 的 RRF 融合路径，零侵入。

---

#### 3. 意图→知识库路由（Intent-Based Index Routing）

**来源**：文章2（7.5）、文章5（3.1）

**现状差距**：
SparkSage **同时拥有** `IntentClassifier`（意图分类）和 `KnowledgeBase`（多租户，`kb_id` 作用域），但两者**没有连接**。当前意图分类只用于查询拦截（out-of-domain 判断），不会据此路由到不同的知识库/索引。`RetrievalFilter.kb_id` 需要调用方手动指定。

**文章观点**：
- 文章2：按领域拆分索引（技术文档/客服FAQ/产品手册分开），查询时先用轻量分类器识别意图，只在对应索引检索，避免互相干扰。
- 文章6：领域专用模型碾压通用模型，说明领域隔离的价值。

**建议**：
在 `QAEngine` 中增加可选的 `intent_router: Callable[[IntentResult], str | None]`，将意图分类结果映射到 `kb_id`，自动设置 `RetrievalFilter.kb_id`。这是连接两个已有组件的"胶水代码"，实现成本极低。

**契合度**：极高。两个组件都已存在，只需一层路由映射。

---

#### 4. IdeaBlock 语义完整性校验门（Block Quality Gate）

**来源**：文章5（1.1.4 分片质量校验）、文章7

**现状差距**：
SparkSage 有严格的 Pydantic 校验（`extra="forbid"`、`min_length`、`max_length`、`QUESTION_MAX`、`RECOMMENDED_ANSWER_MAX`）和 `TextCleaner`，但这些都是**语法/格式层**的校验。生成的 IdeaBlock 可能语法合法但**语义不完整**——例如 `trusted_answer` 没有真正回答 `critical_question`，或 answer 与 question 自相矛盾。

**文章观点**：
- 文章5：规则校验（过滤空白/乱码/重复）+ LLM 语义校验（判断分片是否语义完整），双重过滤后有效分片率从 70% 提升到 92%。

**建议**：
在 `IdeaBlockGenerator` 的生成管线末尾增加可选的 `BlockValidator` 协议：
- 规则层：`trusted_answer` 与 `critical_question` 的关键词重叠率（已有 `keywords` 字段可辅助）。
- LLM 层：让 LLM 判断"答案是否回答了问题"（复用 `LLMClient`，lenient→strict）。
- 不达标的 block 返回重生成或标记为 `DRAFT`，不进入索引。

**契合度**：高。遵循 SparkSage 的 lenient→strict 模式和 strict validation 哲学。

---

### P1 — 高价值，实现成本中等

#### 5. 父子文档 / Small-to-Big 检索 + 窗口检索

**来源**：文章1（策略9 分层RAG）、文章2（7.4 窗口检索）、文章7（Small-to-Big）

**现状差距**：
IdeaBlock 是**扁平结构**——同源文档的 block 之间没有"父子"或"兄弟"关系。`parents` 字段仅用于 Distill 合并谱系，不是层级上下文。`source.locator` 记录了来源位置但无法据此检索"相邻 block"。当一个 IdeaBlock 被命中时，无法自动扩展到同一文档/章节的上下文 block。

**文章观点**：
- 文章7：Small-to-Big 是生产环境实测最优策略之一（recall 0.86）——小块精准召回，生成时扩展到大块/父块提供完整上下文。
- 文章2：窗口检索——命中一个 chunk 时，带上前后 N 个 chunk，适合长文档连贯场景。

**建议**：
利用现有 `source.uri` + `source.locator` 建立 block 邻接关系：
- 在 `KnowledgeBase` 中维护 `source → [block_ids]` 的有序映射（按 `source.locator` 排序）。
- 在 `Retriever` 中增加可选的 `context_window` 参数：命中 block 后自动带上同源的相邻 N 个 block（类似文章2的 `retrieve_with_window`）。
- 或实现 Small-to-Big：IdeaBlock 作为"子块"精准召回，`DocumentRecord.body_markdown` 的相关段落作为"父块"提供完整上下文（`documents/` 层已存在）。

**契合度**：中高。`source.locator` 和 `DocumentRecord` 已存在，需要增加邻接索引和扩展逻辑。

---

#### 6. 多向量 / 多视角嵌入（Multi-Vector Embedding）

**来源**：文章8（多向量存储）、文章1（策略10 延迟分块）

**现状差距**：
当前每个 IdeaBlock 只有一个向量（`embedding` 字段），由 `embedding_text`（name + question + answer 三字段拼接）生成。这意味着"问题匹配"和"答案匹配"共用同一个向量空间，可能相互稀释。

**文章观点**：
- 文章8：一个文本块对应多个向量（段落分割/滑动窗口），只要其中一个被检索到就召回该数据，显著提升召回率。
- 文章6实验：嵌入模型对效果影响虽小但稳定。

**建议**：
为 IdeaBlock 增加可选的多向量模式（不破坏现有单向量兼容）：
- **问题向量**：仅嵌入 `critical_question`（query-side 匹配，用户提问→block 问题）。
- **答案向量**：仅嵌入 `trusted_answer`（content-side 匹配，查询→答案内容）。
- 检索时分别检索两路，RRF 融合。
- 实现上可扩展 `BlockEmbedder` 支持 `field_selector` 参数，或新增 `MultiVectorStore`。

**契合度**：中。需要扩展嵌入和存储层，但 IdeaBlock 的结构化字段天然支持多视角嵌入——这是相比朴素文本切片的独特优势。

---

#### 7. 知识图谱 / 实体关系检索层

**来源**：文章3（增强型实体页面）、文章4（HyGRAG）、文章1（策略8 知识图谱）

**现状差距**：
SparkSage 已在每个 IdeaBlock 上提取 `entities: list[Entity]`（含 `entity_name`、`entity_type`、`aliases`），但这些实体**仅用于 `RetrievalFilter` 的事后过滤**，没有任何图结构、关系三元组、或多跳遍历能力。`embed/similarity.py` 的 `find_similar_pairs` 检测的是**向量相似**的 block 对，不是实体关系。

**文章观点**：
- 文章4（HyGRAG，WWW 2026）：核心洞察——实体和文本块的简单组合 ≠ 知识融合。需要层次化聚类 + LLM 摘要生成"涌现性理解"，多跳推理准确率提升 9.7%。
- 文章3：将知识图谱中的关联数据物化为自然语言（"链接物化"），LLM 单次检索即可获得多跳信息，准确率提升 29.6%。
- 文章4消融实验：去除实体+关系后，MultiHop-RAG 准确率从 65% 降到 52%。

**建议**（渐进式）：
1. **轻量版**：在 `KnowledgeBase` 中构建 `entity → [block_ids]` 的倒排索引。检索时若命中某 block，自动召回共享相同实体的其他 block（"实体共现扩展"）。这是无图的伪多跳。
2. **中量版**：在生成阶段提取关系三元组（block1.entity —关系— block2.entity），构建实体关系图，支持 1-2 跳遍历。
3. **重量版**：参照 HyGRAG，对共享实体的 block 做聚类 + LLM 摘要，生成社区级"涌现知识"节点。

**契合度**：中。Entity schema 已存在，是自然的扩展点。但图遍历引入了新的存储和查询范式，与当前"文本无关的向量存储"理念有张力——建议作为可选层。

---

### P2 — 价值明确但范围较大 / 优先级较低

#### 8. Agentic RAG（自主工具选择检索）

**来源**：文章1（策略6）、文章3

**现状差距**：
`QAEngine` 是固定管线，检索策略（dense+lexical+rerank）不可在运行时由 agent 自主选择。文章3的 agent 可在 `search_documents`、`follow_entity_link`、`search_knowledge_graph` 之间自主选择。

**建议**：暂不实现完整 agent 循环，但可考虑在 `_multi_retrieve` 中增加条件路由（如检测到查询含编号/精确匹配模式时，优先走 BM25；检测到概念性查询时优先走 dense）。这是"规则化 agent"的低成本近似。

**契合度**：低-中。与 SparkSage 的"协议驱动、无框架依赖"理念有一定张力（引入 agent 框架会破坏零依赖核心）。

---

#### 9. 嵌入模型微调工作流

**来源**：文章1（策略11）、文章2（4.2）

**现状差距**：
`EmbeddingClient` 是可插拔协议，但 SparkSage 不提供任何微调工具/工作流。

**建议**：可提供 `embed/finetune.py` 工具，从 IdeaBlock 语料自动构造正负样本对（同源 block 的 critical_question↔trusted_answer 为正样本，跨源为负样本），输出 sentence-transformers 兼容的训练数据。

**契合度**：中。属于工具链而非核心，可作为独立模块。

---

#### 10. 延迟分块（Late Chunking）

**来源**：文章1（策略10）

**现状差距**：当前嵌入架构是文本级（`EmbeddingClient.embed_batch` 接收文本列表），不支持 token 级嵌入后再池化。

**建议**：暂不实现。延迟分块需要长上下文嵌入模型和完全不同的嵌入架构，与 IdeaBlock 的"结构化字段嵌入"理念有冲突。IdeaBlock 的 `embedding_text` 已通过三字段拼接保留了上下文。

---

## 三、总结：优先级矩阵

| 措施 | 价值 | 成本 | 架构契合度 | 推荐优先级 |
|---|---|---|---|---|
| **1. 自反思检索循环** | ★★★★★ | 中 | ★★★★★ | **P0** |
| **2. HyDE 扩展器** | ★★★★ | 低 | ★★★★★ | **P0** |
| **3. 意图→KB 路由** | ★★★★ | 极低 | ★★★★★ | **P0** |
| **4. Block 语义校验门** | ★★★★ | 中 | ★★★★ | **P0** |
| **5. 父子/窗口检索** | ★★★★ | 中 | ★★★★ | **P1** |
| **6. 多向量嵌入** | ★★★ | 中高 | ★★★ | **P1** |
| **7. 知识图谱层** | ★★★★★ | 高 | ★★★ | **P1**（渐进式） |
| 8. Agentic RAG | ★★★ | 高 | ★★ | P2 |
| 9. 嵌入微调工作流 | ★★★ | 中 | ★★★ | P2 |
| 10. 延迟分块 | ★★ | 高 | ★ | P2（暂缓） |

## 四、建议实施路线

**第一批（1-2周）**——低 hanging fruit，立竿见影：
- 措施 3（意图→KB 路由）：连接两个已有组件，几十行胶水代码。
- 措施 2（HyDE）：在 `QueryExpander` 协议下新增一个实现类。
- 措施 1（自反思检索）：`RetrievalGrader` 协议 + QAEngine 迭代循环。

**第二批（3-4周）**——质量门控 + 上下文扩展：
- 措施 4（Block 语义校验门）：生成管线末端增加验证层。
- 措施 5（父子/窗口检索）：基于 `source.locator` 建立邻接索引。

**第三批（中期）**——知识组织升级：
- 措施 7（知识图谱层）：从轻量版"实体共现扩展"起步，验证效果后再决定是否深入。
- 措施 6（多向量嵌入）：评估问题向量/答案向量分路检索的 recall 增益。

## 五、关键洞察

文章们反复验证的一个核心结论与 SparkSage 的设计哲学高度一致：**分块策略决定了检索质量的上限**（文章7："分块定义了检索对象是什么，embedding 只决定怎么找到它"）。SparkSage 的 IdeaBlock 已在这个最关键的环节上实现了结构性优势。

因此，差距不在"分块"本身，而在：
1. **检索后的自我修正能力**（措施1）——当前是单次直通，缺少迭代优化。
2. **块间关系的利用**（措施5/7）——当前 block 是扁平孤岛，缺少父子/邻接/实体关系连接。
3. **已有组件的未连接**（措施3）——IntentClassifier 和 KnowledgeBase 各自存在但未打通。

这三个方向是投入产出比最高的改进点。

# Agentic RAG 多轮问答优化分析报告

> 针对 issue「继续优化 agentic rag 多轮问答」的深度分析。
> 方法：对照 SparkSage 当前 `agent/` + `qa/` + `query/` + `reader/` 实现，逐项盘点多轮对话与 Agentic RAG 的现状与缺口，并给出可落地、符合本仓库既定架构（Protocol 驱动、lenient→strict、identity 退化、零外部 SDK 进 core）的借鉴方案。

---

## 0. 方法论与可信度说明

issue 给出的 6 篇知乎参考链接，经多渠道尝试（直连 / Wayback Machine / 换 UA）均被知乎反爬 JS 挑战拦截（403 / `zse-ck` 脚本页），且无 Wayback 快照，**无法逐字核对原文**。

因此本报告采用「**以代码现状为锚、以 Agentic RAG 业界共识技术为参照**」的策略：
- 每一项「可借鉴点」都先落到 SparkSage **真实存在的缺口**（附 `file:line`），再对照该技术被广泛讨论的形态给出落地建议；
- 凡依赖原文具体措辞的结论，均在文末「未验证假设」中标注；
- 所有建议严格遵循 `AGENTS.md` 的架构铁律（core 不引第三方 SDK、新增能力即 Protocol + identity 退化 + lenient→strict）。

这样即便无法读到原文，报告中的每一条都能独立成立——因为它锚定的是**项目里真实可证伪的代码缺口**。

---

## 1. 当前实现盘点（已具备的能力）

| 能力 | 位置 | 说明 |
|---|---|---|
| 多轮上下文载体 | `query/context.py:40` `ConversationContext` | 不可变值对象，`with_turn` / `from_pairs` / `as_text(max_turns)` |
| 查询侧指代消解 | `query/processor.py:169` | `rewriter.rewrite(query, context=, intent=)` 接收上下文 |
| 意图分类带上下文 | `query/classifier.py`（经 processor 调用） | `classifier.classify(query, context)` |
| Agent 控制器看上下文 | `agent/prompts.py:103` | `state.context.as_text(max_turns=4)` 注入 controller prompt |
| 会话历史持久化 | `qa/history.py` `QATurn` + `QASessionStore` | 每轮 Q&A 全量序列化进 SQLite/内存，`GET /api/v1/query/history` |
| Agentic 循环 | `agent/engine.py:129` `AgenticQAEngine` | seed 检索 + 有界 ReAct（thought→retrieve/synthesize）+ 回调进度 + 协作取消 |
| 检索侧自反思 | `qa/engine.py:556` `_reflective_retrieve` | grader 打分→低于阈值→refiner 改写→重检索，保留最优 |
| 答案侧守门 | `reader/orchestrator.py:205` `Reader.answer` | 生成→faithfulness 判定→置信/忠实度阈值 abstain |
| 三段对称守门 | `qa/engine.py` | query `min_confidence` → retrieval `min_relevance` → answer `min_faithfulness` |
| Context-Cliff 预算 | `reader/budget.py` `trim_to_token_budget` | 生成前裁剪 + head/tail 重排（lost-in-the-middle 缓解） |

**结论**：SparkSage 的「单轮端到端 QA」+「查询侧多轮指代消解」已经完整；缺口集中在**答案侧与记忆层对多轮的支持**、**Agent 的规划/工具/反思深度**。

---

## 2. 关键缺口识别（核心发现）

以下五条是在源码中**可直接验证**的结构性缺口，按对「多轮 Agentic RAG」影响从大到小排序。

### 缺口 A（致命）：答案生成全程「上下文失明」

`AnswerGenerator` 协议与实现都不接收上下文：
- 协议 `reader/generator.py:61` `generate(self, query, chunks)` —— 无 `context`
- 实现 `reader/generator.py:117` 同样无 `context`
- 编排 `reader/orchestrator.py:205` `Reader.answer(query, chunks)` —— 无 `context`
- `qa/engine.py:456` `self._reader.answer(query, retrieval.chunks)` —— 未透传 context
- `agent/engine.py:340` `self._reader.answer(query, list(state.evidence))` —— 未透传 context

**后果**：即便 `QueryRewriter` 把「那联通呢」改写成「中国移动 2024 净利润之后，中国联通呢」，**最终答案依然在没有前序问答的情况下生成**。这会让多轮里的承接、对比、修正类回答质量明显塌方——这是「多轮问答」体验里最致命的一环。检索用到了上下文，生成却没有，是个不对称的断点。

### 缺口 B：Faithfulness 判定与 Grader 不带上下文

- `reader/faithfulness.py` 的 `judge(query, answer_text, chunks)` 无 context
- `retrieve/grader.py` 的 `grade(query, chunks)` 无 context

**后果**：判官无法理解「同一个吗」「第二个呢」这类上下文依赖问题，打分偏离用户真实意图，进而误触发 abstain 或误放行。

### 缺口 C：会话历史不自动回灌，无记忆压缩

- `api/app.py:1056` `flt, context = _build_filter_from_request(body)` —— context 来自**请求体**，前端不发就没有
- 后端**不会**从 `QASessionStore` 自动重建 `ConversationContext`
- `ConversationContext.as_text(max_turns=4)` 仅取最近 4 轮原文，**无滚动摘要**

**后果**：
1. 前端实现稍有不慎（刷新后只发当前问题），多轮能力即丢失——后端没有兜底。
2. 长会话超过 4 轮，早期关键事实（实体、约束、用户偏好）直接被丢弃，没有「记忆」概念。

### 缺口 D：Agent 动作空间二元化，缺规划与工具

- `agent/models.py:34` `ActionType` 仅 `RETRIEVE` / `SYNTHESIZE`
- `agent/engine.py` 每步一次检索、串行、贪心（一次决定一个子查询）
- 无「先出完整计划再执行」、无并行检索、无计算器/SQL/时间范围/标签过滤等结构化工具

**后果**：对「分别对比 A、B、C 三家公司在 2022-2024 的营收增速」这类需要**结构化分解 + 并行 + 数值计算**的问题，当前 Agent 只能串行盲搜，且无法做算术。

### 缺口 E：Agent 无答案级自反思闭环

- Agent 循环只判断「证据是否足够」（controller），**不判断「答案是否够好」**
- `QAEngine` 的 retrieval grader/refiner 自反思**没有**移植到 `AgenticQAEngine`
- 答案生成后即终止，无「答案质量差→补充检索/改写→重答」的闭环

**后果**：Agent 攒了一堆证据但答得不好时，没有自我纠偏机制；`aborted=True` 时直接出 best-effort 答案。

### 次要缺口（影响中等）

- **F 语义缓存对上下文不敏感**：`qa/engine.py` `QACache.lookup(query)` 仅按 query 字符串命中，同一短语在不同会话语境下会被误命中（Agent 路径目前禁缓存，但 default 路径有此风险）。
- **G 无流式输出**：`on_progress` 已存在但未接到流式 HTTP 响应，多轮对话 UX 下用户需干等。
- **H 上下文渲染粗糙**：`as_text` 只输出 `role: content` 平铺，不区分「上一轮的检索事实」与「用户的元偏好」，token 利用率低。

---

## 3. 可借鉴技术点 → SparkSage 落地方案

每条给出：**技术点 → 借鉴理由 → 现状缺口 → 落地建议（符合本仓库架构）→ 优先级**。

---

### 3.1【P0】答案生成 / 判官 / Reader 全链路接入会话上下文

**借鉴技术**：多轮 RAG 的「上下文一致生成」——检索、改写、生成、判官共享同一份对话状态，避免答案侧失明。

**借鉴理由**：这是多轮问答质量的**最大杠杆**，且与 issue 主题「多轮问答」直接对应。当前检索已用上下文、生成却没有（缺口 A、B），修复它是「让已经存在的多轮能力真正兑现」。

**落地建议**（向后兼容、不破坏现有调用方）：
1. `AnswerGenerator.generate` 与 `Reader.answer` 增加可选 `context: ConversationContext | None = None`（协议默认 `None`，旧实现/第三方实现零破坏）。
2. `reader/prompts.py:answer_messages` 在 system/user 之间插入「Conversation so far」段（复用 `context.as_text`，受 `max_context_tokens` 联合预算约束）。
3. `qa/engine.py:456` 与 `agent/engine.py:340` 把已持有的 `context` 透传下去（两处本就有 `context` 变量在作用域内，改动极小）。
4. `RetrievalGrader.grade` 与 `FaithfulnessJudge.judge` 同步加可选 `context`（lenient：协议新参数默认 `None`，prompt 仅在非空时拼接）。

**风险/成本**：低。纯参数透传 + prompt 拼接；现有测试传 `context=None` 即兼容。需注意 Context-Cliff 预算要把上下文 token 一并计入，防止「上下文挤掉证据」。

**优先级**：**P0**。建议作为本次优化的第一块落地。

---

### 3.2【P0】后端自动回灌会话上下文 + 滚动记忆摘要

**借鉴技术**：「Conversation Memory」+「Sliding window + rolling summary」——长会话用最近 N 轮原文 + 早期摘要，避免硬截断丢事实。

**借鉴理由**：当前后端不自动重建 context（缺口 C），多轮能力依赖前端正确性，脆弱；且 `max_turns=4` 硬截断在长会话里丢关键约束。这是「让多轮在生产里可用」的必备基础设施。

**落地建议**（两层，均可渐进）：
1. **自动回灌（低风险，先做）**：`QAService.ask` 在 `context is None` 时，从 `QASessionStore` 取该 `kb_id` 最近 N 轮（user+assistant）重建 `ConversationContext`。可加配置开关 `auto_hydrate_context: bool = True` 与 `context_turns: int`。请求体显式传 context 时仍以请求为准（不覆盖）。
2. **滚动摘要（中风险，后做）**：新增 `qa/memory.py`：
   - 一个 `ConversationMemory` 协议（`observe(turn)` / `summarize() -> str` / `window(n) -> ConversationContext`）+ `SummarizingMemory`（LLM 实现，lenient→strict，identity 退化 `IdentityMemory`）。
   - 当累积轮数超阈值，把最旧若干轮压缩成一段 `summary`，`as_text` 时输出 `Summary: ...` + 最近 N 轮原文。
   - 与 `QATurn` 解耦：记忆是查询期构造的视图，`QASessionStore` 仍是唯一事实源。

**架构契合**：`query/context.py` 已是纯数据值对象；记忆层放 `qa/` 复用 `LLMClient`，不污染 `query/`（保持 `query/` 不依赖 `retrieve` 的既有分层）。

**优先级**：**P0**（自动回灌）/ **P1**（滚动摘要）。

---

### 3.3【P1】Agent 「先规划后执行」模式 + 并行检索

**借鉴技术**：Plan-and-Solve / ReWOO 式「先产出结构化计划，再（可并行）执行，最后合成」——相比纯 ReAct 的贪心单步，对多跳/对比题更稳、可并行降延迟。

**借鉴理由**：当前 Agent 是「一步一决策」的串行 ReAct（缺口 D），对「对比 A/B/C」类问题容易在第二步才意识到要查 B，延迟高且易遗漏。规划式分解能让多跳覆盖率与延迟双双改善。

**落地建议**（范式可插拔，沿用本仓库「控制器即 Protocol」惯例）：
1. 新增 `agent/planner.py`：一个 `AgentPlanner` 协议（`plan(question, context) -> list[SubQuery]`）+ `LLMAgentPlanner`（lenient→strict，输出结构化子查询列表）+ `IdentityPlanner`（退化：返回 `[question]`）。
2. `AgenticQAEngine` 增加可选 `planner=`：当接入时，先 `plan` 出全部子查询，用现有 `_multi_retrieve`（`qa/engine.py:618` 已实现的 RRF 多路融合）**并行**检索并去重，再进入 controller 判断「是否够 / 要不要再补」。
3. 保持 `IdentityPlanner` 退化路径 → 不接 planner 时行为与现在完全一致（回归安全网）。

**注意**：子查询去重 + 证据 `_merge_evidence`（`agent/engine.py:105`）已存在，可直接复用。

**优先级**：**P1**。

---

### 3.4【P1】Agent 答案级自反思闭环（把 QAEngine 的 grader/refiner 移植到 Agent）

**借鉴技术**：CRAG / Self-RAG 式「答案级反思」——生成后判质量，差则补检索/改写/重答，而非 best-effort 直出。

**借鉴理由**：当前 `QAEngine` 有完整的 retrieval-side 自反思（`qa/engine.py:556`），但 `AgenticQAEngine` 没有继承（缺口 E），导致 Agent 攒够证据但答得差时无法纠偏。把已验证可用的机制平移过来，成本低、收益直接。

**落地建议**：
1. `AgenticQAEngine.__init__` 增加可选 `retrieval_grader=` / `query_refiner=` / `min_relevance=`（与 `QAEngine` 同名同默认）。
2. 在 `agent/engine.py` 合成前，对 `state.evidence` 做一次 grader 打分：低于阈值则让 refiner 基于当前证据 + 原问题产出新子查询，再走一轮检索（计入 `max_iterations` 预算，防失控）。
3. 答案生成后若 `faithfulness < min_faithfulness` 且还有迭代余量，可触发一次「针对性补检索」（用判官 reasoning 当作 refiner 的反馈信号）。

**架构契合**：`RetrievalGrader` / `QueryRefiner` 已是 Protocol，直接注入即可，core 零改动。

**优先级**：**P1**。

---

### 3.5【P2】扩展 Agent 工具空间（结构化检索 + 数值计算）

**借鉴技术**：Tool-augmented RAG —— 在「检索」之外加「按标签/实体/时间范围过滤检索」「计算器」「SQL」等结构化工具，让 Agent 处理对比/数值题。

**借鉴理由**：当前 `ActionType` 仅二元（缺口 D），无法表达「按 year=2024 过滤」「算增速差」这类结构化意图。IdeaBlock 自带 `tags`/`entities`/`source.locator`，天然适合结构化过滤检索。

**落地建议**（渐进、不破坏现有二元控制器）：
1. `ActionType` 增加 `RETRIEVE_FILTERED`（带 `RetrievalFilter` 的检索）与可选 `CALCULATE`（纯表达式求值，stdlib `ast` 限定白名单，无外部依赖）。
2. 工具用 Protocol 表达：`AgentTool.execute(args) -> str`，注册表 `AgentToolkit`，控制器输出 `action + tool_args`（lenient→strict）。
3. 计算工具零依赖（受限 `ast` 解析），结构化检索复用现有 `Retriever.search(filter=)`——无需新后端。

**风险**：中等。扩展动作空间会让 controller prompt 与 schema 变复杂；建议先上 `RETRIEVE_FILTERED`，`CALCULATE` 视实际需求再上。

**优先级**：**P2**。

---

### 3.6【P2】流式输出（on_progress → SSE/流式 HTTP）

**借鉴技术**：流式 Agentic —— 把 thinking/retrieving/synthesizing 的轨迹实时推给前端，改善多轮对话等待体验。

**借鉴理由**：Agent 跑一轮可能数十秒，当前是同步阻塞（缺口 G）。`on_progress` 回调已存在（`agent/engine.py:283`），只差「接到 HTTP 流式响应」这最后一公里。

**落地建议**：
1. `api/app.py` 新增可选流式路由（如 `POST /api/v1/query/stream`），用 FastAPI `StreamingResponse` + SSE，把 `AgentProgress` 逐条推送。
2. 复用现有 `DistillJob`/`JobManager`（`distill/job.py`）的轮询态机思路：也可把 agent run 包成 pollable job（roadmap 里已规划 `/api/v1/query/agent`），两者择一。
3. 前端按 `phase` 渲染「思考中…检索中…合成中…」。

**架构契合**：`AgentProgress`（`agent/models.py:130`）字段已对齐 `DistillProgress`，前端可复用同一套轮询/渲染组件。

**优先级**：**P2**（UX，非正确性）。

---

### 3.7【P3】上下文感知的语义缓存键

**借鉴技术**：Context-aware cache key —— 缓存命中需同时匹配「语义」与「对话语境」。

**借鉴理由**：当前 `QACache.lookup(query)`（`qa/engine.py:369`）仅按 query 语义命中（缺口 F），「那联通呢」在不同会话会被误命中。

**落地建议**：缓存键改为 `(query 向量, context 摘要向量)` 的联合相似度，或简单做法——对 `context is None` 的首轮才启用缓存（多轮禁缓存，牺牲一点命中换正确性）。Agent 路径本就禁缓存，主要修 default 路径。

**优先级**：**P3**。

---

## 4. 优先级路线图建议

| 阶段 | 内容 | 预期收益 | 风险 |
|---|---|---|---|
| **第 1 步（P0）** | 缺口 A+B：答案生成/判官/Reader/Grader 全链路接入 `context`（§3.1） | 多轮答案质量立即提升，兑现已有的查询侧多轮能力 | 低（参数透传） |
| **第 2 步（P0）** | 缺口 C 上：后端自动从 `QASessionStore` 回灌 context（§3.2.1） | 多轮不再依赖前端正确性，生产可用 | 低 |
| **第 3 步（P1）** | 缺口 E：Agent 答案级自反思（§3.4），复用现成 grader/refiner | Agent 答错能自纠 | 低（注入现成 Protocol） |
| **第 4 步（P1）** | 缺口 C 下：`qa/memory.py` 滚动摘要（§3.2.2） | 长会话不丢早期事实 | 中（新 LLM 调用） |
| **第 5 步（P1）** | 缺口 D 上：Agent 规划式分解 + 并行检索（§3.3） | 多跳/对比题覆盖率与延迟双赢 | 中 |
| **第 6 步（P2）** | 流式输出（§3.6）、扩展工具（§3.5） | UX + 结构化问题能力 | 中 |
| **第 7 步（P3）** | 上下文感知缓存（§3.7） | 多轮下缓存正确性 | 低 |

**关键原则**：第 1-3 步几乎都是「把已存在的多轮/反思能力**接通到答案侧与 Agent**」，改动小、回归安全（identity 退化保证可关），应优先合入；记忆压缩、规划分解、工具扩展属于能力扩展，随后跟进。

---

## 5. 附录：未验证假设（因原文不可读）

以下结论**不依赖**原文，仅依赖源码可证伪的缺口，故独立成立；但若与某篇原文高度重合，属正常——这些是 Agentic RAG 业界共识技术：

- 假设 1：原文讨论了「记忆/摘要压缩」类技术 → 对应 §3.2（锚定缺口 C：`as_text(max_turns=4)` 硬截断 + 无自动回灌）。
- 假设 2：原文讨论了「Plan-then-Execute / 并行检索」→ 对应 §3.3（锚定缺口 D：二元 ActionType + 串行贪心）。
- 假设 3：原文讨论了「答案级自反思 / Self-RAG / CRAG」→ 对应 §3.4（锚定缺口 E：Agent 无答案级闭环）。
- 假设 4：原文讨论了「工具增强 / Tool-use RAG」→ 对应 §3.5。
- 假设 5：原文讨论了「流式 / 可观测 agent」→ 对应 §3.6。

**建议**：若能拿到原文文本（截图/PDF/复制粘贴），可把每篇的核心论点与本报告 §3 一一对照，补一张「原文要点 ↔ SparkSage 落地点」映射表，进一步收敛优先级。当前报告已确保：即便完全不参考原文，所列每一条都是项目里真实存在、可立即着手修复的缺口。

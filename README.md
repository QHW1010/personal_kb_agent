# Personal KB Agent · 个人知识库问答 Agent

一个可直接部署的个人知识库问答服务：**FastAPI + LangGraph StateGraph + 向量检索/重排 + NL2SQL**，
带 JWT 鉴权、SSE 流式输出、Token 成本台账，以及可一键启动的 Docker Compose 编排。

> 定位：求职作品 / 生产级 RAG 骨架。所有 Agent 产出必须通过 Pydantic 模式校验才会返回用户。

---

## 目录

- [核心能力](#核心能力)
- [核心技术亮点](#核心技术亮点)
- [架构](#架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [文档摄入与检索验证](#文档摄入与检索验证)
- [双维度评估：检索 + 忠实度](#双维度评估检索--忠实度)
- [接口示例](#接口示例)
- [SSE 事件协议](#sse-事件协议)
- [环境变量](#环境变量)
- [项目结构](#项目结构)
- [设计取舍](#设计取舍)
- [测试](#测试)
- [路线图](#路线图)
- [演示截图](#演示截图)

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 对话问答 | 检索知识库 → 重排 → 生成，答案附**可追溯引用** |
| 多轮对话 | 引用按轮隔离：历史轮次的 `[n]` 不会污染当前答案，只带出本轮检索到的段落 |
| 流式输出 | SSE 逐 token 推送，首字延迟低；`stream=false` 可切一次性 JSON |
| NL2SQL | 统计类问题自动翻译成**只读** SQL 并执行，返回原始结果行 |
| 鉴权 | bcrypt 密码哈希 + JWT Bearer；注册即签发 Token |
| 成本可控 | 每次调用记录 prompt/completion token 与估算 USD 成本；上下文只带最近 N 轮 |
| 韧性 | 所有外部 I/O 强制超时 + 指数退避重试，失败降级而非 5xx |
| 可替换 | 向量库（Chroma/Qdrant/Pinecone）、Embedding、Reranker、LLM 均为可插拔实现 |

**零 Key 可跑**：默认 `LLM_PROVIDER=fake` 使用离线确定性模型，不需要任何 API Key
就能跑通全链路、跑通测试、看到流式效果。切真实模型只需改环境变量。

---

## 核心技术亮点

| # | 亮点 | 关键做法 | 可验证的结果 |
|---|---|---|---|
| 1 | **LangGraph 三层意图路由** | 寒暄正则直判 → 模型分类（`temperature=0`）→ 规则兜底，三层级联在同一个节点内 | 路由永远有解；寒暄不花 API 成本；模型超时/脏输出自动降级，不会 5xx |
| 2 | **首字节 9.0s → 1.3s** | 打点定位到 `ChatOpenAI` **构造**耗时 8s，整体挪到启动期 `warmup_llm()`；tiktoken 词典缓存固定到 `./data/tiktoken` | TTFT **9047ms → 1389ms（-84.7%）**，41 帧逐 token 到达 |
| 3 | **防幻觉三道闸** | SQL 提示词禁写 `LIMIT`；裸引用标记按本轮条数校正；出口按本轮 `chunk_id` 白名单过滤 | SQL 路线不再出现"引用 0 条却标 `[1]`"；越界编号被丢弃并告警 |
| 4 | **多轮引用隔离** | 历史答案进提示词前抹除 `[n]`；越界编号裁剪；出口白名单兜底 | 第二轮寒暄 citations 恒为 0；连续 RAG 轮引用不跨轮、不指向未检索段落 |
| 5 | **双维度评估集（41 题）** | 检索跑三条臂（向量粗召回 / 全链路 / 线上 top-3）；忠实度拆到 claim 级，由 LLM 逐条核查答案是否被 citations 支持，并用注入式对照实验验证裁判真的会扣分 | 线上 **Recall@3 0.95**、**MRR 0.90**（向量臂 0.85 / 0.77）；39 题 **247 条断言 0 条编造**；裁判自检 **1.00 → 0.67** |
| 6 | **受限网络下的交付** | HTTPS 被阻断时改走 `ssh://git@ssh.github.com:443`，不依赖代理软件 | `github.com:443` 超时的情况下仍可推送，且 `.git/config` 不再存明文 token |

前四项是代码能力，第五项是度量能力，第六项是环境适配 —— 都是这套系统真实踩过并修掉的问题，
不是设计时想当然写上去的。下面「设计取舍」有每一条的根因分析。

---

## 架构

```mermaid
flowchart TB
    subgraph Client
        C[Browser / curl]
    end

    subgraph API["FastAPI 应用"]
        R1[/POST /api/v1/auth/register/]
        R2[/POST /api/v1/chat/]
        R3[/POST /api/v1/knowledge/ingest/]
        AUTH[JWT 依赖注入]
        RL[限流 Redis/内存]
    end

    subgraph Graph["LangGraph StateGraph"]
        RT[intent_router<br/>寒暄直判 / LLM 分类 / 规则兜底]
        RV[retrieve<br/>向量召回 + 重排]
        SQ[sql_builder<br/>NL→SQL→只读校验→执行]
        SY[synthesize<br/>LLM 流式生成]
        DR[direct]
    end

    subgraph RAG
        EM[Embedder<br/>hashing / bge]
        VS[(Vector Store<br/>Chroma / Memory)]
        RR[Reranker<br/>lexical / cross-encoder]
    end

    subgraph Data
        PG[(PostgreSQL / SQLite<br/>users · documents<br/>chat_messages · usage_records)]
        RD[(Redis<br/>可选，自动降级内存)]
    end

    LLM[Chat Model<br/>DeepSeek / OpenAI 兼容<br/>offline 兜底]

    C --> R1 --> AUTH --> PG
    C --> R2 --> AUTH --> RL --> RT
    RT -->|rag| RV --> EM --> VS --> RR --> SY
    RT -->|sql| SQ --> PG --> SY
    RT -->|direct| DR
    SY --> LLM
    SY --> V[Pydantic AgentAnswer 校验]
    V --> SSE[SSE: route → citations → token… → done]
    SSE --> C
    R3 --> EM --> VS
    R3 --> PG
    RL --> RD
```

**请求时序（RAG 路线）**

```
client ──POST /chat──▶ auth ─▶ 限流 ─▶ 读最近 N 轮 ─▶ intent_router
                                                        │  route 事件
                        ┌───────────────────────────────┘
                        ▼
              retrieve → rerank → 组装 context ──▶ citations 事件
                        │
                        ▼
              synthesize（LLM 流式）── token 事件逐个下发 ──▶ client
                        │
                        ▼
              AgentAnswer 校验 ─▶ 落库 + 计费 ─▶ done 事件（含 citations）
```

---

## 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| Web | FastAPI（异步）+ Uvicorn | 自动 OpenAPI，`/docs` 可直接调试 |
| Agent 编排 | LangGraph `StateGraph` | 条件边路由，节点职责单一；`intent_router`/`retrieve`/`synthesize` |
| LLM | DeepSeek `deepseek-chat`（OpenAI 协议） | 换网关只改 `OPENAI_BASE_URL`；`LLM_PROVIDER=fake` 时走离线确定性模型 |
| RAG | 向量召回 + 重排 | Embedding / 向量库 / Reranker 三层可替换 |
| Embedding | fastembed（ONNX）+ `BAAI/bge-small-zh-v1.5` | 512 维，权重约 90MB，走 hf-mirror；不依赖 torch |
| 向量库 | Chroma（默认） | 集合名按模型维度版本化；装不上自动降级进程内实现 |
| 关系库 | PostgreSQL + SQLAlchemy 2.0 | 异步驱动 asyncpg；本地默认 SQLite |
| 迁移 | Alembic | `alembic upgrade head` |
| 缓存/限流 | Redis | 连不上自动降级内存，`/health` 会标 degraded |
| 认证 | JWT（PyJWT）+ bcrypt | 不使用 passlib，规避其与 bcrypt 4.x 的兼容问题 |
| 流式 | SSE + `stream_mode="messages"` | 关闭 nginx 缓冲（`X-Accel-Buffering: no`） |
| 测试 | pytest + pytest-asyncio + httpx | 覆盖认证、RAG、NL2SQL、SSE、守卫 |
| 部署 | Docker Compose | api + postgres + redis；配置就绪，待本机 Docker 环境可用后验证 |

---

## 快速开始

### 方式一：Docker Compose（一键）

> **状态：配置就绪，待本机 Docker 环境可用后验证。**
>
> `Dockerfile` 与 `docker-compose.yml` 已按生产路径写好（多阶段构建、非 root 运行、启动前执行 `alembic upgrade head`、健康检查串联启动顺序），但开发本机当时没有可用的 Docker daemon，因此这条路径**尚未实际跑通**。下方命令是设计意图，首次使用请以实际输出为准；本地已完整验证的是下方的「方式二」。

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f api
```

服务就绪后：

- Swagger UI: <http://localhost:8000/docs>
- 健康检查: <http://localhost:8000/health>

首次启动会自动执行 `alembic upgrade head` 建表，并灌入 4 篇演示语料。

### 方式二：本地开发

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# 首次运行会把 bge-small-zh 的 ONNX 权重（约 90MB）下到 ./data/models
# 走 hf-mirror.com 镜像，国内网络通常 30 秒内完成
python -m scripts.ingest

# 启动（默认 SQLite + Chroma + fastembed 真实语义向量）
uvicorn app.main:app --reload --port 8000
```

> 完全离线或 CI 环境可把 `.env` 里的 `EMBEDDING_PROVIDER` 设成 `hashing`（零依赖、非语义），
> 或 `VECTOR_BACKEND=memory` 走进程内向量库 —— 两条降级路径都不需要联网。

接真实大模型（DeepSeek / 通义 / vLLM / Azure，只要兼容 OpenAI 协议）：

```bash
pip install -r requirements-llm.txt
```

然后在 `.env` 里填：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

> Key 只写进 `.env`（已被 `.gitignore` 排除），`.env.example` 里只有占位符。
> 换成 DeepSeek 后记得同步 `PRICE_PROMPT_PER_1K` / `PRICE_COMPLETION_PER_1K`，
> 否则用量台账里的美元数字是 OpenAI 的价格。

### 端到端验证（真实流式）

```bash
python -m scripts.e2e_chat
python -m scripts.e2e_chat "知识库里一共有多少篇文档？"
python -m scripts.e2e_chat --base-url http://127.0.0.1:8000   # 打已启动的服务
```

脚本跑的是完整链路：ASGI/HTTP → JWT → chat_service → LangGraph → DeepSeek → SSE，
逐帧解析并断言"真的流式、真的有引用"，任一断言不过就以非 0 退出。

### 冒烟验证

```bash
# 1) 注册拿 Token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"sup3rsecret!"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 2) 流式对话
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"重排序是怎么工作的？","stream":true}'
```

---

## 文档摄入与检索验证

`docs/` 下的 Markdown / 纯文本会被切块写入 Chroma，检索命中时会带上**来源文件名、段落号、
字符区间、章节路径**四项元数据，可以精确回原文定位。

### 摄入

```bash
python -m scripts.ingest                     # 增量（已入库的跳过）
python -m scripts.ingest --force             # 强制重建（覆盖同名 source）
python -m scripts.ingest --dry-run           # 只看切块结果，不写库
python -m scripts.ingest --pattern "*.md"    # 只摄入匹配文件
```

脚本启动时会先打印本次实际生效的配置，避免"改了半天 .env 其实没读到"：

```
Embedder  : fastembed / BAAI/bge-small-zh-v1.5 (dim=512)
Collection: kb_documents_fastembed_baai_bge_small_zh_v1_5_512
Backend   : chroma
Chunk     : size=400 overlap=60
[1/6] agent-orchestration.md      ok  title='Agent 编排与流式输出' chunks=9  elapsed=1.4s
[3/6] evaluation-methodology.md   ok  title='评估方法学：检索与忠实度双维度' chunks=6  elapsed=0.3s
[6/6] rag-pipeline.md             ok  title='RAG 检索管道设计' chunks=10 elapsed=0.5s
汇总: 新增 5 篇 / 跳过 1 / 失败 0
本次新增切片 35 | 向量库总数 44 | 台账切片 44
```

单篇失败不会中断整批；结束时若"向量库条数"与"数据库台账条数"不一致，会自动触发重建并说明原因。

> 服务启动时还会灌 4 篇演示语料（`RAG 链路说明` / `SQL 查询能力` 等），让新 clone 的仓库
> 零配置就能问出东西。只想看到自己的文档时把 `SEED_DEMO_DOCS` 设为 `false`。

### 检索验证

```bash
python -m scripts.test_retrieval "不可违反的工程约束有哪几条？"
python -m scripts.test_retrieval --top-k 10 --top-n 5 --no-rerank "重排序怎么做"
python -m scripts.test_retrieval --verify "SQL 为什么禁用硬编码 LIMIT"   # 回查原文偏移
```

输出（真实运行结果）：

```
#1  rerank=0.6034  vector=0.6388  text
    source : kb-core-design.md
    title  : Personal KB Agent 个人知识库问答系统核心设计文档
    section: Personal KB Agent 个人知识库问答系统核心设计文档 > 四、不可违反的工程约束
    para   : #10   chars=[968,1342)  len=374
    偏移校验: ✅ content[char_start:char_end] == 切片文本
    preview: ## 四、不可违反的工程约束 1. **强制输出校验**：所有 Agent 的输出必须经过...
```

同时给出 `rerank`（融合分，最终排序依据）与 `vector`（余弦相似度）两个分数：
两者不一致的条目正是重排起作用的地方。`--verify` 会拿 `char_start/char_end` 回查数据库原文，
校验"落库时记的偏移"与"当时切的文本"完全一致 —— 这是引用能高亮的前提。

### 切分策略

按**结构**切，不按固定字数暴力切：

- Markdown 标题、围栏代码块、表格各自成块，代码块和表格不会被拦腰截断，`kind` 一并入库（`text` / `code` / `table`）；
- 超长块走滑窗，但若已在新行断开就不再叠加 overlap（避免产生 50 字左右的碎片）；
- 窗口起点前移到最近的句子边界（`\n。！？；`），不会出现从半个词开始的切片；
- 纯标题块并入紧随其后的正文块 —— 否则检索会命中一个只有 13 字标题的"幽灵片段"；
- 强制不变量：`content[char_start:char_end] == chunk.text`，由测试守护。

块长默认 400 字符、重叠 60。`bge-small-zh-v1.5` 上下文 512 token，中文接近 1 字 1 token，
留出的余量是为了避免 tokenizer 静默截断尾部。

---

## 双维度评估：检索 + 忠实度

「检索准」和「没编造」是两件事，必须分开测，也必须放在一张表里看：段落找对了，模型仍可能在
引用齐全的外壳下补一句"因此生产环境建议关闭该功能"；反过来，模型也可以忠实地基于错误的段落
给出一个「忠实的错误答案」。所以评估集是二维的 —— 41 道题、同一批题、两个维度、一张交叉表。

| 维度 | 指标 | 回答的问题 | 入口 |
|---|---|---|---|
| 检索 | Recall@3 / Recall@5 / MRR | 该找到的段落找到了吗 | `scripts/evaluate.py` |
| 忠实度 | claim 级支持率 | 答案是否只说了引用里的话 | `scripts/evaluate_faithfulness.py` |
| 汇总 | 交叉表 | 哪些题"有引用但编造" | `scripts/evaluate_all.py` |

```bash
python -m scripts.evaluate_all                  # 两个维度 + 交叉表（需要真实模型）
python -m scripts.evaluate_all --retrieval-only # 没 Key 也能跑，只看检索
python -m scripts.evaluate_all --json | jq .    # 机器可读
python -m scripts.evaluate_all --min-recall 0.85 --min-score 0.95   # CI 门禁
python -m scripts.evaluate --rows               # 逐题看排名，排障用
```

真实输出（41 题 / 10 篇文档 / 44 个切片，DeepSeek 生成 + 裁判，本机 45 秒跑完）：

```
══════════════════════════════════════════════════════════════════════════════
  个人知识库 Agent · 双维度评估报告
══════════════════════════════════════════════════════════════════════════════
  语料      : 10 篇文档 / 44 个切片
  Embedder  : fastembed/BAAI/bge-small-zh-v1.5 (dim=512)
  题量      : 41   语料来源: docs/
──────────────────────────────────────────────────────────────────────────────
  维度一 · 检索（Recall@K / MRR）
──────────────────────────────────────────────────────────────────────────────
  向量粗召回        Recall@3=0.85  Recall@5=0.93  MRR=0.768
  全链路 top-5      Recall@3=0.95  Recall@5=0.98  MRR=0.904
  线上 top-3        Recall@3=0.95  MRR=0.898
──────────────────────────────────────────────────────────────────────────────
  分档（线上 top-N 口径）
  分层        n=1    R@3=1.00  MRR=1.000
  路由        n=3    R@3=0.67  MRR=0.750
  约束        n=2    R@3=0.50  MRR=0.500
  …
──────────────────────────────────────────────────────────────────────────────
  维度二 · 忠实度（LLM 裁判，claim 级）
──────────────────────────────────────────────────────────────────────────────
  参与判定  : 39 题   弃答 2   判定失败 0   平均引用 3.0
  原子断言  : 247 条  →  被支持 247  不被支持 0
  题级平均  : 1.0000   微平均 1.0000   满分题 39/39
  裁判自检  : 原答案 1.00 → 注入编造后 0.67   ✅ 检出
══════════════════════════════════════════════════════════════════════════════
  交叉：检索 × 忠实度
   · 双通过（命中且无编造）      : 39
   · 命中但有编造（最危险）      : 0
   · 未命中但答案全被支持        : 0
   · 双失败                      : 0
   · 拒答（不判幻觉，也不算答对）: 2
══════════════════════════════════════════════════════════════════════════════
```

| 指标 | 定义 | 向量粗召回 | 全链路 top-5 | 线上 top-3 |
|---|---|---|---|---|
| **Recall@3** | 前 3 条里至少有一条相关片段 | 0.85 | 0.95 | **0.95** |
| **Recall@5** | 前 5 条里至少有一条相关片段 | 0.93 | 0.98 | — |
| **MRR** | 首个相关片段排名的倒数均值 | 0.768 | 0.904 | **0.898** |
| **忠实度** | 被引用支持的断言 ÷ 全部断言 | — | — | **1.000（247 条）** |

（分档表共 23 个分组，其中 21 组 `R@3=1.00`；「路由」0.67 与「约束」0.50 是仅有的两处缺口，
原因在下面「这组数字能说明什么」里逐题拆过。）

**判分规则故意做严**：一个片段算"相关"，必须**来源文件名对得上**且**正文含这道题的锚点原文**。
只对文件名，会让同一篇文档里任意片段都算命中；只看锚点，会让别的文档碰巧复述同一句话时判错。

### 检索：为什么跑三条臂

- **向量粗召回**：只看 embedding 本身，不重排；
- **全链路 top-5**：召回 8 条 → 重排 → 保留 5 条，衡量"如果全给模型"的上限；
- **线上 top-3**：与 `RAG_RERANK_TOP_N` 一致，是提示词里真实存在的那 3 条。

三臂的读法：向量 → 全链路，`Recall@3` 涨 0.10、`MRR` 涨 0.136，这是重排的贡献；全链路 → 线上，
`MRR` 掉 0.006，因为有 1 道题的正确片段排在第 4 位、被 3 条上限截掉了。只跑单一深度，这两个数
会混成一个 —— 既说不清重排值多少，也看不出 top-N 截断的代价。

### 忠实度：拆到 claim 才算评估

判定不让模型打一个"可信度分"（那种分数没有定义，也不可复现），而是把答案拆成原子断言逐条核查：

1. **claim 级**：分数 = 被支持断言 ÷ 全部断言，同时给题级平均（用户视角）与微平均（按断言加权）。
2. **证据只给 citations**，不给整篇文档。给了全文，"文档里确实这么写"会被当成"答案被引用支持"，
   而这恰好是要测的东西 —— 答案必须被它自己声称的依据支持。
3. **弃答单列**。答案只说"资料不足"时记 abstained，既不算 1.0 也不算 0 —— 否则等于奖励模型不回答。
4. **裁判自检**（`--calibration`）：往真实答案里注入一句与引用无关、但语气专业的编造
   （"默认开启多租户隔离，并向 Prometheus 推送指标"），分数从 1.00 掉到 0.67。
   没有这一步，满分只能说明裁判睡着了。

### 这组数字能说明什么，不能说明什么

2 道未命中，两道都值得说：

- **「对话上下文只保留最近 N 轮…」0 命中**。答案确实在库里，但它和另外 4 条并列约束挤在同一个
  374 字切片里，单条约束的语义被稀释，在向量空间里不像一段"关于上下文窗口"的内容。
  改进方向很明确：列表型文档要二次切分，一条约束一片。
- **「为什么寒暄不交给大模型判断？」向量与全链路都排第 4**，被线上 top-3 截掉。这是 top-N 的
  直接代价，也正是上面「线上 MRR 比 top-5 臂低 0.006」的唯一来源。

2 道拒答，正好就是这 2 道。模型说的是"资料不足"而不是编一个理由，所以"命中但有编造"是 0。
两件事合起来才是防幻觉的真实效果：检索兜不住的时候，系统会承认不知道。

不能说明的：

- **域内自建**。题目由读过语料的人出，用词贴近原文，真实用户会更口语。绝对值只能用于回归对比，
  不能对外宣称"检索准确率 95%"。
- **基数小**。44 个切片，随机命中基线约 `3/44 ≈ 0.07`；41 道题里涨跌 1-2 道就会让数字动，
  置信区间很宽。
- **同族裁判**。生成与判定都用 `deepseek-chat`，自我偏好会让忠实度偏高。裁判自检只证明"能识别
  明显编造"，不证明"判得准"；要判得准得上异族裁判或人工抽检。
- **单次运行**。生成温度 0.2、裁判温度 0，重跑仍有小幅波动（实测 0-2 题的判定会变）。

所以这组数字的用法是门槛而不是奖状：`--min-recall` / `--min-score` 接进 CI，改切分、换模型、
动提示词之后跑一遍，掉下来就拦住。

单测：`tests/test_evaluate.py`（判分与指标）、`tests/test_eval_set.py`（锚点必须落在单个切片内 ——
文档改措辞会让失效的题立刻报错）、`tests/test_faithfulness.py`（解析健壮性、弃答口径、聚合）。

---

## 接口示例

### 注册 / 登录

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"sup3rsecret!"}'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {"id": "9f2c...", "email": "demo@example.com", "is_active": true, "created_at": "2026-11-10T07:10:00Z"}
}
```

### 非流式提问

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"重排序是怎么工作的？","stream":false}'
```

```json
{
  "conversation_id": "0b9c...",
  "message_id": "4d1e...",
  "answer": "围绕「重排序是怎么工作的？」，知识库中《RAG 链路说明》给出了以下要点：\n\n1. 检索增强生成分为两阶段：第一阶段用向量库做粗召回，默认取前 8 个片段...",
  "route": "rag",
  "citations": [
    {"chunk_id": "a1b2...", "document_id": "7c3d...", "title": "RAG 链路说明", "source": "demo", "snippet": "检索增强生成分为两阶段...", "score": 0.87}
  ],
  "sql": null,
  "rows": null,
  "confidence": 0.87,
  "needs_clarification": false,
  "usage": {"model": "offline-deterministic", "prompt_tokens": 412, "completion_tokens": 96, "total_tokens": 508, "cost_usd": 0.00012, "latency_ms": 38}
}
```

### 自然语言触发 SQL

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"知识库里一共有多少篇文档？","stream":false}'
```

```json
{
  "route": "sql",
  "sql": "SELECT count(*) AS total_documents FROM documents",
  "rows": [{"total_documents": 4}],
  "answer": "针对「知识库里一共有多少篇文档？」，我把问题转换成了只读 SQL 并执行，共返回 1 行结果：\n\n- {'total_documents': 4}",
  "citations": []
}
```

### 上传文档

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"团队规范","content":"提交前必须跑 pytest。","source":"wiki"}'
```

### 其他

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 依赖健康度（db / 向量库 / 缓存 / llm） |
| GET | `/health/live` | 存活探针，不碰外部依赖 |
| GET | `/api/v1/auth/me` | 当前用户 |
| GET | `/api/v1/chat/usage` | Token 与成本汇总 |
| GET | `/api/v1/chat/history/{conversation_id}` | 回看会话 |
| POST | `/api/v1/knowledge/search` | 只跑检索链路（含重排） |
| GET | `/api/v1/knowledge/stats` | 索引与检索配置概览 |

---

## SSE 事件协议

`POST /api/v1/chat`（`stream=true`）返回 `text/event-stream`：

```
event: start
data: {"conversation_id":"0b9c...","model":"deepseek-chat"}

event: route
data: {"route":"rag","source":"llm","reason":"询问设计原因需查文档"}

event: citations
data: {"citations":[{"source":"kb-core-design.md","paragraph_index":10,"char_start":968,"char_end":1342,"score":0.4356}]}

event: token
data: {"t":"禁止在 SQL 提示词里硬编码 LIMIT，"}

event: token
data: {"t":"是为了避免多行答案被截断。"}

event: done
data: {"conversation_id":"0b9c...","message_id":"4d1e...","answer":"...","route":"rag","citations":[...],"usage":{...}}
```

| 事件 | 时机 | 载荷 |
|---|---|---|
| `start` | 流开始 | `conversation_id`, `model` |
| `route` | 意图确定后、正文之前 | `route`, `source`(`llm`/`fast_rule`/`fallback`), `reason` |
| `citations` | 检索完成、正文之前 | `citations[]`（来源、段落号、字符区间、分数） |
| `token` | 每个增量 | `{"t": "文本片段"}` |
| `done` | 正常结束 | 完整 `ChatResponse`（含引用、SQL、rows、usage） |
| `error` | 异常 | `{"code","message"}`，客户端应停止拼接 |

前端对接要点：`done` 覆盖所有结构化字段，`token` 只用于实时打字效果；
若中途收到 `error`，丢弃已拼接内容并提示重试。

`route` 与 `citations` 都在第一个 `token` 之前到达，所以界面可以先把
"走的是哪条链路"和"依据了哪几个文档段落"渲染出来，再流式填正文。
`source=llm` 表示意图由模型判定，`fast_rule` 是寒暄直判，`fallback` 是模型不可用后回落规则。

---

## 环境变量

完整清单见 [`.env.example`](.env.example)。关键项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SECRET_KEY` | 占位值 | **生产必须替换**（`secrets.token_urlsafe(48)`） |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/app.db` | Compose 下覆盖为 PostgreSQL |
| `AUTO_CREATE_TABLES` | `true` | 生产设 `false`，改用 Alembic |
| `REDIS_URL` | 空 | 留空 = 进程内存缓存 |
| `VECTOR_BACKEND` | `auto` | `auto` / `chroma` / `memory` |
| `CHROMA_COLLECTION` | `kb_documents` | 基础名；fastembed 会派生带模型维度的后缀 |
| `EMBEDDING_PROVIDER` | `fastembed` | `hashing` 零依赖兜底 / `fastembed` ONNX / `sentence-transformers` 需 torch |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文小模型，512 维，权重约 90MB |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像；代码里也会 `setdefault`，双保险 |
| `FASTEMBED_CACHE_DIR` | `./data/models` | 权重落地目录，Compose 下挂卷复用 |
| `RAG_TOP_K` / `RAG_RERANK_TOP_N` | `8` / `3` | 粗召回条数 / 精排保留条数 |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `400` / `60` | 块长与重叠，需给 512 token 留余量 |
| `RAG_MODEL_LOAD_TIMEOUT_SECONDS` | `600` | 首次下载模型用，**不能**复用 `TOOL_TIMEOUT_SECONDS` |
| `SEED_DEMO_DOCS` | `true` | 启动时灌 4 篇演示语料；只想看自己 `docs/` 的内容就设 `false` |
| `LLM_PROVIDER` | `fake` | `fake` = 离线确定性；`openai` = 任何 OpenAI 兼容网关（DeepSeek 用这个） |
| `MODEL_NAME` / `LLM_MODEL` | `deepseek-chat` | 两种写法都认；DeepSeek 官方示例用 `MODEL_NAME` |
| `OPENAI_API_KEY` / `LLM_API_KEY` | 空 | 真实模型的 Key，**只写在 `.env` 里，`.env` 已被 gitignore** |
| `OPENAI_BASE_URL` / `LLM_BASE_URL` | 空 | DeepSeek 填 `https://api.deepseek.com/v1` |
| `LLM_TIMEOUT_SECONDS` | `90` | 正文生成超时；复用 `TOOL_TIMEOUT_SECONDS`(10s) 会掐断长答案 |
| `LLM_INTENT_TIMEOUT_SECONDS` | `20` | 意图分类是短任务，超时就回落规则路由 |
| `HISTORY_MAX_TURNS` | `6` | 上下文窗口（轮），防止成本失控 |
| `TOOL_TIMEOUT_SECONDS` / `TOOL_MAX_RETRIES` | `10` / `2` | 外部 I/O 超时与重试 |
| `RATE_LIMIT_PER_MINUTE` | `30` | 每用户每分钟请求上限 |
| `SQL_MAX_ROWS` | `200` | 执行层行数保护（**不是**提示词里的 LIMIT） |

---

## 项目结构

```
personal_kb_agent/
├── AGENTS.md                 # 工程约束（角色/技术栈/不可违反项/验收标准）
├── app/
│   ├── main.py               # 应用装配 + lifespan
│   ├── api/
│   │   ├── deps.py           # JWT 当前用户
│   │   └── v1/               # health / auth / chat / knowledge
│   ├── agents/               # 编排层（不碰 HTTP，不碰 SQL 细节）
│   │   ├── graph.py          # StateGraph 拓扑
│   │   ├── nodes.py          # intent_router / retrieve / sql / synthesize / direct
│   │   ├── state.py          # AgentState
│   │   ├── llm.py            # DeepSeek(OpenAI 协议) / offline 模型工厂 + 预热
│   │   └── prompts.py        # 提示词（SQL 提示词刻意不含 LIMIT）
│   ├── rag/                  # 检索层
│   │   ├── chunking.py       # 结构感知切分（标题/代码块/表格 + 字符偏移）
│   │   ├── embeddings.py     # fastembed / hashing / sentence-transformers
│   │   ├── vectorstore.py    # Chroma / memory（集合按模型维度版本化）
│   │   ├── reranker.py       # lexical / cross-encoder
│   │   ├── retriever.py      # 召回 → 重排 → 带偏移的引用
│   │   ├── citation_guard.py # 引用隔离：历史去标记 / 越界裁剪 / 白名单
│   │   └── ingest.py         # 遍历 docs/ → 切块 → upsert → 台账
│   ├── core/                 # 横切关注点
│   │   ├── config.py errors.py security.py
│   │   ├── resilience.py     # 超时 + 重试
│   │   └── cache.py          # Redis/内存 + 限流
│   ├── db/                   # models / session / base
│   ├── schemas/              # auth / chat / agent（输出校验闸门）
│   └── services/             # chat_service / usage
├── alembic/                  # 迁移脚本
├── docs/                     # 知识库语料（6 篇设计文档，44 个切片）
├── tests/                    # pytest（认证/RAG/NL2SQL/SSE/切分/引用/编排/评估）
├── scripts/
│   ├── ingest.py             # 摄入 docs/，打印生效配置与切块统计
│   ├── test_retrieval.py     # 单问句检索验证（分数 / 来源 / 偏移校验）
│   ├── e2e_chat.py           # 端到端真实问答（LangGraph → DeepSeek → SSE）
│   ├── eval_set.py           # 41 道评估题（检索与忠实度共用同一批）
│   ├── evaluate.py           # 检索：三条臂的 Recall@K / MRR
│   ├── evaluate_faithfulness.py  # 忠实度：生成 + LLM 裁判
│   ├── evaluate_all.py       # 双维度汇总 + 交叉表（CI 入口）
│   ├── faithfulness.py       # 忠实度判定的纯逻辑：提示词 / 解析 / 聚合
│   └── seed_demo.py          # 手动灌演示语料
├── Dockerfile
└── docker-compose.yml
```

分层红线：`api/` 只做协议转换，`services/` 编排业务流程，`agents/` 只关心推理与工具，
`rag/` 只管检索，`db/` 只认 ORM。**不存在**一个文件里同时出现路由、SQL 和检索。

---

## 设计取舍

这里是几个值得在面试里讲的点：

1. **为什么路由用规则而不是 LLM？** 分类任务窄（greeting / 聚合 / 其他），规则可解释、零成本、
   可单测、不会因为模型抽风把统计问题送进 RAG。真实场景可换成小模型分类器，接口不变。
2. **为什么 `LLM_PROVIDER=fake` 也是第一等公民？** 没有 Key 的评审者、CI、以及每一个新
   clone 的开发者都应该能立刻跑通。离线模型走的是标准 LangChain `BaseChatModel`，
   因此 `stream_mode="messages"` 的行为与真实模型完全一致 —— 流式链路不会"只在生产环境才暴露 bug"。
3. **SQL 为什么不在提示词里写 `LIMIT 10`？** 那会把"有多少条记录"这种问题的答案直接截断成 10。
   行数保护放在执行层（`SQL_MAX_ROWS`），并显式回报 `truncated` 标记，让上层知道"这是被截过的"。
4. **为什么 Pydantic 校验放在 service 而不是节点里？** 节点可以局部失败并降级；但**出口只有一个**，
   校验点就必须在出口。`validate_answer()` 是唯一放行函数，非法结构抛 `AgentOutputError`（502）而不是
   悄悄返回脏数据。
5. **向量库为什么允许降级到内存？** Chroma 依赖 `onnxruntime` 等重依赖，在某些平台装不上。
   降级让"跑不起来"变成"跑得慢一点"，同时 `/health` 会如实报告当前后端。
6. **为什么不用 passlib？** passlib 1.7.4 读取 `bcrypt.__about__` 在 bcrypt>=4.1 上会崩。
   直接调 `bcrypt` 三行代码，省掉一个长期踩坑源。
7. **为什么用 fastembed（ONNX）而不是 sentence-transformers？** 后者拖整条 torch 依赖链，
   冷装动辄几百 MB、十几分钟；fastembed 只复用 `onnxruntime`，同一个 `bge-small-zh-v1.5`
   在 512 维下语义效果一致，`pip check` 也不会带进 transformers 的版本冲突。
8. **为什么切分按结构而不是按字数？** 固定窗口会把代码块和表格拦腰切断，检索出来是一段无法阅读的残片。
   这里用 Markdown 标题 / 围栏 / 表格作硬边界，只在块内超长时才滑窗，并且保证
   `content[char_start:char_end] == chunk.text` —— 这条不变量让"引用回原文高亮"从愿望变成事实。
9. **为什么集合名要带模型名和维度？** 换 embedding 模型就换了向量空间，往旧集合里塞新向量不会报错，
   只会让检索结果悄悄变差（384 维 vs 512 维）。集合名内嵌 `provider_model_dim`，
   加上构造时的维度守卫和写入前校验，冲突在启动期就炸出来，而不是等线上发现召回率掉了。
10. **为什么 chunk_id 用 `sha1(document_id|index|text)` 而不是 uuid4？** 相同内容必须落同一个 id，
    否则重跑一次摄入就是一份重复数据。确定性 id + `upsert` 让摄入天然幂等，`--force` 才需要先删。
11. **意图路由为什么让模型判断，而不是纯正则？** 规则只认字面（"多少" + "文档"），
    问"文档里有几种检索策略，区别是什么"会被误判成 SQL。真实模型能读懂意图，
    但会超时、会返回脏格式，所以做成**级联**：寒暄正则直判（零成本）→ 模型分类（默认路径）
    → 规则兜底。三层都在同一个节点里，路由永远不会"无解"。
12. **为什么不能用"薄封装"包一层 OpenAI SDK？** LangGraph 的 `stream_mode="messages"`
    靠 `BaseChatModel` 的流式回调工作。早期实现里有个只转发 `astream` 的普通 Python 对象，
    看着能跑，实际一个 token 都推不出去 —— 客户端会等整段答案生成完才收到响应。
    现在直接返回 `ChatOpenAI`，它本身就是 `BaseChatModel`。
13. **为什么要把 LLM 预热放在启动期？** 首次构造 `ChatOpenAI` 要 8 秒左右
    （`import langchain_openai` 4.7s + openai SDK 客户端初始化 3.5s + tiktoken 词典下载 4.7s），
    这段成本放在第一个用户请求上，TTFT 会从 2.4 秒变成 9 秒。启动期打一次 1-token 请求，
    用户侧就只剩推理时间；顺带把 tiktoken 缓存固定到 `./data/tiktoken`，容器重启不必重下。
14. **多轮对话为什么必须"洗"历史答案？** 上一轮的回答里带着 `[1]`、`[2]`，它随 messages
    一起进提示词，模型很自然地继续写 `[1]` —— 而这一轮 citations 可能是完全不同的三段。
    编号看起来合法，指向的却是上一轮的片段，这种错比"没有引用"更难发现。所以做三层：
    历史答案进提示词前抹掉标记（`sanitize_history`）、越界编号按本轮条数裁剪
    （`align_citation_marks`）、出口再按本轮 `chunk_id` 白名单过滤一次
    （`filter_citations_by_hits`）。三层都是纯函数，各自有单测。
15. **为什么评估要同时给 Recall 和 MRR？** 语料小的时候 Recall 几乎是天花板，看不出差别；
    MRR 衡量"正确片段排在第几"，对重排这类微调更敏感。41 道题的评估集里，向量臂 `MRR` 0.768、
    重排后 0.904，而两个臂的 `Recall@5` 只差 0.05 —— 只看 Recall，重排的价值会被低估一半。
16. **为什么评估要分三条臂，而不是只报一个数？** 线上只把重排后的前 3 条塞进提示词。用 top-5 算
    MRR 会把排在第 4、5 位的命中算成 0.25、0.2，而用户根本看不到这两条，那个数字是虚高的。
    所以「向量 / 全链路 top-5 / 线上 top-3」三条臂分开报：前两条说清重排值多少，第三条对齐真实链路。
    实测正是这条差异暴露了一道题的正确片段排在第 4、被截掉了。
17. **忠实度为什么不用一个整体分，而要拆到 claim？** 让模型给"可信度 0.9"没有定义、不可复现，
    而且它几乎总是偏高。拆成原子断言后，分数有了分母（本项目 247 条），哪一条没被引用支持也说得出来。
    另外两件事必须一起做：证据只给 citations（给了全文就测不出"是否被它声称的依据支持"），
    弃答单列（否则等于奖励模型不回答）。最后还得跑一次注入式自检 —— 往里塞一句编造，
    分数从 1.00 掉到 0.67，才敢说那个满分不是裁判失灵。
18. **HTTPS 被墙时怎么推代码？** `github.com:443` 直连超时、代理对 github 返回 502，
    但 `ssh.github.com:443` 的 TCP 是通的。把 remote 改成
    `ssh://git@ssh.github.com:443/<owner>/<repo>.git` 即可，**完全不走 HTTP 代理**。
    代价是需要把公钥加到 GitHub；收益是 `.git/config` 里不再有明文 PAT。
    排查顺序建议按"DNS 解析 → 直连 → 系统代理 → 备用端口"走，别一上来就怀疑凭据。

---

## 测试

```bash
pytest                # 全量（含 1 例需真实模型的 integration，默认跳过）→ 145 passed, 1 skipped
pytest -q tests/test_chat.py
pytest --cov=app      # 需 pip install pytest-cov
RUN_FASTEMBED_IT=1 pytest -m integration   # 真跑一次 ONNX 模型
```

覆盖范围：

| 文件 | 覆盖点 |
|---|---|
| `test_auth.py` | 注册/登录/重复邮箱/错误密码/JWT 往返/超长密码 |
| `test_rag.py` | 切块、分词、embedding 确定性、重排顺序、入库后可检索、引用结构 |
| `test_chunking.py` | 偏移不变量、代码块/表格不被切断、章节路径、段落号单调、确定性 id、标题必带正文 |
| `test_citation_meta.py` | 旧 payload 反序列化、新元数据字段、去重、脏元数据容错 |
| `test_fastembed_branch.py` | 注入式假模型不下载、前缀生效、缺包报错、未知模型报错 |
| `test_index_bootstrap.py` | `SEED_DEMO_DOCS` 开关、重复启动不重复灌语料、旧调用点兼容 |
| `test_chat.py` | 路由分类、RAG 带引用、NL2SQL 返回正确行、SSE 事件流、会话续接、`top_k` 透传 |
| `test_orchestration.py` | LLM 意图路由与回落、脏输出解析、真实 usage 入账、引用标记清理、SSE 事件顺序 |
| `test_citation_isolation.py` | 历史标记净化（不误伤用户提问）、越界编号裁剪、出口白名单、两轮寒暄零引用、连续 RAG 不越界 |
| `test_evaluate.py` | 判分双条件（来源+锚点）、Recall@K / MRR 计算、评估集完整性、中文表格对齐 |
| `test_eval_set.py` | 41 个锚点必须逐字落在对应文档的**单个切片**内、锚点不能过泛、覆盖 ≥5 篇文档 |
| `test_faithfulness.py` | 裁判输出解析（围栏/裸列表/中文布尔/脏字段）、解析失败不静默通过、弃答记 None、题级与微平均 |
| `test_guards.py` | 只读 SQL 校验、空答案拒绝、`route=sql` 必须带 SQL、提示词无 LIMIT |
| `test_usage.py` | token 估算、成本计算、台账累计、上下文窗口上限、限流 |
| `test_health.py` | 健康检查、存活探针、`/docs` 与 OpenAPI 可用 |

---

## 路线图

- [x] 评测集：41 题 / 44 切片，检索三臂（Recall@K / MRR）
- [x] 忠实度评估：claim 级 LLM 裁判 + 注入式裁判自检（`scripts/evaluate_all.py`）
- [x] 多轮引用隔离：历史轮次的 citations 不污染当前答案
- [ ] 文档解析：PDF / DOCX / 网页抓取入库（当前仅文本）
- [ ] 多租户：文档按 `user_id` 隔离，检索时带 `where` 过滤
- [ ] 混合检索：BM25 + 向量加权融合（当前重排已含词项覆盖率）
- [ ] 列表型文档二次切分：一条约束一片（当前「约束」分组 R@3 只有 0.50，根因是长切片稀释单条约束）
- [ ] 评估集扩容：再扩到数百切片、题量上百道，并引入异族裁判模型消除自我偏好
- [ ] OpenTelemetry 链路追踪 + Prometheus 指标
- [ ] 前端：流式打字 + 引用高亮的对话界面

---

## 演示截图

以下截图均取自本地实际运行（`127.0.0.1:8000`，Chroma 向量库 + 离线确定性模型），不是设计稿。

### Swagger UI · 全量接口

![Swagger UI 全量接口](docs/images/swagger.png)

5 个分组（health / auth / chat / knowledge / meta）共 14 条接口，点开即调，鉴权走右上角 `Authorize`。

### SSE 流式对话

![SSE 流式输出](docs/images/stream.png)

`curl -N` 的真实响应：`event: start` → 24 个 `event: token`（逐字到达，非一次性返回）→ `event: done`。
该次问答 prompt 575 + completion 204 = 779 tokens，耗时 162 ms，带 3 条知识库引用。

### Token 用量与成本汇总

![Token 用量与成本汇总](docs/images/usage.png)

每次对话自动写入一条 `usage_records`；`/api/v1/chat/usage` 按用户汇总调用次数、prompt/completion tokens 与估算 USD 成本，无需额外埋点。

---

## License

MIT

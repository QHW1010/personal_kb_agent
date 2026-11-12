"""评估集：检索维度与忠实度维度共用同一批题。

一题一条，字段含义：
- `question`：用户口吻的提问。故意不抄标题措辞，但要能在语料里被定位到。
- `source`  ：期望命中的文件名（相对 docs/）。判分时与锚点同时校验。
- `anchor`  ：必须能在该文档的**单个切片内**逐字找到的短语。跨切片的锚点永远判不中，
              会把题目变成噪声，`tests/test_eval_set.py` 会逐题验证这一点。
- `topic`   ：分组标签，只用于报告分档统计，不参与判分。

写题的边界：
题目全部来自本仓库 docs/ 的真实内容，属于「域内自建」评估集 —— 用词与原文接近，
绝对分数会偏好于真实用户提问。它的用途是回归对比（改检索/改切分后是否变差），
不是对外宣称的准确率。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    question: str
    source: str
    anchor: str
    topic: str = ""
    note: str = ""


EVAL_SET: list[EvalCase] = [
    # ---------------- rag-pipeline.md：切分 / 嵌入 / 向量库 / 重排 ----------------
    EvalCase(
        question="检索层为什么可以脱离 Agent 单独评测？",
        source="rag-pipeline.md",
        anchor="不认路由、不拼提示词、不调用大模型",
        topic="分层",
    ),
    EvalCase(
        question="只含标题的切片为什么要并到后面的正文里？",
        source="rag-pipeline.md",
        anchor="只含标题的幽灵片段会被并入紧随其后的正文块",
        topic="切分",
    ),
    EvalCase(
        question="引用能精确回原文高亮，靠的是哪条不变量？",
        source="rag-pipeline.md",
        anchor="content[char_start:char_end] == chunk.text",
        topic="切分",
    ),
    EvalCase(
        question="块长为什么默认取 400 而不是正好占满模型上下文？",
        source="rag-pipeline.md",
        anchor="bge-small-zh-v1.5 的上下文是 512 token",
        topic="切分",
    ),
    EvalCase(
        question="生产路径用哪个嵌入模型，向量是多少维？",
        source="rag-pipeline.md",
        anchor="fastembed 加载 BAAI/bge-small-zh-v1.5，输出 512 维向量",
        topic="嵌入",
    ),
    EvalCase(
        question="离线或 CI 环境里嵌入会降级成什么，代价是什么？",
        source="rag-pipeline.md",
        anchor="降级路径是 hashing，384 维、零依赖、完全确定性",
        topic="嵌入",
    ),
    EvalCase(
        question="嵌入调用为什么必须丢到线程池里跑？",
        source="rag-pipeline.md",
        anchor="必须经 asyncio.to_thread 丢到线程池",
        topic="嵌入",
    ),
    EvalCase(
        question="向量库的集合名为什么要内嵌模型名和维度？",
        source="rag-pipeline.md",
        anchor="集合名内嵌 embedding 的 provider、model 与维度",
        topic="向量库",
    ),
    EvalCase(
        question="切片 id 为什么用哈希而不是随机 uuid？",
        source="rag-pipeline.md",
        anchor="chunk_id 用 sha1 对 document_id、index、text 三者拼接后取哈希",
        topic="向量库",
    ),
    EvalCase(
        question="lexical 重排的三项权重分别是多少？",
        source="rag-pipeline.md",
        anchor="0.55 乘以向量余弦相似度",
        topic="重排",
    ),
    EvalCase(
        question="精排为什么只保留 3 条，不把 8 条全给模型？",
        source="rag-pipeline.md",
        anchor="默认粗召回 8 条、精排保留 3 条",
        topic="重排",
    ),
    EvalCase(
        question="启动期的索引自检为什么不去看库里有没有文档？",
        source="rag-pipeline.md",
        anchor="比对数据库台账的切片总数与向量库的实际条数",
        topic="索引",
    ),
    # ---------------- agent-orchestration.md：拓扑 / 路由 / 流式 ----------------
    EvalCase(
        question="主图一共几个节点，怎么分流的？",
        source="agent-orchestration.md",
        anchor="主图只有五个节点",
        topic="拓扑",
    ),
    EvalCase(
        question="为什么路由节点不能直接叫 route？",
        source="agent-orchestration.md",
        anchor="节点 id 不能与 state 的键重名",
        topic="拓扑",
    ),
    EvalCase(
        question="为什么寒暄不交给大模型判断？",
        source="agent-orchestration.md",
        anchor="八百毫秒不值得",
        topic="路由",
    ),
    EvalCase(
        question="意图分类用的模型参数是怎么设的？",
        source="agent-orchestration.md",
        anchor="temperature 固定 0、max_tokens 限制 64",
        topic="路由",
    ),
    EvalCase(
        question="模型路由失败以后，这次请求的意图来源会被标成什么？",
        source="agent-orchestration.md",
        anchor="意图来源标记为 fallback",
        topic="路由",
    ),
    EvalCase(
        question="行数保护为什么不能写在提示词里？",
        source="agent-orchestration.md",
        anchor="行数保护放在执行层，不放在提示词里",
        topic="SQL",
    ),
    EvalCase(
        question="流式输出时为什么只放行一个节点的 token？",
        source="agent-orchestration.md",
        anchor="只有 synthesize 节点发出的 token 才放行",
        topic="流式",
    ),
    EvalCase(
        question="流式生成为什么不让它自动重试？",
        source="agent-orchestration.md",
        anchor="重试对流式是毒药",
        topic="流式",
    ),
    EvalCase(
        question="如果一个 token 都没发出去就报错了，系统怎么兜底？",
        source="agent-orchestration.md",
        anchor="拿到完整答案后再按 24 字切片推给客户端",
        topic="流式",
    ),
    # ---------------- kb-core-design.md：设计稿与工程约束 ----------------
    EvalCase(
        question="这个系统的项目定位是什么？",
        source="kb-core-design.md",
        anchor="生产级的个人知识库问答 Agent",
        topic="定位",
    ),
    EvalCase(
        question="后端 Web 框架有哪些硬性要求？",
        source="kb-core-design.md",
        anchor="FastAPI（要求异步，自动生成 OpenAPI 文档）",
        topic="技术栈",
    ),
    EvalCase(
        question="SQL 子图里的 validate_sql 节点具体做什么？",
        source="kb-core-design.md",
        anchor="执行 dry-run 验证语法",
        topic="SQL",
    ),
    EvalCase(
        question="SQL 子图里的 fetch_schema 节点对表结构有什么要求？",
        source="kb-core-design.md",
        anchor="动态读取当前数据库 schema，严禁硬编码表结构",
        topic="SQL",
    ),
    EvalCase(
        question="对话上下文只保留最近 N 轮，是为了防止什么问题？",
        source="kb-core-design.md",
        anchor="防止上下文窗口爆炸",
        topic="约束",
    ),
    EvalCase(
        question="代码分层这条约束具体要求各层不能碰什么？",
        source="kb-core-design.md",
        anchor="api 不含 SQL、agents 不碰 HTTP、rag 不认路由",
        topic="约束",
    ),
    # ---------------- ops-and-deploy.md：启动 / 超时 / 运维 ----------------
    EvalCase(
        question="一键启动时会拉起哪几个容器？",
        source="ops-and-deploy.md",
        anchor="拉起三个容器：api、postgres、redis",
        topic="部署",
    ),
    EvalCase(
        question="上线前必须替换的密钥用什么命令生成？",
        source="ops-and-deploy.md",
        anchor="secrets.token_urlsafe(48)",
        topic="配置",
    ),
    EvalCase(
        question="为什么超时要分成好几档，而不是共用一个值？",
        source="ops-and-deploy.md",
        anchor="超时分四档，不能混用",
        topic="配置",
    ),
    EvalCase(
        question="启动期预热模型失败了，服务还会起来吗？",
        source="ops-and-deploy.md",
        anchor="预热失败不阻断启动",
        topic="启动",
    ),
    EvalCase(
        question="为什么存活探针里不能检查外部依赖？",
        source="ops-and-deploy.md",
        anchor="把外部依赖写进存活探针，Redis 抖一下就会触发容器重启",
        topic="运维",
    ),
    EvalCase(
        question="github 直连推送不通时，改走哪个地址？",
        source="ops-and-deploy.md",
        anchor="ssh.github.com 的 443 端口",
        topic="运维",
    ),
    # ---------------- api-and-security.md：认证 / 错误码 / 协议 ----------------
    EvalCase(
        question="密码哈希为什么不用 passlib？",
        source="api-and-security.md",
        anchor="它 1.7.4 版本会读取 bcrypt 的私有属性",
        topic="认证",
    ),
    EvalCase(
        question="访问令牌用什么算法签名，默认多久过期？",
        source="api-and-security.md",
        anchor="JWT 用 HS256 签名，默认有效期 1440 分钟",
        topic="认证",
    ),
    EvalCase(
        question="输出校验失败为什么返回 502 而不是 500？",
        source="api-and-security.md",
        anchor="502 表示上游给了不合规的东西",
        topic="错误码",
    ),
    EvalCase(
        question="为什么要在响应头里关掉反向代理的缓冲？",
        source="api-and-security.md",
        anchor="nginx 默认会缓冲上游响应",
        topic="协议",
    ),
    # ---------------- evaluation-methodology.md：指标定义与偏差 ----------------
    EvalCase(
        question="为什么只测检索召回是不够的？",
        source="evaluation-methodology.md",
        anchor="检索指标回答的是「该找到的段落找到了吗」",
        topic="指标",
    ),
    EvalCase(
        question="MRR 具体是怎么算的？",
        source="evaluation-methodology.md",
        anchor="MRR 取首个相关片段排名的倒数再平均",
        topic="指标",
    ),
    EvalCase(
        question="判分为什么要同时校验来源和锚点两条？",
        source="evaluation-methodology.md",
        anchor="来源文件名对得上，且正文包含这道题指定的锚点原文",
        topic="判分",
    ),
    EvalCase(
        question="随机基线大概怎么估算？",
        source="evaluation-methodology.md",
        anchor="随机基线约等于 K 除以切片总数",
        topic="基线",
    ),
]


def by_topic() -> dict[str, int]:
    """各分组的题量，报告里用来交代覆盖度。"""
    counts: dict[str, int] = {}
    for case in EVAL_SET:
        counts[case.topic] = counts.get(case.topic, 0) + 1
    return counts

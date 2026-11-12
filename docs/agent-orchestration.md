# Agent 编排与流式输出

> 本文对应 `app/agents/` 与 `app/services/chat_service.py` 的实现。

## 一、图拓扑

主图只有五个节点。START 进入 intent_router，条件边按 route 分流：rag 走 retrieve，
sql 走 sql_builder，direct 走 direct。retrieve 与 sql_builder 都汇入 synthesize，
synthesize 与 direct 各自连到 END。

intent_router 决定要不要查、查文档还是查库；synthesize 是唯一产出自然语言的节点；
direct 直接返回能力介绍，不花任何模型调用。条件边的取值只允许 rag、sql、direct，非法值
兜底成 rag，路由不会出现「无解」的分支。

节点 id 不能与 state 的键重名。state 里有 route 和 sql 两个键，所以节点只能叫
intent_router 和 sql_builder——拿 route 当节点名，LangGraph 编译时直接报错。

## 二、三层意图路由

单个意图节点里级联三层判断，任何一层失效都不会让路由无解。

第一层是寒暄正则直判，整句匹配你好、在吗、你是谁、谢谢这类说法。为了一个「你好」花一次
API 调用和八百毫秒不值得，直接判 direct，意图来源标记为 fast_rule。

第二层是模型分类，temperature 固定 0、max_tokens 限制 64。分类任务不该有创造性，也不该
占用正文的额度。模型被要求只输出一个含 intent 与 reason 的 JSON 对象，但真实输出不可控：
可能带 json 代码围栏、可能多一句解释、也可能只回一个单词。解析器兼容这三种形态，实在
解析不出才抛错。

第三层是规则兜底。规则库很窄：寒暄，以及聚合词与实体词同时出现才判 sql，其余一律 rag。
它只认字面，问「文档里有几种检索策略」会被聚合词误导，所以规则只做兜底，不做默认路径。
模型超时或返回脏格式时，意图来源标记为 fallback，路由仍然有答案。

## 三、SQL 子图与只读约束

目标态的 SQL 子图是六节点：check_relevance 判断是否适合查库、fetch_schema 动态读取表
结构、generate_sql 生成查询、validate_sql 做 dry-run 验证、execute_sql 执行、narrate_result
翻译成自然语言。当前实现把前几步压缩进 sql_builder 一个节点，但两条硬约束不变。

第一条，生成的 SQL 必须过 SQLPlan 校验，只允许 SELECT 或 WITH 开头的只读语句。任何
INSERT、UPDATE、DELETE、DROP 都会被 Pydantic 拒绝，根本送不到数据库。

第二条，行数保护放在执行层，不放在提示词里。执行时一次多取一行来判断是否截断，上限默认
200 行，结果里显式带 truncated 标记。提示词里写死 LIMIT 会把「有多少条记录」这类问题的
答案直接截成 10，这是错的。

无 Key 时走离线意图库：八条正则规则覆盖文档数、用户数、对话数、token 用量、成本、平均
延迟、最新文档、切片排行。它保证零配置也能演示真实的 NL2SQL 链路。

## 四、上下文组装

检索命中后按固定模板拼上下文：每个片段带片段编号前缀、标题与正文，编号顺序与 citations
严格一致，这样模型写 [1] 时指向的就是第一条引用。

SQL 路线额外附上 SQL 语句与结果集，并明确告诉模型本次共几行。模型经常在截断的结果上
编造「还有很多」之类的表述，把真实行数写进上下文能压掉一部分。

一条片段都没有时，上下文替换为固定提示「本次没有检索到任何片段」，而不是留空。留空会让
模型自由发挥，而诚实地说「没找到」需要显式提示。

## 五、流式：为什么必须是 BaseChatModel

LangGraph 的 stream_mode 取 messages 时依赖 LangChain 回调机制拿 token，而回调只在
BaseChatModel 里被触发。早期实现用一个普通 Python 对象包了一层 OpenAI SDK，只转发
astream，看着能跑、单测也能过，实际一个 token 都推不出来——客户端会等整段答案生成完才
收到第一个字节。

现在模型工厂直接返回 ChatOpenAI，它本身就是 BaseChatModel；离线模型继承 SimpleChatModel
并实现异步流式方法，按 12 字一片模拟真实 token 粒度。两者的流式行为一致，所以「只在生产
环境才暴露的流式 bug」被提前到了本地。

流式同时订阅 messages 与 values 两种模式：messages 给增量 token，values 给每个节点结束
后的完整状态。只有 synthesize 节点发出的 token 才放行给用户——intent_router 和 sql_builder
也会调模型，它们输出的是 JSON 或 SQL，混进正文会让用户看到一坨脏数据。

事件顺序是刻意设计的：start、route、citations、若干 token、最后 done。route 与 citations
都在第一个 token 之前到达，前端可以先渲染走的哪条链路、依据哪几个段落，再流式填正文。

## 六、失败降级

流式链路有两个失败点，处理方式不同。如果一个 token 都还没发出去就异常，退回一次性调用
模型，拿到完整答案后再按 24 字切片推给客户端——用户仍然能得到答案，只是没有逐字效果。
如果已经发了一部分，只能发 error 事件并终止，因为重试会让客户端收到重复 token。

所以 synthesize 的重试次数是 0：重试对流式是毒药。非流式路径可以重试，流式不行。

## 七、出口校验

所有节点产物最终都经过 validate_answer()，这是唯一放行函数。它做四件事：SQL 路线没能
生成 SQL 时降级为直答而不是冒充查询结果；答案为空时给兜底文案；按本轮命中的 chunk_id
白名单过滤引用；用 Pydantic 校验答案结构，任何字段类型不符直接抛 AgentOutputError，对外
是 502 而不是脏数据。

校验放在 service 而不是节点里，理由是节点允许局部失败并降级，而出口只能有一个。校验点
跟着出口走，非法结构就不可能漏到用户面前。

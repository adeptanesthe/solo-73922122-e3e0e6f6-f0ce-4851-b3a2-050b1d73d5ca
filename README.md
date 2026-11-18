# 流式 CTC 解码工作台

面向语音算法工程师的本地网页工具：上传词表与声学模型逐帧概率，按束宽做 CTC
前缀束搜索，分块推进，随时确认 / 撤销字幕前缀，并查看每条候选的对数概率。

- 后端：FastAPI + NumPy（CTC 前缀束搜索为纯 Python/NumPy 实现，无需下载模型）
- 前端：单个静态页面（原生 JS，无构建步骤）
- 部署：`docker compose up` 即可访问 <http://localhost:8000>

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

### 本机运行

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

需要 Python 3.10+。页面会列出 `samples/` 下的内置样例，点击即可载入，无需任何模型文件。

## 使用流程

1. **加载数据**：上传两个 JSON 文件，或直接点击内置样例。
   - 词表：`["<blank>", "啊", "阿", "波", "播"]`（第 0 项为 blank，最多再加 8 个单字）
   - 帧概率：`{"frames": [[0.9, 0.06, 0.04], ...]}`，最多 2000 帧，每帧长度等于词表，
     取值 `0～1`（`0` 表示 `log 0 = -∞`）
2. **设置束宽**（1～16）与分块大小，点击「推进分块」或「连续推进」。
3. **查看候选**：表格按分数排序列出每个候选的 `log P`、概率、末字的 blank / 非 blank
   路径概率。把「查看帧」滑到当前帧之前的任意帧，可以回放当时已推进到的候选（只读）。
4. **确认前缀**：点「确认当前候选」，或在输入框中输入更短的前缀后确认。确认后只保留
   兼容候选，后续推进不会再把不兼容路径带回束中。
5. **撤销确认**：恢复「未做该次确认时」在当前帧应有的束与分数（不是简单解除文字限制），
   已推进的帧不会回退。

## 关键语义与实现

### CTC 规则

- 空白记作第 0 类；同一文本的所有路径在**剪枝前**按 `log-sum-exp` 合并，
  因此同一字符串不会占用两个束位。
- 相同学分按文本的 Unicode 码点升序排列（`sorted(key=(-logp, text))`）。
- 连续重复字符只有在中间有 blank 时才会展开成两个字符；无 blank 的连续同字
  坍缩为一个字符。
- 每束保留 `logp_blank` / `logp_nonblank`，即末字以 blank 结束或非 blank 结束的
  路径概率质量。

### 任意分块一致性

`CTCBeamDecoder.advance` 只是对 `step` 的循环，确认限制也在 `step` 剪枝前应用，
所以「一次推进 N 帧」与「逐帧推进 N 次」得到完全相同的束与分数。测试
`test_chunked_equals_per_frame` 用随机 23 帧数据逐位比较。

### 确认与撤销

- 确认时记录当前束的副本（只含更早的确认限制）、确认帧号、确认文本及其精确 CTC
  前向概率（blank / 非 blank 拆分）。分数从不重新归一化。
- 撤销时从记录的副本出发，用记录的前缀限制重放确认之后已消费的帧，得到「若从未做过
  这次确认」在当前帧应有的束。这是撤销正确性的关键：仅移除文字过滤会留下被限制后
  剪枝丢失的候选。
- 撤销后若再次确认，会生成新的记录，历史确认列表也随之更新。

### 并发与迟到结果

- 每个会话有一个单线程 executor，所有推进 / 确认 / 撤销按提交顺序串行执行，
  读取（`GET /api/sessions/{id}`）也在会话锁内取快照，不会看到半更新状态。
  因此重算（撤销后的重放）期间仍可点击推进或再次撤销：这些请求排队在重算之后，
  作用于重算结果，重算的迟到响应不会覆盖它们。
- 每次状态变更使 `revision` 递增；响应携带该值。前端丢弃 `revision` 更小的响应
  （以及不属于当前会话的响应），并在切换会话时递增 `generation`。
- 操作失败（非法前缀、帧越界、会话过期等）返回 4xx，且**不修改**会话；
  前端保留原字幕与进度，仅显示错误信息。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/sessions` | 用 JSON 创建会话（`vocab`、`beam_width`、`frames`） |
| POST | `/api/sessions/upload` | 上传词表与帧文件（multipart） |
| GET | `/api/sessions/{id}` | 读取当前状态 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| POST | `/api/advance` | `{session_id, to_frame}` 推进到帧边界 |
| POST | `/api/peek` | `{session_id, frame}` 回放已推进帧的候选 |
| POST | `/api/confirm` | `{session_id, prefix}` 延长已确认前缀 |
| POST | `/api/undo` | 撤销最近一次确认 |
| GET | `/api/samples`、`/api/samples/{name}` | 内置样例 |

状态中的 `beams[].logp_blank` / `logp_nonblank` 为 `null` 表示 `-∞`（JSON 无 `-Infinity`）。
`prefix_scores` 是确认时记录的末字 blank / 非 blank 路径概率（以及完整前缀分数
`logp` 与该前缀之前的分数 `previous_logp`），之后不再重新归一化或重算。

## 测试

```bash
python3 -m pytest -q tests            # 单元 + HTTP + 并发排序测试（37 项）
python3 scripts/make_samples.py       # 重新生成内置样例
```

可选的浏览器级冒烟测试（需要 jsdom，在服务已启动时执行）：

```bash
mkdir -p /tmp/uitest && npm --prefix /tmp/uitest install jsdom
JSDOM_DIR=/tmp/uitest node tests/ui_smoke.mjs http://127.0.0.1:8000
```

该脚本使用 jsdom 驱动真实页面，覆盖：样例载入、分块推进、确认前缀、撤销后进度保留、
非法确认的错误提示、历史帧只读回放、连续推进到结尾、零概率样例不出现 NaN。

## 内置样例

| 文件 | 覆盖点 |
| --- | --- |
| `samples/sample_repeat.json` | 连续同字、blank 分隔后的重复字符 |
| `samples/sample_blank_heavy.json` | 大量 blank 帧中间夹短音 |
| `samples/sample_sparse_zero.json` | 含 `0` 概率（`-∞`），界面不出现 NaN |

## 已知限制

- 会话保存在单进程内存中，重启即丢失；多副本部署需要外部存储。
- 束搜索为纯 Python 循环，2000 帧 × 束宽 16 约需 1.5 秒，期间通过单线程队列串行处理；
  未做多进程并行。
- 页面只接受 UTF-8 编码的 JSON/文本文件，单文件大小受反向代理默认限制。

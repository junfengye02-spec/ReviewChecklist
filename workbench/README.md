# 招投标文件智能审评工作台

独立的 Vue 3 + Vite 操作工作台，只连接 `tender_review_backend` 的 `/api/v1` 业务接口。

## 本地运行

先在后端目录启动显式本地 Fake/demo 装配：

```powershell
$env:TENDER_REVIEW_ADAPTER_MODE = "fake"
$env:TENDER_REVIEW_ENVIRONMENT = "local"
$env:TENDER_REVIEW_WORKBENCH_DEMO_ENABLED = "true"
python -m uvicorn tender_review.api.main:app --host 127.0.0.1 --port 8000
```

再启动工作台：

```powershell
cd workbench
npm.cmd install
npm.cmd run dev -- --port 5178
```

打开 `http://127.0.0.1:5178`。如后端不是 `8000` 端口，复制 `.env.example` 为 `.env.local` 并修改 `TENDER_REVIEW_BACKEND_URL`。

本地 demo 数据固定为 synthetic/provisional、`claims_allowed=false`，人工标注与独立复核为 `0/4`。demo 复核人只会写入当前进程的 Fake 仓储，不会写入真实业务基线；Finding 的通过以及规则发布/回滚受后端和前端双重门禁。

## 新建审评任务

页面右上角的“上传并新建审评”会先调用 `POST /api/v1/documents` 登记
PDF，再使用返回的文档快照 ID、内容 SHA-256、选定规则版本哈希和模型配置
哈希调用 `POST /api/v1/review-jobs`。请求带幂等键，创建成功后自动切换到
新任务的进度视图。

来源系统、来源文档编号、规则版本、模型配置 ID/哈希和最大尝试次数均会在
页面上明确填写。local Fake/demo 会预填临时模型配置；生产环境必须替换为
已注册的真实模型配置。相同内容换用不同来源编号会被后端内容寻址门禁拒绝，
不会产生重复文档。

创建成功后任务会先显示为 `QUEUED`。要继续执行解析、检索和审评处理，API
与 Worker 必须连接同一份持久化队列和对象存储；当前 local Fake 仅用于页面
和 API 契约演示，API 进程与单独启动的 Fake Worker 不共享内存队列，不应把
它当作异步生产链路。

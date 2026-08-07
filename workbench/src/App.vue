<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import {
  AlertCircle,
  ArchiveRestore,
  Ban,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  Clock3,
  Database,
  FileCheck2,
  FilePlus2,
  FileSearch,
  FileText,
  Fingerprint,
  GitCompareArrows,
  Gauge,
  History,
  LoaderCircle,
  PlayCircle,
  RefreshCcw,
  SearchX,
  ShieldAlert,
  ShieldCheck,
  Square,
  UserRound,
  XCircle,
} from 'lucide-vue-next'
import { tenderApi } from './api/client'

const tabs = [
  { key: 'progress', label: '任务进度', icon: Clock3 },
  { key: 'evidence', label: '证据复核', icon: ClipboardCheck },
  { key: 'annotations', label: '标注数据集', icon: Database },
  { key: 'rules', label: '规则版本', icon: GitCompareArrows },
  { key: 'report', label: '评测报告', icon: BarChart3 },
  { key: 'optimization', label: '优化轨迹', icon: History },
  { key: 'admission', label: '压测准入', icon: Gauge },
]

const activeTab = ref('progress')
const isLoading = ref(true)
const loadError = ref('')
const partialErrors = ref([])
const notice = ref(null)
const actionBusy = ref('')
const index = ref(null)
const reviewJob = ref(null)
const checkpoints = ref([])
const findings = ref([])
const selectedFindingId = ref('')
const decisions = ref({})
const ruleGroups = ref([])
const selectedRuleSetId = ref('')
const selectedRuleVersionId = ref('')
const ruleDiff = ref(null)
const evaluationRun = ref(null)
const evaluationReport = ref(null)
const a4EvaluationRuns = ref([])
const a4EvaluationReport = ref(null)
const optimizationRuns = ref([])
const auditEvents = ref([])
const a7AdmissionReport = ref(null)
const annotationDatasets = ref([])
const selectedAnnotationDatasetId = ref('')
const annotationStatusFilter = ref('ALL')
const reviewerId = ref('')
const reviewReason = ref('')
const rollbackReason = ref('')
const evaluationDatasetId = ref('')
const showCreateTask = ref(false)
const selectedUploadFile = ref(null)
const sourceSystem = ref('tender-workbench')
const sourceDocumentId = ref('')
const selectedCreateRuleVersionId = ref('')
const createModelConfigId = ref('')
const createModelConfigHash = ref('')
const createMaxAttempts = ref(3)
const activeJobId = ref('')
let activeController = null

const activeFinding = computed(() => (
  findings.value.find((item) => item.finding_id === selectedFindingId.value)
  || findings.value[0]
  || null
))
const activeRuleGroup = computed(() => (
  ruleGroups.value.find((group) => group.ruleSetId === selectedRuleSetId.value)
  || ruleGroups.value[0]
  || null
))
const activeRuleVersion = computed(() => (
  activeRuleGroup.value?.versions.find((version) => version.rule_version_id === selectedRuleVersionId.value)
  || activeRuleGroup.value?.versions.at(-1)
  || null
))
const availableRuleVersions = computed(() => ruleGroups.value.flatMap((group) => (
  group.versions.map((version) => ({ ...version, ruleSetId: group.ruleSetId }))
)))
const selectedCreateRuleVersion = computed(() => (
  availableRuleVersions.value.find((version) => version.rule_version_id === selectedCreateRuleVersionId.value)
  || activeRuleVersion.value
  || availableRuleVersions.value.at(-1)
  || null
))
const activeAnnotationDataset = computed(() => (
  annotationDatasets.value.find(
    (item) => item.dataset_version_id === selectedAnnotationDatasetId.value,
  )
  || annotationDatasets.value.at(-1)
  || null
))
const latestA4EvaluationRun = computed(() => a4EvaluationRuns.value[0] || null)
const latestCheckpointMetrics = computed(() => {
  const checkpoint = [...checkpoints.value].reverse().find(
    (item) => checkpointValue(item, 'metrics_source'),
  )
  if (!checkpoint) return null
  return {
    source: checkpointValue(checkpoint, 'metrics_source'),
    retryCount: checkpointValue(checkpoint, 'retry_count', '0'),
    tokenStatus: checkpointValue(checkpoint, 'model_token_status', 'not_collected'),
    promptTokens: checkpointValue(checkpoint, 'prompt_tokens', null),
    completionTokens: checkpointValue(checkpoint, 'completion_tokens', null),
    costStatus: checkpointValue(checkpoint, 'model_cost_status', 'not_collected'),
    nodeDurations: (checkpoint.state?.values || [])
      .filter((item) => item.key.startsWith('node_duration_ms:'))
      .map((item) => ({
        key: item.key,
        node: item.key.split(':').slice(2).join(':'),
        duration: item.value,
      })),
  }
})
const filteredAnnotationSamples = computed(() => {
  const samples = activeAnnotationDataset.value?.samples || []
  if (annotationStatusFilter.value === 'ALL') return samples
  return samples.filter((item) => item.status === annotationStatusFilter.value)
})
const reviewerReady = computed(() => reviewerId.value.trim().length > 0 && reviewReason.value.trim().length > 0)
const canApproveFinding = computed(() => Boolean(
  activeFinding.value
  && activeFinding.value.human_approval_allowed
  && activeFinding.value.provenance?.status === 'verified'
  && activeFinding.value.provenance?.claims_allowed
  && reviewerReady.value,
))
const canPublishRule = computed(() => {
  const version = activeRuleVersion.value
  return Boolean(
    version
    && reviewerId.value.trim()
    && version.status === 'WAITING_APPROVAL'
    && version.evaluation_gate?.status === 'PASSED'
    && !version.evaluation_gate?.provisional
    && version.evaluation_gate?.claims_allowed
    && version.provenance?.status === 'verified'
    && version.provenance?.claims_allowed,
  )
})
const canRollbackRule = computed(() => Boolean(
  activeRuleVersion.value?.status === 'PUBLISHED'
  && activeRuleVersion.value?.provenance?.status === 'verified'
  && activeRuleVersion.value?.provenance?.claims_allowed
  && reviewerId.value.trim()
  && rollbackReason.value.trim(),
))

function display(value, fallback = '-') {
  if (value === null || value === undefined || value === '') return fallback
  return String(value)
}

function checkpointValue(checkpoint, key, fallback = '') {
  const value = checkpoint?.state?.values?.find((item) => item.key === key)?.value
  return display(value, fallback)
}

function shortHash(value) {
  const text = display(value, '')
  if (!text) return '-'
  return text.length > 20 ? `${text.slice(0, 12)}…${text.slice(-6)}` : text
}

function formatTime(value) {
  if (!value) return '-'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(new Date(value))
}

function statusTone(value) {
  const normalized = String(value || '').toUpperCase()
  if (['COMPLETED', 'PASSED', 'PUBLISHED', 'APPROVED', 'SUCCEEDED', 'VERIFIED', 'FROZEN'].includes(normalized)) return 'success'
  if (['FAILED', 'NOT_READY', 'BLOCKED', 'OPTIMIZATION_FAILED', 'REJECTED', 'DEAD', 'CANCELLED', 'CONFLICT'].includes(normalized)) return 'danger'
  if (['NOT_RUN', 'NO_DECISION'].includes(normalized)) return 'muted'
  if (['WAITING_HUMAN', 'WAITING_APPROVAL', 'PROVISIONAL', 'DRAFT', 'PENDING', 'RUNNING', 'PENDING_ANNOTATION', 'PENDING_REVIEW'].includes(normalized)) return 'warning'
  return 'muted'
}

function sourceLabel(value) {
  return {
    real: '真实业务数据',
    provisional: '临时数据',
    synthetic: '合成数据',
    'external-platform': '外部平台数据',
    EXTERNAL_PLATFORM: '外部平台数据',
  }[value] || display(value, '未知来源')
}

function statusLabel(value) {
  return {
    provisional: '临时 / 不可声明',
    verified: '已验证',
    unknown: '未知',
    WAITING_HUMAN: '等待人工复核',
    WAITING_APPROVAL: '等待审批',
    BLOCKED: '已阻断',
    OPTIMIZATION_FAILED: '优化失败',
    COMPLETED: '已完成',
    RUNNING: '运行中',
    DRAFT: '草稿',
    PUBLISHED: '已发布',
    FAILED: '失败',
    CANCELLED: '已取消',
    PROVISIONAL: '临时门禁',
    PASSED: '通过',
    PENDING: '待评测',
    PENDING_DECISION: '待复核',
    REJECTED: '已驳回',
    INSUFFICIENT_EVIDENCE: '证据不足',
    PENDING_ANNOTATION: '待标注',
    PENDING_REVIEW: '待复核',
    CONFLICT: '冲突仲裁',
    VERIFIED: '已独立复核',
    FROZEN: '已冻结',
    NOT_READY: '未就绪 / 已阻断',
    NOT_RUN: '未运行',
    NO_DECISION: '无准入结论',
    KEEP_MYSQL_QUEUE: '保持 MySQL 队列',
    PROPOSE_ROCKETMQ_ADMISSION: '建议进入 RocketMQ 准入评审',
    KEEP_REDIS_OUT: '不引入 Redis',
    PROPOSE_REDIS_ADMISSION: '建议进入 Redis 准入评审',
  }[value] || display(value)
}

function stageLabel(value) {
  return {
    PARSING: '文档解析',
    INDEXING: '索引构建',
    RETRIEVING: '证据检索',
    EXTRACTING: '结构化抽取',
    COMPARING: '规则比对',
    VERIFYING: '证据核验',
    REPORTING: '报告生成',
  }[value] || statusLabel(value)
}

function gateLabel(value) {
  if (value === true) return '通过'
  if (value === false) return '未通过'
  return '未运行'
}

function gateTone(value) {
  if (value === true) return 'success'
  if (value === false) return 'danger'
  return 'muted'
}

function metricValue(metric) {
  if (!metric.collected || metric.status === 'unknown' || metric.value === null) return '未采集'
  return metric.unit ? `${metric.value} ${metric.unit}` : display(metric.value)
}

function decisionCount(findingId) {
  return decisions.value[findingId]?.length || 0
}

function isOptimizationCancellable(job) {
  return ['PENDING', 'RUNNING'].includes(String(job?.status || '').toUpperCase())
}

function abortActiveRequest() {
  activeController?.abort()
  activeController = null
}

async function loadWorkbench() {
  abortActiveRequest()
  const controller = new AbortController()
  activeController = controller
  isLoading.value = true
  loadError.value = ''
  partialErrors.value = []
  notice.value = null
  try {
    const workbench = await tenderApi.workbench(controller.signal)
    index.value = workbench
    const jobs = workbench.review_job_ids || []
    const ruleSets = workbench.rule_set_ids || []
    const runIds = workbench.evaluation_run_ids || []
    const optimizationIds = workbench.optimization_job_ids || []
    const results = await Promise.allSettled([
      loadTaskBundle(activeJobId.value || jobs[0], controller.signal),
      loadRuleBundle(ruleSets, controller.signal),
      loadEvaluationBundle(runIds[0], controller.signal),
      loadOptimizationBundle(optimizationIds, controller.signal),
      loadAnnotationDatasets(controller.signal),
      tenderApi.a4EvaluationRuns(controller.signal).then(async (value) => {
        a4EvaluationRuns.value = value
        a4EvaluationReport.value = value[0]
          ? await tenderApi.a4EvaluationReport(value[0].run_id, controller.signal)
          : null
      }),
      tenderApi.auditEvents(controller.signal).then((value) => { auditEvents.value = value }),
      tenderApi.a7AdmissionReport(controller.signal).then((value) => { a7AdmissionReport.value = value }),
    ])
    partialErrors.value = results
      .filter((result) => result.status === 'rejected' && result.reason?.name !== 'AbortError')
      .map((result) => result.reason?.message || '部分数据加载失败')
    if (workbench.demo_mode && !reviewerId.value) reviewerId.value = 'demo-reviewer'
  } catch (error) {
    if (error.name !== 'AbortError') loadError.value = error.message || '工作台加载失败'
  } finally {
    if (activeController === controller) {
      isLoading.value = false
      activeController = null
    }
  }
}

async function loadTaskBundle(jobId, signal) {
  reviewJob.value = null
  checkpoints.value = []
  findings.value = []
  decisions.value = {}
  selectedFindingId.value = ''
  if (!jobId) return
  const [job, checkpointItems, findingItems] = await Promise.all([
    tenderApi.reviewJob(jobId, signal),
    tenderApi.checkpoints(jobId, signal),
    tenderApi.findings(jobId, signal),
  ])
  reviewJob.value = job
  checkpoints.value = [...checkpointItems].sort((a, b) => a.sequence - b.sequence)
  findings.value = findingItems
  selectedFindingId.value = findingItems[0]?.finding_id || ''
  const decisionPairs = await Promise.all(
    findingItems.map(async (finding) => [
      finding.finding_id,
      await tenderApi.findingDecisions(finding.finding_id, signal),
    ]),
  )
  decisions.value = Object.fromEntries(decisionPairs)
  evaluationDatasetId.value = findingItems[0]?.provenance?.dataset_version_id || ''
}

async function loadRuleBundle(ruleSetIds, signal) {
  const groups = await Promise.all(
    ruleSetIds.map(async (ruleSetId) => ({
      ruleSetId,
      versions: await tenderApi.ruleVersions(ruleSetId, signal),
    })),
  )
  ruleGroups.value = groups
  selectedRuleSetId.value = groups[0]?.ruleSetId || ''
  selectedRuleVersionId.value = groups[0]?.versions.at(-1)?.rule_version_id || ''
  await loadActiveDiff(signal)
}

async function loadActiveDiff(signal) {
  ruleDiff.value = null
  const group = activeRuleGroup.value
  const version = activeRuleVersion.value
  if (!group || !version) return
  const against = version.parent_version_id
    || group.versions.find((item) => item.rule_version_id !== version.rule_version_id)?.rule_version_id
  if (!against) return
  ruleDiff.value = await tenderApi.ruleDiff(version.rule_version_id, against, signal)
}

async function selectRuleSet(ruleSetId) {
  selectedRuleSetId.value = ruleSetId
  selectedRuleVersionId.value = activeRuleGroup.value?.versions.at(-1)?.rule_version_id || ''
  await runUiAction('rule-diff', () => loadActiveDiff(), '')
}

async function selectRuleVersion(versionId) {
  selectedRuleVersionId.value = versionId
  await runUiAction('rule-diff', () => loadActiveDiff(), '')
}

async function loadEvaluationBundle(runId, signal) {
  evaluationRun.value = null
  evaluationReport.value = null
  if (!runId) return
  ;[evaluationRun.value, evaluationReport.value] = await Promise.all([
    tenderApi.evaluationRun(runId, signal),
    tenderApi.evaluationReport(runId, signal),
  ])
}

async function loadOptimizationBundle(jobIds, signal) {
  optimizationRuns.value = await Promise.all(
    jobIds.map(async (jobId) => {
      const [job, attempts] = await Promise.all([
        tenderApi.optimizationJob(jobId, signal),
        tenderApi.optimizationAttempts(jobId, signal),
      ])
      return { job, attempts }
    }),
  )
}

async function loadAnnotationDatasets(signal) {
  annotationDatasets.value = await tenderApi.annotationDatasets(signal)
  selectedAnnotationDatasetId.value = annotationDatasets.value.at(-1)?.dataset_version_id || ''
}

async function runUiAction(key, action, successText) {
  actionBusy.value = key
  notice.value = null
  try {
    const result = await action()
    if (successText) notice.value = { tone: 'success', text: successText }
    return result
  } catch (error) {
    notice.value = {
      tone: 'danger',
      text: `${error.message || '操作失败'}${error.requestId ? ` · 请求 ${error.requestId}` : ''}`,
    }
    return null
  } finally {
    actionBusy.value = ''
  }
}

async function submitDecision(decision) {
  const finding = activeFinding.value
  if (!finding || (decision === 'APPROVE' && !canApproveFinding.value)) return
  if (!reviewerReady.value) {
    notice.value = { tone: 'danger', text: '请填写具名复核人和复核理由。' }
    return
  }
  const result = await runUiAction(
    `finding-${decision}`,
    () => tenderApi.submitFindingDecision(finding.finding_id, {
      reviewer_kind: 'human',
      reviewer_id: reviewerId.value.trim(),
      decision,
      reason: reviewReason.value.trim(),
    }),
    index.value?.demo_mode
      ? '本地 demo 决定已写入当前 Fake 审计链，不会进入真实业务基线。'
      : '人工复核决定已写入不可变审计链。',
  )
  if (!result) return
  const position = findings.value.findIndex((item) => item.finding_id === finding.finding_id)
  findings.value.splice(position, 1, result.finding)
  decisions.value = {
    ...decisions.value,
    [finding.finding_id]: [...(decisions.value[finding.finding_id] || []), result.decision],
  }
  reviewReason.value = ''
  auditEvents.value = await tenderApi.auditEvents().catch(() => auditEvents.value)
}

async function evaluateRule() {
  const version = activeRuleVersion.value
  const datasetId = evaluationDatasetId.value.trim()
  if (!version || !datasetId) {
    notice.value = { tone: 'danger', text: '发起评测前必须填写数据集版本 ID。' }
    return
  }
  const updated = await runUiAction(
    'rule-evaluate',
    () => tenderApi.evaluateRule(version.rule_version_id, datasetId),
    '评测请求已创建；发布仍需等待可信门禁完成。',
  )
  if (updated) replaceActiveRuleVersion(updated)
}

function replaceActiveRuleVersion(updated) {
  const versions = activeRuleGroup.value?.versions
  if (!versions) return
  const position = versions.findIndex((item) => item.rule_version_id === updated.rule_version_id)
  if (position >= 0) versions.splice(position, 1, updated)
}

async function publishRule() {
  const version = activeRuleVersion.value
  if (!version || !canPublishRule.value) return
  const updated = await runUiAction(
    'rule-publish',
    () => tenderApi.publishRule(version.rule_version_id, reviewerId.value.trim()),
    '规则版本已发布。',
  )
  if (updated) replaceActiveRuleVersion(updated)
}

async function rollbackRule() {
  const group = activeRuleGroup.value
  const version = activeRuleVersion.value
  if (!group || !version || !canRollbackRule.value) return
  const updated = await runUiAction(
    'rule-rollback',
    () => tenderApi.rollbackRule(group.ruleSetId, {
      target_version_id: version.rule_version_id,
      approver_id: reviewerId.value.trim(),
      reason: rollbackReason.value.trim(),
    }),
    '规则集已回滚到选定版本。',
  )
  if (updated) replaceActiveRuleVersion(updated)
}

async function cancelOptimization(run) {
  if (!isOptimizationCancellable(run.job)) return
  const updated = await runUiAction(
    `optimization-${run.job.optimization_job_id}`,
    () => tenderApi.cancelOptimization(run.job.optimization_job_id),
    '优化作业已取消。',
  )
  if (updated) run.job = updated
}

async function sha256Hex(value) {
  if (!globalThis.crypto?.subtle) throw new Error('当前浏览器不支持配置哈希计算')
  const bytes = new TextEncoder().encode(value)
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

async function openCreateTask() {
  showCreateTask.value = true
  const version = selectedCreateRuleVersion.value || activeRuleVersion.value
  selectedCreateRuleVersionId.value = version?.rule_version_id || ''
  if (!createModelConfigId.value && index.value?.demo_mode) {
    createModelConfigId.value = 'synthetic-local-demo'
  }
  if (!createModelConfigHash.value && index.value?.demo_mode) {
    createModelConfigHash.value = await sha256Hex(JSON.stringify({
      id: 'synthetic-local-demo',
      source: 'local-demo',
      version: 1,
    }))
  }
}

function handleUploadFile(event) {
  const [file] = event.target.files || []
  selectedUploadFile.value = file || null
  if (file && !sourceDocumentId.value) {
    sourceDocumentId.value = file.name.replace(/\.pdf$/i, '')
  }
}

function resetCreateTask() {
  showCreateTask.value = false
  selectedUploadFile.value = null
  sourceDocumentId.value = ''
}

async function createReviewTask() {
  const version = selectedCreateRuleVersion.value
  if (!selectedUploadFile.value || !sourceSystem.value.trim() || !sourceDocumentId.value.trim()) {
    notice.value = { tone: 'danger', text: '请先选择 PDF，并填写来源系统和文档编号。' }
    return
  }
  if (!version?.rule_version_id || !version.content_sha256) {
    notice.value = { tone: 'danger', text: '当前没有可用的规则版本哈希，无法创建任务。' }
    return
  }
  if (!createModelConfigId.value.trim() || !/^[0-9a-f]{64}$/i.test(createModelConfigHash.value.trim())) {
    notice.value = { tone: 'danger', text: '请填写已注册模型配置 ID 和 64 位 SHA-256 哈希。' }
    return
  }

  actionBusy.value = 'create-task'
  notice.value = null
  try {
    const controller = new AbortController()
    const snapshot = await tenderApi.uploadDocument(
      selectedUploadFile.value,
      sourceSystem.value.trim(),
      sourceDocumentId.value.trim(),
      controller.signal,
    )
    const job = await tenderApi.createReviewJob(
      {
        schema_version: 1,
        document_snapshot_id: snapshot.id,
        document_sha256: snapshot.sha256,
        rule_version_id: version.rule_version_id,
        rule_version_hash: version.content_sha256,
        model_config_id: createModelConfigId.value.trim(),
        model_config_hash: createModelConfigHash.value.trim().toLowerCase(),
        max_attempts: Number(createMaxAttempts.value) || 3,
      },
      {
        idempotencyKey: globalThis.crypto?.randomUUID?.() || `tender-job-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        callerId: sourceSystem.value.trim(),
      },
      controller.signal,
    )
    activeJobId.value = job.id
    activeTab.value = 'progress'
    await loadTaskBundle(job.id, controller.signal)
    showCreateTask.value = false
    notice.value = {
      tone: 'success',
      text: `审评任务 ${job.id} 已创建，文件已登记并进入队列。`,
    }
    selectedUploadFile.value = null
  } catch (error) {
    notice.value = {
      tone: 'danger',
      text: `${error.message || '创建审评任务失败'}${error.requestId ? ` · 请求 ${error.requestId}` : ''}`,
    }
  } finally {
    actionBusy.value = ''
  }
}

onMounted(loadWorkbench)
onUnmounted(abortActiveRequest)
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <span class="brand-mark" aria-hidden="true"><FileCheck2 :size="22" /></span>
        <div>
          <h1>招投标文件智能审评平台</h1>
          <p>证据驱动审评与规则回归工作台</p>
        </div>
      </div>
      <div class="topbar-actions">
        <span v-if="index" class="source-tag">{{ sourceLabel(index.source_type) }}</span>
        <span v-if="index" class="status-tag" :class="statusTone(index.status)">{{ statusLabel(index.status) }}</span>
        <button class="button primary" type="button" :disabled="isLoading || actionBusy !== ''" @click="openCreateTask">
          <FilePlus2 :size="16" />
          <span>上传并新建审评</span>
        </button>
        <button class="button secondary" type="button" :disabled="isLoading" aria-label="刷新工作台" @click="loadWorkbench">
          <RefreshCcw :size="16" :class="{ spinning: isLoading }" />
          <span>刷新</span>
        </button>
      </div>
    </header>

    <div v-if="index?.demo_mode" class="demo-boundary" role="status">
      <ShieldAlert :size="18" />
      <div>
        <strong>本地演示边界 · demo reviewer</strong>
        <span>人工标注与独立复核 {{ index.human_annotation_cases }}/{{ index.required_human_cases }}；数据为 synthetic/provisional，claims_allowed=false。复核决定只写入当前 Fake 进程，不进入真实业务基线。</span>
      </div>
    </div>

    <section v-if="showCreateTask" class="create-task-panel" aria-label="上传招标文件并创建审评任务">
      <div class="create-task-heading">
        <div>
          <span class="eyebrow">新建任务</span>
          <h2>上传招标文件并创建审评任务</h2>
          <p>文件先登记为不可变文档，再使用选定规则和模型配置创建带幂等键的任务。</p>
        </div>
        <button class="icon-button" type="button" aria-label="关闭新建任务" title="关闭" @click="resetCreateTask">×</button>
      </div>
      <form class="create-task-grid" @submit.prevent="createReviewTask">
        <label class="file-field">
          <span>招标文件（PDF）</span>
          <input type="file" accept="application/pdf,.pdf" @change="handleUploadFile" />
          <small>{{ selectedUploadFile?.name || '请选择 PDF 文件' }}</small>
        </label>
        <label>
          <span>来源系统</span>
          <input v-model="sourceSystem" type="text" maxlength="64" required />
        </label>
        <label>
          <span>来源文档编号</span>
          <input v-model="sourceDocumentId" type="text" maxlength="255" required placeholder="例如 tender-2026-001" />
        </label>
        <label>
          <span>规则版本</span>
          <select v-model="selectedCreateRuleVersionId" required>
            <option v-for="version in availableRuleVersions" :key="version.rule_version_id" :value="version.rule_version_id">
              {{ version.rule_version_id }} · {{ statusLabel(version.status) }}
            </option>
          </select>
          <small v-if="selectedCreateRuleVersion">规则哈希：{{ shortHash(selectedCreateRuleVersion.content_sha256) }}</small>
        </label>
        <label>
          <span>模型配置 ID</span>
          <input v-model="createModelConfigId" type="text" maxlength="128" required placeholder="已注册的 model_config_id" />
        </label>
        <label>
          <span>模型配置 SHA-256</span>
          <input v-model="createModelConfigHash" type="text" minlength="64" maxlength="64" required placeholder="64 位十六进制哈希" />
        </label>
        <label>
          <span>最大尝试次数</span>
          <input v-model.number="createMaxAttempts" type="number" min="1" max="100" required />
        </label>
        <div class="create-task-actions">
          <small v-if="index?.demo_mode">当前为 local Fake/demo；新任务会进入本地队列，不代表生产运行。</small>
          <button class="button secondary" type="button" :disabled="actionBusy !== ''" @click="resetCreateTask">取消</button>
          <button class="button primary" type="submit" :disabled="actionBusy !== '' || !selectedUploadFile">{{ actionBusy === 'create-task' ? '创建中…' : '创建审评任务' }}</button>
        </div>
      </form>
    </section>

    <div v-if="loadError" class="message danger" role="alert">
      <AlertCircle :size="17" />
      <span>{{ loadError }}</span>
      <button type="button" @click="loadWorkbench">重试</button>
    </div>
    <div v-else-if="partialErrors.length" class="message warning" role="status">
      <AlertCircle :size="17" />
      <span>{{ partialErrors.join('；') }}</span>
    </div>
    <div v-if="notice" class="message" :class="notice.tone" role="status">
      <CheckCircle2 v-if="notice.tone === 'success'" :size="17" />
      <AlertCircle v-else :size="17" />
      <span>{{ notice.text }}</span>
    </div>

    <nav class="tabs" aria-label="标书审评工作台视图" role="tablist">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        role="tab"
        :aria-selected="activeTab === tab.key"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        <component :is="tab.icon" :size="16" />
        <span>{{ tab.label }}</span>
      </button>
    </nav>

    <main class="workspace">
      <div v-if="isLoading" class="loading-state" aria-live="polite">
        <LoaderCircle :size="24" class="spinning" />
        <span>正在加载可追溯审评数据</span>
      </div>

      <template v-else-if="index">
        <section v-if="activeTab === 'progress'" class="view" role="tabpanel" aria-labelledby="progress-title">
          <div class="view-heading">
            <div>
              <h2 id="progress-title">任务进度</h2>
              <p v-if="reviewJob">任务 {{ reviewJob.id }} · 输入 {{ shortHash(reviewJob.input_fingerprint) }}</p>
            </div>
            <span v-if="reviewJob" class="status-tag" :class="statusTone(reviewJob.status)">{{ statusLabel(reviewJob.status) }}</span>
          </div>

          <div v-if="reviewJob" class="summary-strip">
            <div><span>当前阶段</span><strong>{{ stageLabel(reviewJob.stage) }}</strong></div>
            <div><span>尝试次数</span><strong>{{ reviewJob.attempt_count }} / {{ reviewJob.max_attempts }}</strong></div>
            <div><span>规则版本</span><strong :title="reviewJob.rule_version_id">{{ reviewJob.rule_version_id }}</strong></div>
            <div><span>待复核 Finding</span><strong>{{ findings.length }}</strong></div>
          </div>

          <div v-if="reviewJob?.safe_failure_code" class="gate-warning" role="status">
            <AlertCircle :size="15" />
            <span><strong>{{ reviewJob.safe_failure_code }}</strong> · {{ reviewJob.safe_failure_category }} · {{ reviewJob.safe_failure_retryable ? '可重试' : '不可重试' }}</span>
          </div>

          <div v-if="reviewJob" class="operations-strip" aria-label="任务恢复与观测摘要">
            <div><span>恢复次数</span><strong>{{ reviewJob.recovery_count }}</strong><small>{{ reviewJob.recovery_metric_source }}</small></div>
            <div><span>调用重试</span><strong>{{ latestCheckpointMetrics?.retryCount || 0 }}</strong><small>{{ latestCheckpointMetrics?.source || '未采集' }}</small></div>
            <div><span>模型 Token</span><strong>{{ latestCheckpointMetrics?.promptTokens !== null ? `${latestCheckpointMetrics.promptTokens} / ${latestCheckpointMetrics.completionTokens}` : '未采集' }}</strong><small>{{ latestCheckpointMetrics?.tokenStatus || 'not_collected' }}</small></div>
            <div><span>模型成本</span><strong>{{ latestCheckpointMetrics?.costStatus || '未采集' }}</strong><small>pricing_configuration</small></div>
          </div>

          <div v-if="checkpoints.length" class="progress-list">
            <article v-for="checkpoint in checkpoints" :key="checkpoint.node_name" class="progress-row">
              <span class="progress-index"><Check :size="13" /></span>
              <div><strong>{{ stageLabel(checkpoint.stage) }}</strong><span>{{ checkpoint.node_name }} · checkpoint {{ checkpoint.sequence }}</span><small v-if="checkpointValue(checkpoint, 'langgraph_checkpoint_id')">{{ shortHash(checkpointValue(checkpoint, 'langgraph_checkpoint_id')) }}</small></div>
              <code :title="checkpoint.output_artifact_id">{{ checkpoint.output_artifact_id }}</code>
              <time>{{ formatTime(checkpoint.completed_at) }}</time>
            </article>
          </div>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有可显示的任务 checkpoint</span></div>

          <div v-if="latestCheckpointMetrics?.nodeDurations.length" class="node-metrics" aria-label="节点耗时">
            <div v-for="item in latestCheckpointMetrics.nodeDurations" :key="item.key"><span>{{ item.node }}</span><strong>{{ item.duration }} ms</strong></div>
          </div>

          <section class="subsection">
            <div class="view-heading compact"><div><h3>最近审计事件</h3><p>仅展示身份、动作、资源与追踪标识。</p></div></div>
            <div v-if="auditEvents.length" class="table-scroll audit-table">
              <table>
                <thead><tr><th>时间</th><th>身份</th><th>动作</th><th>资源</th><th>结果</th><th>请求</th></tr></thead>
                <tbody>
                  <tr v-for="event in auditEvents" :key="event.event_id">
                    <td>{{ formatTime(event.occurred_at) }}</td>
                    <td><span class="actor-kind">{{ event.actor.kind }}</span> {{ event.actor.actor_id }}</td>
                    <td>{{ event.action }}</td>
                    <td>{{ event.resource.resource_type }} / {{ event.resource.resource_id }}</td>
                    <td><span class="status-tag" :class="statusTone(event.result)">{{ statusLabel(event.result) }}</span></td>
                    <td><code>{{ event.job_id || event.request_id }}</code><small v-if="event.checkpoint_id">checkpoint {{ shortHash(event.checkpoint_id) }}</small></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="inline-empty">当前没有可查询的审计事件。</div>
          </section>
        </section>

        <section v-else-if="activeTab === 'annotations'" class="view" role="tabpanel" aria-labelledby="annotations-title">
          <div class="view-heading">
            <div>
              <h2 id="annotations-title">人工标注数据集</h2>
              <p v-if="activeAnnotationDataset">{{ activeAnnotationDataset.dataset_name }} · v{{ activeAnnotationDataset.version_number }} · {{ shortHash(activeAnnotationDataset.manifest_sha256) }}</p>
            </div>
            <div class="tag-line">
              <span v-if="activeAnnotationDataset" class="status-tag" :class="statusTone(activeAnnotationDataset.status)">{{ statusLabel(activeAnnotationDataset.status) }}</span>
              <span v-if="activeAnnotationDataset" class="status-tag warning">claims_allowed=false</span>
            </div>
          </div>

          <div v-if="activeAnnotationDataset" class="summary-strip">
            <div><span>样本总数</span><strong>{{ activeAnnotationDataset.samples.length }}</strong></div>
            <div><span>待标注</span><strong>{{ activeAnnotationDataset.samples.filter((item) => item.status === 'PENDING_ANNOTATION').length }}</strong></div>
            <div><span>待复核</span><strong>{{ activeAnnotationDataset.samples.filter((item) => item.status === 'PENDING_REVIEW').length }}</strong></div>
            <div><span>冲突仲裁</span><strong>{{ activeAnnotationDataset.samples.filter((item) => item.status === 'CONFLICT').length }}</strong></div>
          </div>

          <div class="evaluation-toolbar">
            <label v-if="annotationDatasets.length > 1">
              <span>数据集版本</span>
              <select v-model="selectedAnnotationDatasetId">
                <option v-for="dataset in annotationDatasets" :key="dataset.dataset_version_id" :value="dataset.dataset_version_id">{{ dataset.dataset_name }} · v{{ dataset.version_number }}</option>
              </select>
            </label>
            <label>
              <span>样本状态</span>
              <select v-model="annotationStatusFilter">
                <option value="ALL">全部状态</option>
                <option value="PENDING_ANNOTATION">待标注</option>
                <option value="PENDING_REVIEW">待复核</option>
                <option value="CONFLICT">冲突仲裁</option>
                <option value="VERIFIED">已独立复核</option>
                <option value="FROZEN">已冻结</option>
              </select>
            </label>
          </div>

          <div v-if="filteredAnnotationSamples.length" class="table-scroll">
            <table>
              <thead><tr><th>样本</th><th>问题标签</th><th>集合</th><th>状态</th><th>文档哈希</th><th>标注 / 复核 / 仲裁</th></tr></thead>
              <tbody>
                <tr v-for="sample in filteredAnnotationSamples" :key="sample.sample_id">
                  <td><strong>{{ sample.sample_id }}</strong><small>{{ sample.query_id }}</small></td>
                  <td>{{ sample.question_label }}</td>
                  <td>{{ sample.split }}</td>
                  <td><span class="status-tag" :class="statusTone(sample.status)">{{ statusLabel(sample.status) }}</span></td>
                  <td><code :title="sample.document_sha256">{{ shortHash(sample.document_sha256) }}</code></td>
                  <td>{{ sample.annotation?.actor_id || '-' }} / {{ sample.review?.actor_id || '-' }} / {{ sample.adjudication?.actor_id || '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="empty-state"><SearchX :size="24" /><span>当前筛选没有样本</span></div>
        </section>

        <section v-else-if="activeTab === 'evidence'" class="view" role="tabpanel" aria-labelledby="evidence-title">
          <div class="view-heading">
            <div><h2 id="evidence-title">证据复核</h2><p>核对招标文件、页码、章节、连续原文与内容哈希</p></div>
            <label v-if="findings.length > 1" class="compact-field"><span>Finding</span><select v-model="selectedFindingId"><option v-for="finding in findings" :key="finding.finding_id" :value="finding.finding_id">{{ finding.finding_id }}</option></select></label>
          </div>

          <div v-if="activeFinding" class="evidence-layout">
            <div class="evidence-main">
              <div class="finding-heading">
                <div><span class="eyebrow">Finding {{ activeFinding.finding_id }}</span><h3>{{ activeFinding.message }}</h3></div>
                <div class="tag-line"><span class="source-tag">{{ sourceLabel(activeFinding.provenance.status) }}</span><span class="status-tag warning">claims_allowed=false</span></div>
              </div>
              <article v-for="evidence in activeFinding.evidence" :key="evidence.chunk_id" class="evidence-record">
                <header><FileText :size="17" /><div><strong>{{ evidence.document_id }}</strong><span>第 {{ evidence.page_number }} 页<span v-if="evidence.page_end && evidence.page_end !== evidence.page_number">–{{ evidence.page_end }}</span> · {{ evidence.section_path.join(' / ') }}</span></div></header>
                <blockquote>{{ evidence.excerpt }}</blockquote>
                <div class="hash-line"><Fingerprint :size="14" /><code :title="evidence.text_sha256">{{ evidence.text_sha256 }}</code></div>
              </article>
              <div class="provenance-grid">
                <div><span>数据集</span><code>{{ activeFinding.provenance.dataset_version_id }}</code></div>
                <div><span>检索方案</span><code>{{ activeFinding.provenance.retrieval_variant }}</code></div>
                <div><span>输入哈希</span><code :title="activeFinding.provenance.review_input_sha256">{{ activeFinding.provenance.review_input_sha256 }}</code></div>
                <div><span>结果哈希</span><code :title="activeFinding.provenance.retrieval_results_sha256">{{ activeFinding.provenance.retrieval_results_sha256 }}</code></div>
              </div>
            </div>

            <aside class="review-panel" aria-label="人工复核操作">
              <div><span class="eyebrow">人工决定</span><h3>{{ index.demo_mode ? 'demo reviewer（本地 Fake）' : '具名业务复核' }}</h3><p>已记录 {{ decisionCount(activeFinding.finding_id) }} 条不可变决定</p></div>
              <label><span><UserRound :size="14" />{{ index.demo_mode ? '演示复核人' : '复核人' }}</span><input v-model="reviewerId" type="text" autocomplete="name" required /></label>
              <label><span><FileSearch :size="14" />复核理由</span><textarea v-model="reviewReason" rows="5" required placeholder="说明驳回或证据不足的理由"></textarea></label>
              <div v-if="!activeFinding.human_approval_allowed || activeFinding.provenance.status !== 'verified'" class="gate-warning"><Ban :size="16" /><span>provisional Finding 不允许记录为人工批准；后端门禁同样会拒绝。</span></div>
              <div class="review-actions">
                <button class="button primary" type="button" :disabled="!canApproveFinding || actionBusy !== ''" title="仅 verified 且允许人工批准的 Finding 可通过" @click="submitDecision('APPROVE')"><CheckCircle2 :size="15" />通过</button>
                <button class="button danger" type="button" :disabled="!reviewerReady || actionBusy !== ''" @click="submitDecision('REJECT')"><XCircle :size="15" />驳回</button>
                <button class="button secondary wide" type="button" :disabled="!reviewerReady || actionBusy !== ''" @click="submitDecision('INSUFFICIENT_EVIDENCE')"><SearchX :size="15" />证据不足</button>
              </div>
            </aside>
          </div>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有待复核 Finding</span></div>
        </section>

        <section v-else-if="activeTab === 'rules'" class="view" role="tabpanel" aria-labelledby="rules-title">
          <div class="view-heading"><div><h2 id="rules-title">规则版本</h2><p>版本不可变，评测、发布和回滚受可信数据门禁约束</p></div></div>
          <template v-if="ruleGroups.length">
            <div class="rule-toolbar">
              <label><span>规则集</span><select :value="selectedRuleSetId" @change="selectRuleSet($event.target.value)"><option v-for="group in ruleGroups" :key="group.ruleSetId" :value="group.ruleSetId">{{ group.ruleSetId }}</option></select></label>
              <label><span>版本</span><select :value="activeRuleVersion?.rule_version_id" @change="selectRuleVersion($event.target.value)"><option v-for="version in activeRuleGroup?.versions || []" :key="version.rule_version_id" :value="version.rule_version_id">v{{ version.version_number }} · {{ version.rule_version_id }}</option></select></label>
              <div class="tag-line"><span class="status-tag" :class="statusTone(activeRuleVersion?.status)">{{ statusLabel(activeRuleVersion?.status) }}</span><span class="source-tag">{{ sourceLabel(activeRuleVersion?.provenance.status) }}</span></div>
            </div>

            <div class="rule-grid">
              <div class="rule-main">
                <div class="evaluation-toolbar">
                  <label><span>评测数据集版本</span><input v-model="evaluationDatasetId" type="text" placeholder="dataset_version_id" /></label>
                  <button class="button secondary" type="button" :disabled="!activeRuleVersion || !evaluationDatasetId.trim() || actionBusy !== ''" @click="evaluateRule"><PlayCircle :size="15" />发起评测</button>
                </div>
                <div class="table-scroll rule-history">
                  <table>
                    <thead><tr><th>版本</th><th>状态</th><th>门禁</th><th>来源</th><th>声明</th><th>创建时间</th></tr></thead>
                    <tbody><tr v-for="version in activeRuleGroup?.versions || []" :key="version.rule_version_id" :class="{ selected: version.rule_version_id === activeRuleVersion?.rule_version_id }" tabindex="0" @click="selectRuleVersion(version.rule_version_id)" @keyup.enter="selectRuleVersion(version.rule_version_id)"><td><strong>v{{ version.version_number }}</strong><small>{{ version.rule_version_id }}</small></td><td><span class="status-tag" :class="statusTone(version.status)">{{ statusLabel(version.status) }}</span></td><td>{{ statusLabel(version.evaluation_gate?.status || 'PENDING') }}</td><td>{{ sourceLabel(version.provenance.status) }}</td><td>{{ version.provenance.claims_allowed ? '允许' : '不允许' }}</td><td>{{ formatTime(version.created_at) }}</td></tr></tbody>
                  </table>
                </div>
              </div>

              <aside class="rule-panel">
                <div><span class="eyebrow">发布门禁</span><h3>{{ activeRuleVersion?.rule_version_id }}</h3></div>
                <ul class="gate-list">
                  <li><span :class="['gate-dot', gateTone(activeRuleVersion?.status === 'WAITING_APPROVAL')]"></span><div><strong>审批状态</strong><span>{{ activeRuleVersion?.status === 'WAITING_APPROVAL' ? '已到审批边界' : '未到审批边界' }}</span></div></li>
                  <li><span :class="['gate-dot', gateTone(activeRuleVersion?.evaluation_gate?.status === 'PASSED')]"></span><div><strong>评测门禁</strong><span>{{ statusLabel(activeRuleVersion?.evaluation_gate?.status || 'PENDING') }}</span></div></li>
                  <li><span :class="['gate-dot', gateTone(activeRuleVersion?.provenance.status === 'verified')]"></span><div><strong>数据来源</strong><span>{{ sourceLabel(activeRuleVersion?.provenance.status) }}</span></div></li>
                  <li><span :class="['gate-dot', gateTone(activeRuleVersion?.provenance.claims_allowed)]"></span><div><strong>可声明性</strong><span>{{ activeRuleVersion?.provenance.claims_allowed ? '允许' : '不允许' }}</span></div></li>
                </ul>
                <label><span>审批人</span><input v-model="reviewerId" type="text" /></label>
                <label><span>回滚理由</span><textarea v-model="rollbackReason" rows="3" placeholder="回滚时必填"></textarea></label>
                <div class="rule-actions"><button class="button primary" type="button" :disabled="!canPublishRule || actionBusy !== ''" title="仅非 provisional 的可信门禁可发布" @click="publishRule"><ShieldCheck :size="15" />发布</button><button class="button secondary" type="button" :disabled="!canRollbackRule || actionBusy !== ''" title="provisional 规则不可回滚" @click="rollbackRule"><ArchiveRestore :size="15" />回滚</button></div>
              </aside>
            </div>

            <section class="subsection">
              <div class="view-heading compact"><div><h3>版本差异</h3><p v-if="ruleDiff">{{ ruleDiff.from_version_id }} → {{ ruleDiff.to_version_id }}</p></div><span v-if="ruleDiff" class="status-tag muted">{{ ruleDiff.changes.length }} 处变更</span></div>
              <div v-if="ruleDiff?.changes.length" class="diff-scroll"><div class="diff-list"><div v-for="change in ruleDiff.changes" :key="`${change.path}-${change.operation}`" class="diff-row"><span class="diff-op">{{ change.operation }}</span><code>{{ change.path }}</code><div><small>之前</small><pre>{{ change.before_json || '∅' }}</pre></div><ChevronRight :size="16" /><div><small>之后</small><pre>{{ change.after_json || '∅' }}</pre></div></div></div></div>
              <div v-else class="inline-empty">该版本没有可显示的差异。</div>
            </section>
          </template>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有规则版本数据</span></div>
        </section>

        <section v-else-if="activeTab === 'report'" class="view" role="tabpanel" aria-labelledby="report-title">
          <div class="view-heading">
            <div><h2 id="report-title">评测报告</h2><p v-if="evaluationRun">{{ evaluationRun.name }} · {{ evaluationRun.run_id }}</p></div>
            <div v-if="evaluationReport" class="tag-line"><span class="source-tag">{{ sourceLabel(evaluationReport.source_type) }}</span><span class="status-tag" :class="statusTone(evaluationReport.status)">{{ statusLabel(evaluationReport.status) }}</span></div>
          </div>
          <section v-if="latestA4EvaluationRun" class="subsection traceability" aria-label="A4 评测运行状态">
            <div class="view-heading compact"><div><h3>A4 发布门禁运行</h3><p>{{ latestA4EvaluationRun.run_id }} · {{ statusLabel(latestA4EvaluationRun.status) }}</p></div><span class="status-tag" :class="statusTone(latestA4EvaluationRun.status)">{{ latestA4EvaluationRun.claims_allowed ? '可声明' : '不可声明' }}</span></div>
            <div class="hash-grid"><div><span>dataset_version_id</span><code>{{ latestA4EvaluationRun.binding.dataset_version_id }}</code></div><div><span>dataset_manifest_sha256</span><code :title="latestA4EvaluationRun.binding.dataset_manifest_sha256">{{ shortHash(latestA4EvaluationRun.binding.dataset_manifest_sha256) }}</code></div><div><span>binding_sha256</span><code :title="latestA4EvaluationRun.binding.binding_sha256">{{ shortHash(latestA4EvaluationRun.binding.binding_sha256) }}</code></div><div><span>report_sha256</span><code :title="latestA4EvaluationRun.report_sha256">{{ shortHash(latestA4EvaluationRun.report_sha256) }}</code></div></div>
            <ul v-if="latestA4EvaluationRun.blockers.length" class="limitations"><li v-for="item in latestA4EvaluationRun.blockers" :key="item"><AlertCircle :size="14" />{{ item }}</li></ul>
            <div v-if="a4EvaluationReport?.metric_differences.length" class="table-scroll metric-table"><table><thead><tr><th>指标</th><th>基线</th><th>候选</th><th>差异</th><th>门禁</th></tr></thead><tbody><tr v-for="item in a4EvaluationReport.metric_differences" :key="item.metric_id"><td><strong>{{ item.metric_id }}</strong></td><td>{{ item.baseline_value }}</td><td>{{ item.candidate_value }}</td><td>{{ item.delta }}</td><td><span class="status-tag" :class="statusTone(item.passed ? 'PASSED' : 'FAILED')">{{ item.passed ? '通过' : '未通过' }}</span></td></tr></tbody></table></div>
            <div v-if="a4EvaluationReport?.failure_samples.length" class="table-scroll metric-table"><table><thead><tr><th>失败样本</th><th>阶段</th><th>类别</th><th>说明</th></tr></thead><tbody><tr v-for="item in a4EvaluationReport.failure_samples" :key="item.sample_id"><td><code>{{ item.sample_id }}</code></td><td>{{ item.stage }}</td><td>{{ item.category }}</td><td>{{ item.detail }}</td></tr></tbody></table></div>
            <ul v-if="a4EvaluationReport?.difference_sources.length" class="limitations"><li v-for="item in a4EvaluationReport.difference_sources" :key="`${item.source}-${item.detail}`"><GitCompareArrows :size="14" />{{ item.source }} · {{ item.detail }}</li></ul>
          </section>
          <template v-if="evaluationRun && evaluationReport">
            <div class="report-boundary"><ShieldAlert :size="18" /><div><strong>指标声明边界</strong><span>仅 claims_allowed=true 的 verified/real 指标可声明。本报告人工数据 {{ evaluationReport.human_annotation_cases }}/{{ evaluationReport.required_human_cases }}；未知指标显示“未采集”，不以 0 代替。</span></div></div>
            <div class="report-sections">
              <section v-for="section in evaluationReport.sections" :key="section.section_id" class="metric-section">
                <h3>{{ section.title }}</h3>
                <div class="table-scroll metric-table"><table><thead><tr><th>指标</th><th>值</th><th>状态</th><th>来源</th><th>解释</th></tr></thead><tbody><tr v-for="metric in section.metrics" :key="metric.metric_id"><td><strong>{{ metric.label }}</strong><small>{{ metric.metric_id }}</small></td><td><span :class="{ 'unknown-value': !metric.collected }">{{ metricValue(metric) }}</span></td><td><span class="status-tag" :class="statusTone(metric.status)">{{ metric.claims_allowed ? '可声明' : statusLabel(metric.status) }}</span></td><td>{{ sourceLabel(metric.source_type) }}</td><td>{{ metric.interpretation }}</td></tr></tbody></table></div>
              </section>
            </div>
            <section class="subsection traceability">
              <div class="view-heading compact"><div><h3>运行与 provenance</h3><p>run、输入、结果、配置、代码和报告哈希一一对应</p></div><code :title="evaluationRun.report_sha256">{{ shortHash(evaluationRun.report_sha256) }}</code></div>
              <div class="hash-grid"><div><span>run_id</span><code>{{ evaluationRun.run_id }}</code></div><div><span>dataset_version_id</span><code>{{ evaluationRun.dataset_version_id }}</code></div><div v-for="(value, key) in evaluationRun.hashes" :key="key"><span>{{ key }}</span><code :title="value">{{ value }}</code></div></div>
              <ul class="limitations"><li v-for="item in evaluationReport.limitations" :key="item"><AlertCircle :size="14" />{{ item }}</li></ul>
            </section>
          </template>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有评测运行报告</span></div>
        </section>

        <section v-else-if="activeTab === 'optimization'" class="view" role="tabpanel" aria-labelledby="optimization-title">
          <div class="view-heading"><div><h2 id="optimization-title">优化轨迹</h2><p>有界候选、失败与阻断轨迹，保留三道联合回归门禁和 checkpoint</p></div></div>
          <div v-if="optimizationRuns.length" class="optimization-list">
            <article v-for="run in optimizationRuns" :key="run.job.optimization_job_id" class="optimization-run">
              <header>
                <div><span class="eyebrow">{{ run.job.optimization_job_id }}</span><h3>{{ run.job.base_rule_version_id }} → {{ run.job.candidate_rule_version_id || '无候选版本' }}</h3></div>
                <div class="tag-line"><span class="status-tag" :class="statusTone(run.job.status)">{{ statusLabel(run.job.status) }}</span><span class="source-tag">{{ sourceLabel(run.job.provenance.source_type) }}</span><button class="icon-button" type="button" :disabled="!isOptimizationCancellable(run.job) || actionBusy !== ''" title="取消优化作业" aria-label="取消优化作业" @click="cancelOptimization(run)"><Square :size="15" /></button></div>
              </header>
              <div class="optimization-summary"><div><span>轮次</span><strong>{{ run.job.current_round }} / {{ run.job.max_rounds }}</strong></div><div><span>候选/轮</span><strong>{{ run.job.candidates_per_round }}</strong></div><div><span>稳定性重复</span><strong>{{ run.job.required_stability_runs }}</strong></div><div><span>Checkpoint</span><code :title="run.job.last_checkpoint_sha256">{{ shortHash(run.job.last_checkpoint_sha256) }}</code></div></div>
              <div v-if="run.job.readiness" class="hash-grid"><div><span>A5 readiness</span><strong>{{ statusLabel(run.job.readiness.status) }}</strong></div><div><span>A4 run</span><code>{{ run.job.readiness.a4_evaluation_run_id || '未绑定' }}</code></div><div><span>A4 report</span><code :title="run.job.readiness.a4_report_sha256">{{ shortHash(run.job.readiness.a4_report_sha256) }}</code></div><div><span>Dataset manifest</span><code :title="run.job.readiness.dataset_manifest_sha256">{{ shortHash(run.job.readiness.dataset_manifest_sha256) }}</code></div></div>
              <ul v-if="run.job.readiness?.blockers?.length" class="limitations"><li v-for="item in run.job.readiness.blockers" :key="item"><AlertCircle :size="14" />{{ item }}</li></ul>
              <div class="attempt-list">
                <section v-for="attempt in run.attempts" :key="attempt.attempt_id" class="attempt-row">
                  <div class="attempt-marker"><span>{{ attempt.attempt_number }}</span></div>
                  <div class="attempt-content">
                    <div class="attempt-heading"><div><strong>第 {{ attempt.attempt_number }} 轮 · {{ statusLabel(attempt.status) }}</strong><span>{{ attempt.root_cause?.root_cause || '未分类' }} · {{ attempt.root_cause?.classifier || '-' }}</span></div><code :title="attempt.checkpoint_sha256">{{ shortHash(attempt.checkpoint_sha256) }}</code></div>
                    <p v-if="attempt.root_cause" class="root-rationale">{{ attempt.root_cause.rationale }}</p>
                    <div v-if="attempt.candidates.length" class="table-scroll candidate-table"><table><thead><tr><th>候选</th><th>类型</th><th>最小改动</th><th>目标样本</th></tr></thead><tbody><tr v-for="candidate in attempt.candidates" :key="candidate.candidate_id"><td><strong>{{ candidate.candidate_id }}</strong><small>{{ candidate.rationale }}</small></td><td>{{ candidate.candidate_type }}</td><td><code>{{ candidate.change.scope }} {{ candidate.change.path }}</code></td><td>{{ candidate.target_sample_ids.length }}</td></tr></tbody></table></div>
                    <div v-for="evaluation in attempt.evaluations" :key="evaluation.candidate_id" class="regression-gates">
                      <div><span>目标样本</span><strong :class="gateTone(evaluation.target_gate_passed)">{{ gateLabel(evaluation.target_gate_passed) }}</strong></div><div><span>合规保护</span><strong :class="gateTone(evaluation.protection_gate_passed)">{{ gateLabel(evaluation.protection_gate_passed) }}</strong></div><div><span>稳定性</span><strong :class="gateTone(evaluation.stability_gate_passed)">{{ gateLabel(evaluation.stability_gate_passed) }}</strong></div><div><span>结果</span><strong :class="statusTone(evaluation.status)">{{ statusLabel(evaluation.status) }}</strong></div><div><span>声明</span><strong :class="evaluation.claims_allowed ? 'success' : 'warning'">{{ evaluation.claims_allowed ? '允许' : '不允许' }}</strong></div>
                    </div>
                    <div v-if="attempt.failure" class="attempt-failure"><XCircle :size="15" /><span>{{ attempt.failure.phase }} / {{ attempt.failure.code }}：{{ attempt.failure.message }}</span></div>
                  </div>
                </section>
              </div>
              <div v-if="run.job.failure_trajectory.length" class="failure-trajectory"><strong>失败轨迹</strong><span v-for="failure in run.job.failure_trajectory" :key="`${failure.phase}-${failure.code}`">{{ failure.phase }} · {{ failure.code }} · {{ failure.message }}</span></div>
            </article>
          </div>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有优化作业数据</span></div>
        </section>
        <section v-else class="view" role="tabpanel" aria-labelledby="admission-title">
          <div class="view-heading"><div><h2 id="admission-title">压测与技术准入</h2></div></div>
          <article v-if="a7AdmissionReport" class="optimization-run">
            <header class="optimization-head">
              <div><span class="eyebrow">A7 · {{ a7AdmissionReport.run_id }}</span><h3>{{ statusLabel(a7AdmissionReport.status) }}</h3></div>
              <strong :class="statusTone(a7AdmissionReport.queue_decision)">{{ statusLabel(a7AdmissionReport.queue_decision) }}</strong>
            </header>
            <div class="optimization-summary">
              <div><span>矩阵覆盖</span><strong>{{ a7AdmissionReport.matrix_completed }} / {{ a7AdmissionReport.matrix_expected }}</strong></div>
              <div><span>性能声明</span><strong :class="a7AdmissionReport.claims_allowed ? 'success' : 'warning'">{{ a7AdmissionReport.claims_allowed ? '允许' : '禁止' }}</strong></div>
              <div><span>当前动作</span><strong>{{ statusLabel(a7AdmissionReport.operational_action) }}</strong></div>
              <div><span>Redis</span><strong>{{ statusLabel(a7AdmissionReport.redis_decision) }}</strong></div>
            </div>
            <div class="hash-grid">
              <div><span>Collector</span><code>{{ a7AdmissionReport.collector_version }}</code></div>
              <div><span>Evidence</span><code :title="a7AdmissionReport.evidence_sha256">{{ shortHash(a7AdmissionReport.evidence_sha256) }}</code></div>
              <div><span>Raw observations</span><code :title="a7AdmissionReport.raw_observations_sha256">{{ shortHash(a7AdmissionReport.raw_observations_sha256) }}</code></div>
              <div><span>Threshold policy</span><code :title="a7AdmissionReport.threshold_policy_sha256">{{ shortHash(a7AdmissionReport.threshold_policy_sha256) }}</code></div>
              <div><span>Git</span><code :title="a7AdmissionReport.binding?.git_commit">{{ shortHash(a7AdmissionReport.binding?.git_commit) }}</code></div>
              <div><span>Workload</span><code :title="a7AdmissionReport.binding?.workload_sha256">{{ shortHash(a7AdmissionReport.binding?.workload_sha256) }}</code></div>
              <div><span>Database</span><code>{{ a7AdmissionReport.authenticity?.database_dialect || '未采集' }}</code></div>
              <div><span>Independent Workers</span><strong>{{ a7AdmissionReport.authenticity?.independent_worker_processes_verified ? '已验证' : '未验证' }}</strong></div>
            </div>
            <ul v-if="a7AdmissionReport.blockers.length" class="limitations"><li v-for="item in a7AdmissionReport.blockers" :key="item"><AlertCircle :size="14" />{{ item }}</li></ul>
          </article>
          <div v-else class="empty-state"><SearchX :size="24" /><span>没有压测准入报告</span></div>
        </section>
      </template>
    </main>
  </div>
</template>

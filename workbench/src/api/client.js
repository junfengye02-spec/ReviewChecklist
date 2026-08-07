const API_BASE = String(import.meta.env.VITE_TENDER_REVIEW_API_BASE || '/api/v1').replace(/\/$/, '')

function requestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return `tender-web-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

async function requestJson(path, { signal, method = 'GET', body, headers = {} } = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    signal,
    headers: {
      Accept: 'application/json',
      'X-Request-ID': requestId(),
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const data = response.status === 204 ? null : await response.json().catch(() => null)
  if (!response.ok) {
    const message = data?.error?.message || data?.detail || `请求失败（${response.status}）`
    const error = new Error(Array.isArray(message) ? message.map((item) => item.msg).join('；') : message)
    error.code = data?.error?.code || `http_${response.status}`
    error.status = response.status
    error.requestId = data?.error?.request_id || response.headers.get('X-Request-ID') || ''
    throw error
  }
  return data
}

async function requestResponse(path, { signal, method = 'GET', body, headers = {} } = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    signal,
    headers: {
      Accept: 'application/json',
      'X-Request-ID': requestId(),
      ...headers,
    },
    body,
  })
  const data = response.status === 204 ? null : await response.json().catch(() => null)
  if (!response.ok) {
    const message = data?.error?.message || data?.detail || `请求失败（${response.status}）`
    const error = new Error(Array.isArray(message) ? message.map((item) => item.msg).join('；') : message)
    error.code = data?.error?.code || `http_${response.status}`
    error.status = response.status
    error.requestId = data?.error?.request_id || response.headers.get('X-Request-ID') || ''
    throw error
  }
  return data
}

const encode = encodeURIComponent

export const tenderApi = {
  workbench: (signal) => requestJson('/workbench', { signal }),
  uploadDocument: (file, sourceSystem, sourceDocumentId, signal) => {
    const form = new FormData()
    form.append('file', file)
    form.append('source_system', sourceSystem)
    form.append('source_document_id', sourceDocumentId)
    return requestResponse('/documents', { method: 'POST', body: form, signal })
  },
  createReviewJob: (payload, { idempotencyKey, callerId }, signal) => requestJson(
    '/review-jobs',
    {
      method: 'POST',
      body: payload,
      signal,
      headers: {
        'Idempotency-Key': idempotencyKey,
        'X-Caller-ID': callerId,
        'X-Call-ID': `tender-review-job-${idempotencyKey}`,
      },
    },
  ),
  reviewJob: (jobId, signal) => requestJson(`/review-jobs/${encode(jobId)}`, { signal }),
  checkpoints: (jobId, signal) => requestJson(`/review-jobs/${encode(jobId)}/checkpoints`, { signal }),
  findings: (jobId, signal) => requestJson(`/review-jobs/${encode(jobId)}/findings`, { signal }),
  findingDecisions: (findingId, signal) => requestJson(`/findings/${encode(findingId)}/decisions`, { signal }),
  submitFindingDecision: (findingId, payload, signal) => requestJson(
    `/findings/${encode(findingId)}/decisions`,
    { method: 'POST', body: payload, signal, headers: { 'X-Call-ID': `tender-finding-${findingId}` } },
  ),
  ruleVersions: (ruleSetId, signal) => requestJson(`/rule-sets/${encode(ruleSetId)}/versions`, { signal }),
  ruleDiff: (versionId, againstVersionId, signal) => requestJson(
    `/rule-versions/${encode(versionId)}/diff?against_version_id=${encode(againstVersionId)}`,
    { signal },
  ),
  evaluateRule: (versionId, datasetVersionId, signal) => requestJson(
    `/rule-versions/${encode(versionId)}/evaluate`,
    { method: 'POST', body: { dataset_version_id: datasetVersionId }, signal },
  ),
  publishRule: (versionId, approverId, signal) => requestJson(
    `/rule-versions/${encode(versionId)}/publish`,
    { method: 'POST', body: { approver_kind: 'human', approver_id: approverId }, signal },
  ),
  rollbackRule: (ruleSetId, payload, signal) => requestJson(
    `/rule-sets/${encode(ruleSetId)}/rollback`,
    { method: 'POST', body: { approver_kind: 'human', ...payload }, signal },
  ),
  annotationDatasets: (signal) => requestJson('/annotation-datasets', { signal }),
  evaluationRun: (runId, signal) => requestJson(`/evaluation-runs/${encode(runId)}`, { signal }),
  evaluationReport: (runId, signal) => requestJson(`/evaluation-runs/${encode(runId)}/report`, { signal }),
  a4EvaluationRuns: (signal) => requestJson('/a4/evaluation-runs?limit=20', { signal }),
  a4EvaluationReport: (runId, signal) => requestJson(`/a4/evaluation-runs/${encode(runId)}/report`, { signal }),
  a7AdmissionReport: (signal) => requestJson('/a7/admission-report', { signal }),
  optimizationJob: (jobId, signal) => requestJson(`/optimization-jobs/${encode(jobId)}`, { signal }),
  optimizationAttempts: (jobId, signal) => requestJson(`/optimization-jobs/${encode(jobId)}/attempts`, { signal }),
  cancelOptimization: (jobId, signal) => requestJson(`/optimization-jobs/${encode(jobId)}/cancel`, { method: 'POST', signal }),
  auditEvents: (signal) => requestJson('/audit-events?limit=30', { signal }),
}

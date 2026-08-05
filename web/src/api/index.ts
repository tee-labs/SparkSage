import axios from 'axios';
import type {
  AskResponse,
  BlockListResponse,
  ConfigResponse,
  ConfigUpdateResponse,
  ConvertResponse,
  CreateKnowledgeBaseRequest,
  DocumentDetail,
  DocumentListResponse,
  FeedbackListResponse,
  FeedbackRating,
  FeedbackStats,
  GenerateResponse,
  HealthResponse,
  IngestAndIndexResponse,
  IngestJobSnapshot,
  IngestJobSubmitResponse,
  KnowledgeBaseInfo,
  KnowledgeBaseListResponse,
  KnowledgeBaseSummary,
  QueryHistoryResponse,
  TagsResponse,
} from '@/types';

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,
});

export function setApiBase(base: string) {
  client.defaults.baseURL = base.replace(/\/$/, '');
}

export interface FeedbackPayload {
  query: string;
  answer_text: string;
  rating: FeedbackRating;
  correction?: string;
  block_ids?: string[];
}

export const api = {
  health: () => client.get<HealthResponse>('/health'),

  // ---- config ----
  getConfig: () => client.get<ConfigResponse>('/config').then((r) => r.data),
  saveConfig: (variables: Record<string, string>) =>
    client.post<ConfigUpdateResponse>('/config', variables).then((r) => r.data),

  // ---- ingest ----
  convert: (file: File, clean = true) => {
    const form = new FormData();
    form.append('file', file);
    form.append('clean', String(clean));
    return client
      .post<ConvertResponse>('/convert', form, { timeout: 0 })
      .then((r) => r.data);
  },
  generate: (
    file: File,
    opts: { clean?: boolean; max_blocks?: number | null; language?: string } = {},
  ) => {
    const form = new FormData();
    form.append('file', file);
    form.append('clean', String(opts.clean ?? true));
    if (opts.max_blocks) form.append('max_blocks', String(opts.max_blocks));
    if (opts.language) form.append('language', opts.language);
    return client
      .post<GenerateResponse>('/generate', form, { timeout: 0 })
      .then((r) => r.data);
  },
  kbIngest: (
    file: File,
    opts: {
      clean?: boolean;
      max_blocks?: number | null;
      language?: string;
      tags?: string[];
      auto_tag?: boolean;
      top_k?: number;
      kb_id?: string | null;
    } = {},
  ) => {
    const form = new FormData();
    form.append('file', file);
    form.append('clean', String(opts.clean ?? true));
    if (opts.max_blocks) form.append('max_blocks', String(opts.max_blocks));
    if (opts.language) form.append('language', opts.language);
    if (opts.tags?.length) form.append('tags', opts.tags.join(','));
    form.append('auto_tag', String(opts.auto_tag ?? true));
    if (opts.top_k) form.append('top_k', String(opts.top_k));
    if (opts.kb_id) form.append('kb_id', opts.kb_id);
    return client
      .post<IngestAndIndexResponse>('/knowledge_base/ingest', form, { timeout: 0 })
      .then((r) => r.data);
  },
  kbIngestAsync: (
    file: File,
    opts: {
      clean?: boolean;
      max_blocks?: number | null;
      language?: string;
      tags?: string[];
      auto_tag?: boolean;
      top_k?: number;
      kb_id?: string | null;
    } = {},
  ) => {
    const form = new FormData();
    form.append('file', file);
    form.append('clean', String(opts.clean ?? true));
    if (opts.max_blocks) form.append('max_blocks', String(opts.max_blocks));
    if (opts.language) form.append('language', opts.language);
    if (opts.tags?.length) form.append('tags', opts.tags.join(','));
    form.append('auto_tag', String(opts.auto_tag ?? true));
    if (opts.top_k) form.append('top_k', String(opts.top_k));
    if (opts.kb_id) form.append('kb_id', opts.kb_id);
    return client
      .post<IngestJobSubmitResponse>('/knowledge_base/ingest/async', form)
      .then((r) => r.data);
  },
  getIngestJob: (jobId: string) =>
    client.get<IngestJobSnapshot>(`/jobs/${jobId}`).then((r) => r.data),
  cancelIngestJob: (jobId: string) =>
    client.post<IngestJobSnapshot>(`/jobs/${jobId}/cancel`).then((r) => r.data),

  // ---- documents ----
  listDocuments: (params: {
    tag?: string;
    q?: string;
    limit?: number;
    offset?: number;
  } = {}) => client.get<DocumentListResponse>('/documents', { params }).then((r) => r.data),
  getDocument: (id: string) =>
    client.get<DocumentDetail>(`/documents/${id}`).then((r) => r.data),
  updateDocument: (id: string, body: Partial<Pick<DocumentDetail, 'title' | 'tags' | 'summary'>>) =>
    client.patch<DocumentDetail>(`/documents/${id}`, body).then((r) => r.data),
  deleteDocument: (id: string) => client.delete(`/documents/${id}`).then((r) => r.data),
  retagDocument: (id: string, opts: { top_k?: number; replace?: boolean } = {}) =>
    client.post<DocumentDetail>(`/documents/${id}/tags`, opts).then((r) => r.data),
  listTags: () => client.get<TagsResponse>('/tags').then((r) => r.data),

  // ---- knowledge base ----
  kbInfo: (kbId?: string) =>
    client
      .get<KnowledgeBaseInfo>('/knowledge_base', { params: kbId ? { kb_id: kbId } : undefined })
      .then((r) => r.data),
  listBlocks: (params: {
    tag?: string;
    language?: string;
    status?: string;
    kb_id?: string;
    limit?: number;
    offset?: number;
  } = {}) => client.get<BlockListResponse>('/knowledge_base/blocks', { params }).then((r) => r.data),
  kbTags: (kbId?: string) =>
    client
      .get<TagsResponse>('/knowledge_base/tags', { params: kbId ? { kb_id: kbId } : undefined })
      .then((r) => r.data),

  // ---- knowledge bases (multi-KB management) ----
  listKnowledgeBases: (params: { limit?: number; offset?: number } = {}) =>
    client.get<KnowledgeBaseListResponse>('/knowledge_bases', { params }).then((r) => r.data),
  getKnowledgeBase: (kbId: string) =>
    client.get<KnowledgeBaseSummary>(`/knowledge_bases/${kbId}`).then((r) => r.data),
  createKnowledgeBase: (body: CreateKnowledgeBaseRequest) =>
    client.post<KnowledgeBaseSummary>('/knowledge_bases', body).then((r) => r.data),
  deleteKnowledgeBase: (kbId: string) =>
    client.delete(`/knowledge_bases/${kbId}`).then((r) => r.data),
  activateKnowledgeBase: (kbId: string) =>
    client.post<KnowledgeBaseSummary>(`/knowledge_bases/${kbId}/activate`).then((r) => r.data),

  // ---- QA ----
  ask: (
    body: {
      query: string;
      kb_id?: string;
      k?: number;
      use_lexical?: boolean;
      use_rerank?: boolean;
      tags?: string[];
      history?: { role: 'user' | 'assistant'; content: string }[];
      mode?: string;
    },
    options?: { onProgress?: (p: AgentProgressEvent) => void; signal?: AbortSignal },
  ) => askQaStream(body, options),

  // ---- feedback ----
  feedbackStats: () => client.get<FeedbackStats>('/feedback').then((r) => r.data),
  feedbackRecords: (params: { limit?: number; offset?: number } = {}) =>
    client.get<FeedbackListResponse>('/feedback/records', { params }).then((r) => r.data),
  submitFeedback: (payload: FeedbackPayload) =>
    client.post('/feedback', payload).then((r) => r.data),

  // ---- QA conversation history ----
  queryHistory: (params: { limit?: number; offset?: number; kb_id?: string | null } = {}) =>
    client.get<QueryHistoryResponse>('/query/history', { params }).then((r) => r.data),
  clearHistory: (kbId?: string | null) =>
    client
      .delete('/query/history', { params: kbId ? { kb_id: kbId } : undefined })
      .then((r) => r.data),
};

export interface AgentProgressEvent {
  iteration: number;
  max_iterations: number;
  phase: string;
  percent: number;
  evidence_count: number;
  action?: string;
  thought?: string;
  query?: string;
  relevance_score?: number;
  relevance_reasoning?: string;
  refined_query?: string;
}

interface AskRequestBody {
  query: string;
  kb_id?: string;
  k?: number;
  use_lexical?: boolean;
  use_rerank?: boolean;
  tags?: string[];
  history?: { role: 'user' | 'assistant'; content: string }[];
  mode?: string;
}

interface AxiosLikeError {
  response?: { status?: number; data?: { detail?: string } };
  message?: string;
}

function makeError(status: number, detail: string): AxiosLikeError {
  return { response: { status, data: { detail } } };
}

function parseSseBlock(raw: string): { event: string; data: unknown } | null {
  let event = '';
  let dataStr = '';
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
  }
  if (!event) return null;
  let data: unknown = dataStr;
  if (dataStr) {
    try {
      data = JSON.parse(dataStr);
    } catch {
      /* keep raw string */
    }
  }
  return { event, data };
}

async function askQaStream(
  body: AskRequestBody,
  options?: { onProgress?: (p: AgentProgressEvent) => void; signal?: AbortSignal },
): Promise<AskResponse> {
  const base = (client.defaults.baseURL ?? '/api/v1').replace(/\/$/, '');
  let resp: Response;
  try {
    resp = await fetch(`${base}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ ...body, stream: true }),
      signal: options?.signal,
    });
  } catch (e) {
    throw (e as Error)?.name === 'AbortError'
      ? makeError(0, '已取消')
      : makeError(0, (e as Error)?.message ?? '请求失败');
  }
  if (!resp.ok || !resp.body) {
    let detail = '请求失败';
    try {
      const data = await resp.json();
      detail = (data as { detail?: string })?.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw makeError(resp.status, detail);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let result: AskResponse | null = null;
  let errorDetail: string | null = null;

  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      errorDetail = (e as Error)?.name === 'AbortError' ? '已取消' : '连接中断';
      break;
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const evt = parseSseBlock(raw);
      if (!evt) continue;
      if (evt.event === 'progress') {
        if (options?.onProgress) options.onProgress(evt.data as AgentProgressEvent);
      } else if (evt.event === 'result') {
        result = evt.data as AskResponse;
      } else if (evt.event === 'error') {
        errorDetail =
          ((evt.data as { detail?: string } | undefined)?.detail) ?? '问答运行出错';
      } else if (evt.event === 'done') {
        break;
      }
    }
    if (result || errorDetail) break;
  }

  try {
    reader.releaseLock();
  } catch {
    /* ignore */
  }
  if (errorDetail) throw makeError(0, errorDetail);
  if (!result) throw makeError(0, '问答运行未返回结果');
  return result;
}

export { client as axiosClient };

import axios from 'axios';
import type {
  AskResponse,
  BlockListResponse,
  ConfigResponse,
  ConfigUpdateResponse,
  ConvertResponse,
  DocumentDetail,
  DocumentListResponse,
  FeedbackListResponse,
  FeedbackRating,
  FeedbackStats,
  GenerateResponse,
  KnowledgeBaseInfo,
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
  health: () => client.get<{ status: string; generator_configured: boolean }>('/health'),

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
      .post<ConvertResponse>('/convert', form)
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
      .post<GenerateResponse>('/generate', form)
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
    return client
      .post<GenerateResponse>('/knowledge_base/ingest', form)
      .then((r) => r.data);
  },

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
  kbInfo: () =>
    client.get<KnowledgeBaseInfo>('/knowledge_base').then((r) => r.data),
  listBlocks: (params: {
    tag?: string;
    language?: string;
    status?: string;
    limit?: number;
    offset?: number;
  } = {}) => client.get<BlockListResponse>('/knowledge_base/blocks', { params }).then((r) => r.data),

  // ---- QA ----
  ask: (body: {
    query: string;
    k?: number;
    use_lexical?: boolean;
    use_rerank?: boolean;
    tags?: string[];
    history?: { role: 'user' | 'assistant'; content: string }[];
  }) => client.post<AskResponse>('/query', body).then((r) => r.data),

  // ---- feedback ----
  feedbackStats: () => client.get<FeedbackStats>('/feedback').then((r) => r.data),
  feedbackRecords: (params: { limit?: number; offset?: number } = {}) =>
    client.get<FeedbackListResponse>('/feedback/records', { params }).then((r) => r.data),
  submitFeedback: (payload: FeedbackPayload) =>
    client.post('/feedback', payload).then((r) => r.data),
};

export { client as axiosClient };

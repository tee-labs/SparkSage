export interface SourceInfo {
  uri?: string | null;
  title?: string | null;
  locator?: string | null;
  system?: string | null;
}

export interface ConvertResponse {
  markdown: string;
  title?: string | null;
  source: SourceInfo;
  cleaned: boolean;
}

export interface GenerationStats {
  raw_block_count: number;
  emitted: number;
  skipped: number;
  errors: string[];
}

export interface IdeaBlock {
  id: string;
  name: string;
  critical_question: string;
  trusted_answer: string;
  tags: string[];
  entities?: unknown[];
  keywords: string[];
  source?: SourceInfo | null;
  language?: string;
  status?: string;
  parents?: string[];
  confidence?: number | null;
  kb_id?: string | null;
  created_at?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export interface GenerateResponse {
  blocks: IdeaBlock[];
  title?: string | null;
  source: SourceInfo;
  cleaned: boolean;
  stats?: GenerationStats | null;
}

export interface ConfigResponse {
  variables: Record<string, string>;
}

export interface ConfigUpdateResponse {
  applied: string[];
  restart_required: boolean;
  message: string;
}

export interface DocumentSummary {
  doc_id: string;
  title?: string | null;
  summary?: string | null;
  tags: string[];
  source: SourceInfo;
  created_at: string;
  updated_at: string;
  content_hash?: string | null;
}

export interface DocumentDetail extends DocumentSummary {
  body_markdown: string;
  metadata: Record<string, unknown>;
}

export interface DocumentListResponse {
  items: DocumentSummary[];
  count: number;
  total: number;
  tag?: string | null;
  q?: string | null;
  limit: number;
  offset: number;
}

export interface TagsResponse {
  tags: string[];
}

export interface Citation {
  block_id: string;
  quote: string;
  uri?: string | null;
  locator?: string | null;
  title?: string | null;
}

export interface RetrievedChunk {
  block_id: string;
  name: string;
  critical_question: string;
  trusted_answer: string;
  score: number;
  rank: number;
}

export interface AskResponse {
  query: string;
  answer: string;
  abstained: boolean;
  abstention_reason?: string | null;
  citations: Citation[];
  retrieved: RetrievedChunk[];
  cached: boolean;
  confidence: number;
  intent?: string | null;
}

export interface KnowledgeBaseInfo {
  kb_id: string;
  name: string;
  block_count: number;
  document_count: number;
  language?: string;
  description?: string | null;
  tags: string[];
}

export interface BlockOut {
  id: string;
  name: string;
  critical_question: string;
  trusted_answer: string;
  tags: string[];
  keywords: string[];
  language: string;
  status: string;
  confidence?: number | null;
  parents: string[];
  source?: SourceInfo | null;
  kb_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface BlockListResponse {
  items: BlockOut[];
  count: number;
  total: number;
  limit: number;
  offset: number;
}

export interface FeedbackStats {
  total: number;
  positive: number;
  negative: number;
  corrected: number;
  approval: number;
}

export interface FeedbackRecord {
  feedback_id: string;
  query: string;
  answer_text: string;
  rating: string;
  correction?: string | null;
  block_ids: string[];
  kb_id?: string | null;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface FeedbackListResponse {
  items: FeedbackRecord[];
  count: number;
  total: number;
  limit: number;
  offset: number;
}

export type FeedbackRating = 'positive' | 'negative' | 'corrected';

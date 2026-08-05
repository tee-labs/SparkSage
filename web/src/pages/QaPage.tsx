import { useEffect, useState } from 'react';
import {
  Card,
  Input,
  Button,
  Space,
  Switch,
  InputNumber,
  Select,
  message,
  Empty,
  Collapse,
  Tag as AntTag,
  Typography,
  Statistic,
  Tooltip,
  Divider,
} from 'antd';
import {
  LikeOutlined,
  DislikeOutlined,
  EditOutlined,
  DeleteOutlined,
} from '@ant-design/icons';
import { api } from '@/api';
import type { AgentProgressEvent } from '@/api';
import type { AskResponse, FeedbackRating } from '@/types';
import Markdown from '@/components/Markdown';
import KbSelector from '@/components/KbSelector';
import { useKnowledgeBases } from '@/components/useKnowledgeBases';

const { Title, Text, Paragraph } = Typography;

interface Turn {
  role: 'user' | 'assistant';
  content: string;
  result?: AskResponse;
}

export default function QaPage() {
  const { kbs, loading: kbLoading, selectedKbId, setSelectedKbId } = useKnowledgeBases();
  const [query, setQuery] = useState('');
  const [k, setK] = useState(5);
  const [useLexical, setUseLexical] = useState(true);
  const [useRerank, setUseRerank] = useState(true);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [mode, setMode] = useState<string>('default');
  const [history, setHistory] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [agentProgress, setAgentProgress] = useState<AgentProgressEvent | null>(null);
  const [feedback, setFeedback] = useState<Record<string, FeedbackRating>>({});
  const [corrections, setCorrections] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    api
      .queryHistory({ limit: 100, kb_id: selectedKbId ?? undefined })
      .then((r) => {
        if (cancelled) return;
        const turns: Turn[] = [...r.items]
          .reverse()
          .map((item) =>
            item.role === 'user'
              ? { role: 'user', content: item.content }
              : {
                  role: 'assistant',
                  content: item.content,
                  result: item.result ?? undefined,
                },
          );
        setHistory(turns);
        setFeedback({});
        setCorrections({});
      })
      .catch(() => {
        // history restore is best-effort
      });
    return () => {
      cancelled = true;
    };
  }, [selectedKbId]);

  const ask = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setAgentProgress({
      iteration: 0,
      max_iterations: 0,
      phase: mode === 'agent' ? 'thinking' : 'understanding',
      percent: 0,
      evidence_count: 0,
    });
    try {
      const res = await api.ask(
        {
          query,
          kb_id: selectedKbId ?? undefined,
          k,
          use_lexical: useLexical,
          use_rerank: useRerank,
          tags: tagFilter.length ? tagFilter : undefined,
          mode,
          history: history.map((t) => ({ role: t.role, content: t.content })),
        },
        {
          onProgress: (p) => setAgentProgress(p),
        },
      );
      const userTurn: Turn = { role: 'user', content: query };
      const assistantTurn: Turn = { role: 'assistant', content: res.answer, result: res };
      setHistory((prev) => [...prev, userTurn, assistantTurn]);
      setQuery('');
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
      setAgentProgress(null);
    }
  };

  const removeTurn = (idx: number) => {
    setHistory((prev) => prev.filter((_, i) => i !== idx && i !== idx + 1));
  };

  const clearHistory = async () => {
    try {
      await api.clearHistory(selectedKbId ?? undefined);
      setHistory([]);
      setFeedback({});
      setCorrections({});
      message.success('历史已清空');
    } catch (e) {
      message.error(errText(e));
    }
  };

  const editTurn = (idx: number) => {
    const turn = history[idx];
    if (turn?.role === 'user') {
      setQuery(turn.content);
      removeTurn(idx);
    }
  };

  const submitFeedback = async (turn: Turn, rating: FeedbackRating) => {
    const res = turn.result;
    if (!res) return;
    try {
      await api.submitFeedback({
        query: res.query,
        answer_text: res.answer,
        rating,
        correction: rating === 'corrected' ? corrections[res.query] : undefined,
        block_ids: res.retrieved.map((c) => c.block_id),
      });
      setFeedback((prev) => ({ ...prev, [res.query]: rating }));
      message.success('反馈已提交');
    } catch (e) {
      message.error(errText(e));
    }
  };

  const tagOptions = [
    'IMPORTANT', 'WARNING', 'TECHNOLOGY', 'PROCESS', 'REFERENCE',
    'FAQ', 'TROUBLESHOOTING', 'SECURITY', 'ARCHITECTURE', 'API',
    'DATASET', 'POLICY',
  ].map((t) => ({ value: t, label: t }));

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Space style={{ justifyContent: 'space-between', width: '100%' }}>
        <Title level={4} style={{ margin: 0 }}>
          问答测试
        </Title>
        <Button size="small" icon={<DeleteOutlined />} onClick={clearHistory}>
          清空历史
        </Button>
      </Space>

      <Card size="small" title="参数">
        <Space wrap>
          <span>知识库：</span>
          <KbSelector
            value={selectedKbId}
            onChange={setSelectedKbId}
            kbs={kbs}
            loading={kbLoading}
          />
          <span>k：</span>
          <InputNumber min={1} max={50} value={k} onChange={(v) => setK(v ?? 5)} />
          <span>use_lexical</span>
          <Switch checked={useLexical} onChange={setUseLexical} />
          <span>use_rerank</span>
          <Switch checked={useRerank} onChange={setUseRerank} />
          <Tooltip
            title={
              mode === 'agent'
                ? 'Agent：LLM 驱动的多轮检索，适合多跳/比较类问题，较慢。'
                : 'Default：单轮问答，速度快，适合简单事实查询。'
            }
          >
            <span>模式：</span>
          </Tooltip>
          <Select
            style={{ width: 140 }}
            value={mode}
            onChange={setMode}
            options={[
              { value: 'default', label: 'Default（单轮）' },
              { value: 'agent', label: 'Agent（多跳推理）' },
            ]}
          />
          <Select
            mode="multiple"
            allowClear
            placeholder="标签过滤"
            style={{ minWidth: 200 }}
            value={tagFilter}
            onChange={setTagFilter}
            options={tagOptions}
          />
        </Space>
      </Card>

      {history.length === 0 && (
        <Empty description="还没有对话，输入问题开始问答" />
      )}

      {history.map((turn, idx) => (
        <Card
          key={idx}
          size="small"
          title={turn.role === 'user' ? '🧑 用户' : '🤖 助手'}
          extra={
            turn.role === 'user' && (
              <Space>
                <Button size="small" onClick={() => editTurn(idx)}>编辑</Button>
                <Button size="small" danger onClick={() => removeTurn(idx)}>删除</Button>
              </Space>
            )
          }
        >
          {turn.role === 'user' ? (
            <Text>{turn.content}</Text>
          ) : (
            <AnswerView turn={turn} />
          )}
          {turn.role === 'assistant' && turn.result && (
            <FeedbackBar
              turn={turn}
              feedback={feedback}
              corrections={corrections}
              setCorrections={setCorrections}
              onSubmit={submitFeedback}
            />
          )}
        </Card>
      ))}

      <Card size="small">
        {loading && agentProgress && (
          <AgentProgressView progress={agentProgress} mode={mode} />
        )}
        <Input.TextArea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入你的问题…"
          onPressEnter={(e) => {
            if (e.ctrlKey || e.metaKey) ask();
          }}
          disabled={loading}
        />
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
          <Button type="primary" onClick={ask} loading={loading} disabled={loading}>
            提问
          </Button>
        </div>
      </Card>
    </Space>
  );
}

function AnswerView({ turn }: { turn: Turn }) {
  const res = turn.result!;
  return (
    <Space direction="vertical" style={{ width: '100%' }} size="small">
      <Markdown>{res.answer}</Markdown>
      <Space wrap>
        <Statistic
          title="置信度"
          value={res.confidence}
          precision={2}
          valueStyle={{ fontSize: 14 }}
        />
        {res.mode === 'agent' && (
          <AntTag color="geekblue">
            Agent 多跳推理（{res.iterations ?? 0} 次迭代{res.aborted ? ' · 已中断' : ''}）
          </AntTag>
        )}
        {res.cached && <AntTag color="purple">缓存命中</AntTag>}
        {res.intent && <AntTag color="cyan">意图：{res.intent}</AntTag>}
        {res.abstained && (
          <AntTag color="orange">放弃回答：{res.abstention_reason ?? ''}</AntTag>
        )}
      </Space>

      {res.mode === 'agent' && (res.steps?.length ?? 0) > 0 && (
        <Collapse
          size="small"
          items={[
            {
              key: 'agent',
              label: `推理过程（${res.steps!.length} 步 · 证据 ${res.retrieved.length}）`,
              children: (
                <Space direction="vertical" style={{ width: '100%' }}>
                  {res.steps!.map((s, i) => (
                    <Card key={i} size="small" type="inner">
                      <Space>
                        <AntTag color="gold">第 {i + 1} 步</AntTag>
                        <AntTag color="blue">命中 {s.retrieved_count}</AntTag>
                      </Space>
                      {s.thought && (
                        <Paragraph style={{ margin: '4px 0' }}>
                          <Text type="secondary">思考：</Text>
                          {s.thought}
                        </Paragraph>
                      )}
                      <Paragraph style={{ margin: '4px 0' }}>
                        <Text type="secondary">子查询：</Text>
                        <Text strong>{s.query}</Text>
                      </Paragraph>
                      {s.observation && (
                        <Paragraph
                          ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
                          style={{ marginBottom: 0 }}
                        >
                          <Text type="secondary">观察：</Text>
                          {s.observation}
                        </Paragraph>
                      )}
                    </Card>
                  ))}
                </Space>
              ),
            },
          ]}
        />
      )}

      {res.citations.length > 0 && (
        <Collapse
          size="small"
          items={[
            {
              key: 'cit',
              label: `引用来源（${res.citations.length}）`,
              children: (
                <Space direction="vertical" style={{ width: '100%' }}>
                  {res.citations.map((c, i) => (
                    <Card key={i} size="small" type="inner">
                      <Paragraph style={{ marginBottom: 4 }}>
                        <Text type="secondary">block_id：</Text>
                        <Text copyable>{c.block_id}</Text>
                      </Paragraph>
                      {c.quote && <Paragraph style={{ marginBottom: 4 }}>{c.quote}</Paragraph>}
                      <Text type="secondary">
                        {[c.uri, c.locator, c.title].filter(Boolean).join(' · ')}
                      </Text>
                    </Card>
                  ))}
                </Space>
              ),
            },
          ]}
        />
      )}

      {res.retrieved.length > 0 && (
        <Collapse
          size="small"
          defaultActiveKey={['ret']}
          items={[
            {
              key: 'ret',
              label: `检索块（${res.retrieved.length}）`,
              children: (
                <Space direction="vertical" style={{ width: '100%' }}>
                  {res.retrieved.map((c) => (
                    <Card key={c.block_id} size="small" type="inner">
                      <Space>
                        <AntTag color="blue">#{c.rank + 1}</AntTag>
                        <AntTag>{c.score.toFixed(3)}</AntTag>
                        <Text strong>{c.name}</Text>
                      </Space>
                      <Paragraph ellipsis={{ rows: 1 }} style={{ margin: '4px 0' }}>
                        <Text type="secondary">问：</Text>
                        {c.critical_question}
                      </Paragraph>
                      <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 0 }}>
                        <Text type="secondary">答：</Text>
                        {c.trusted_answer}
                      </Paragraph>
                    </Card>
                  ))}
                </Space>
              ),
            },
          ]}
        />
      )}
    </Space>
  );
}

function FeedbackBar({
  turn,
  feedback,
  corrections,
  setCorrections,
  onSubmit,
}: {
  turn: Turn;
  feedback: Record<string, FeedbackRating>;
  corrections: Record<string, string>;
  setCorrections: (updater: (prev: Record<string, string>) => Record<string, string>) => void;
  onSubmit: (turn: Turn, rating: FeedbackRating) => void;
}) {
  const res = turn.result!;
  const current = feedback[res.query];
  const [editing, setEditing] = useState(false);
  const correctionValue = corrections[res.query] ?? '';
  return (
    <>
      <Divider style={{ margin: '8px 0' }} />
      <Space>
        <Tooltip title="有帮助">
          <Button
            size="small"
            type={current === 'positive' ? 'primary' : 'default'}
            icon={<LikeOutlined />}
            onClick={() => onSubmit(turn, 'positive')}
          />
        </Tooltip>
        <Tooltip title="无帮助">
          <Button
            size="small"
            danger={current === 'negative'}
            icon={<DislikeOutlined />}
            onClick={() => onSubmit(turn, 'negative')}
          />
        </Tooltip>
        <Tooltip title="纠正">
          <Button
            size="small"
            type={current === 'corrected' ? 'primary' : 'default'}
            icon={<EditOutlined />}
            onClick={() => setEditing((e) => !e)}
          />
        </Tooltip>
        {current && <AntTag color="green">已反馈：{current}</AntTag>}
      </Space>
      {editing && (
        <div style={{ marginTop: 8 }}>
          <Input.TextArea
            rows={2}
            placeholder="请输入正确答案…"
            value={correctionValue}
            onChange={(e) =>
              setCorrections((prev) => ({ ...prev, [res.query]: e.target.value }))
            }
          />
          <div style={{ marginTop: 4, textAlign: 'right' }}>
            <Button
              size="small"
              type="primary"
              disabled={!correctionValue.trim()}
              onClick={() => {
                onSubmit(turn, 'corrected');
                setEditing(false);
              }}
            >
              提交纠正
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

function errText(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const resp = (e as { response?: { data?: { detail?: string }; status?: number } }).response;
    if (resp?.status === 503) return 'QA 未配置（需要 LLM）';
    return resp?.data?.detail ?? '请求失败';
  }
  return (e as Error)?.message ?? String(e);
}

const AGENT_PHASE_LABEL: Record<string, string> = {
  thinking: '思考中（规划子查询）',
  retrieving: '检索知识库',
  synthesizing: '整合证据并生成答案',
  done: '完成',
};

const DEFAULT_PHASE_LABEL: Record<string, string> = {
  understanding: '理解问题（意图识别 + 改写）',
  retrieving: '检索知识库（混合召回 + 重排）',
  generating: '生成答案（基于检索证据）',
  done: '完成',
};

function AgentProgressView({
  progress,
  mode,
}: {
  progress: AgentProgressEvent;
  mode?: string;
}) {
  const labels = mode === 'agent' ? AGENT_PHASE_LABEL : DEFAULT_PHASE_LABEL;
  const label = labels[progress.phase] ?? progress.phase;
  const pct = Math.max(0, Math.min(100, Math.round(progress.percent * 100)));
  const isAgent = mode === 'agent';
  return (
    <div style={{ marginBottom: 12 }}>
      <Space style={{ marginBottom: 6 }}>
        <AntTag color={isAgent ? 'geekblue' : 'blue'}>
          {isAgent ? 'Agent 推理中' : '问答处理中'}
        </AntTag>
        <Text strong>{label}</Text>
        {isAgent && progress.iteration > 0 && (
          <Text type="secondary">
            第 {progress.iteration}/{progress.max_iterations} 轮 · 证据 {progress.evidence_count}
          </Text>
        )}
        {progress.action && <AntTag color="blue">{progress.action}</AntTag>}
      </Space>
      {progress.query && (
        <Paragraph type="secondary" style={{ margin: '2px 0', fontSize: 12 }} ellipsis={{ rows: 1 }}>
          子查询：{progress.query}
        </Paragraph>
      )}
      {progress.relevance_score != null && (
        <Text type="secondary" style={{ fontSize: 12 }}>
          相关度 {progress.relevance_score.toFixed(2)}
          {progress.refined_query ? ' · 已改写查询重试' : ''}
        </Text>
      )}
      <div style={{ marginTop: 6, height: 4, background: '#f0f0f0', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#1677ff', transition: 'width .3s' }} />
      </div>
    </div>
  );
}

import { useState } from 'react';
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
  LoadingOutlined,
} from '@ant-design/icons';
import { api } from '@/api';
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
  const [history, setHistory] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, FeedbackRating>>({});
  const [corrections, setCorrections] = useState<Record<string, string>>({});

  const ask = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await api.ask({
        query,
        kb_id: selectedKbId ?? undefined,
        k,
        use_lexical: useLexical,
        use_rerank: useRerank,
        tags: tagFilter.length ? tagFilter : undefined,
        history: history.map((t) => ({ role: t.role, content: t.content })),
      });
      const userTurn: Turn = { role: 'user', content: query };
      const assistantTurn: Turn = { role: 'assistant', content: res.answer, result: res };
      setHistory((prev) => [...prev, userTurn, assistantTurn]);
      setQuery('');
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
    }
  };

  const removeTurn = (idx: number) => {
    setHistory((prev) => prev.filter((_, i) => i !== idx && i !== idx + 1));
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
      <Title level={4} style={{ margin: 0 }}>
        问答测试
      </Title>

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
            {loading && <LoadingOutlined />} 提问
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
        {res.cached && <AntTag color="purple">缓存命中</AntTag>}
        {res.intent && <AntTag color="cyan">意图：{res.intent}</AntTag>}
        {res.abstained && (
          <AntTag color="orange">放弃回答：{res.abstention_reason ?? ''}</AntTag>
        )}
      </Space>

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

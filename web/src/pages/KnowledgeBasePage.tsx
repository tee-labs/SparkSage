import { useCallback, useEffect, useState } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Select,
  Space,
  Button,
  message,
  List,
  Tag as AntTag,
  Badge,
  Modal,
  Descriptions,
  Typography,
  Pagination,
  Input,
} from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { api } from '@/api';
import type { BlockOut, KnowledgeBaseInfo } from '@/types';
import KbSelector from '@/components/KbSelector';
import { useKnowledgeBases } from '@/components/useKnowledgeBases';

const { Title, Text, Paragraph } = Typography;

const STATUS_COLORS: Record<string, string> = {
  ACTIVE: 'success',
  MERGED: 'warning',
  DRAFT: 'default',
  ARCHIVED: 'default',
};

const PAGE_SIZE = 20;

export default function KnowledgeBasePage() {
  const { kbs, loading: kbLoading, selectedKbId, setSelectedKbId } = useKnowledgeBases();
  const [info, setInfo] = useState<KnowledgeBaseInfo | null>(null);
  const [blocks, setBlocks] = useState<BlockOut[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [language, setLanguage] = useState<string>();
  const [status, setStatus] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState<BlockOut | null>(null);
  const [tags, setTags] = useState<string[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [kb, blk, tagsRes] = await Promise.all([
        api.kbInfo(selectedKbId ?? undefined),
        api.listBlocks({
          tag: tagFilter.length ? tagFilter.join(',') : undefined,
          language,
          status,
          kb_id: selectedKbId ?? undefined,
          limit: PAGE_SIZE,
          offset: (page - 1) * PAGE_SIZE,
        }),
        api.kbTags(selectedKbId ?? undefined),
      ]);
      setInfo(kb);
      setBlocks(blk.items);
      setTotal(blk.total);
      setTags(tagsRes.tags);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
    }
  }, [tagFilter, language, status, page, selectedKbId]);

  useEffect(() => {
    load();
  }, [load]);

  const tagOptions = tags.map((t) => ({ value: t, label: t }));

  const card = (b: BlockOut) => (
    <Card
      size="small"
      hoverable
      onClick={() => setActive(b)}
      title={
        <Space>
          <Text strong>{b.name}</Text>
          <Badge status={(STATUS_COLORS[b.status] as 'success') ?? 'default'} text={b.status} />
        </Space>
      }
      extra={b.confidence != null ? <Text type="secondary">conf {(b.confidence * 100).toFixed(0)}%</Text> : null}
    >
      <Paragraph style={{ marginBottom: 4 }} ellipsis={{ rows: 1 }}>
        <Text type="secondary">问：</Text>
        {b.critical_question}
      </Paragraph>
      <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 4 }}>
        <Text type="secondary">答：</Text>
        {b.trusted_answer}
      </Paragraph>
      {(b.tags?.length ?? 0) > 0 && (
        <Space size={[4, 4]} wrap>
          {b.tags.map((t) => (
            <AntTag key={t}>{t}</AntTag>
          ))}
        </Space>
      )}
    </Card>
  );

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Title level={4} style={{ margin: 0 }}>
        知识库浏览
      </Title>

      <Row gutter={16}>
        <Col>
          <Card size="small">
            <Statistic title="Block 总数" value={info?.block_count ?? 0} loading={loading && !info} />
          </Card>
        </Col>
        <Col>
          <Card size="small">
            <Statistic title="Document 总数" value={info?.document_count ?? 0} loading={loading && !info} />
          </Card>
        </Col>
      </Row>

      <Card size="small">
        <Space wrap>
          <span>知识库：</span>
          <KbSelector
            value={selectedKbId}
            onChange={(v) => {
              setSelectedKbId(v);
              setPage(1);
            }}
            kbs={kbs}
            loading={kbLoading}
          />
          <Select
            mode="multiple"
            allowClear
            placeholder="标签过滤"
            style={{ minWidth: 220 }}
            value={tagFilter}
            onChange={(v) => {
              setTagFilter(v);
              setPage(1);
            }}
            options={tagOptions}
          />
          <Select
            allowClear
            placeholder="语言"
            style={{ width: 120 }}
            value={language}
            onChange={(v) => {
              setLanguage(v);
              setPage(1);
            }}
            options={[
              { value: 'en', label: 'en' },
              { value: 'zh', label: 'zh' },
              { value: 'ja', label: 'ja' },
            ]}
          />
          <Select
            allowClear
            placeholder="状态"
            style={{ width: 140 }}
            value={status}
            onChange={(v) => {
              setStatus(v);
              setPage(1);
            }}
            options={['ACTIVE', 'MERGED', 'DRAFT', 'ARCHIVED'].map((s) => ({
              value: s,
              label: s,
            }))}
          />
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>
      </Card>

      <List
        grid={{ gutter: 16, xs: 1, sm: 1, md: 2, lg: 2, xl: 3 }}
        dataSource={blocks}
        loading={loading}
        locale={{ emptyText: '没有匹配的 block' }}
        renderItem={(b) => <List.Item>{card(b)}</List.Item>}
      />
      <Pagination
        current={page}
        pageSize={PAGE_SIZE}
        total={total}
        onChange={setPage}
        showTotal={(t) => `共 ${t} 个 block`}
        showSizeChanger={false}
      />

      <Modal
        title="Block 详情"
        open={!!active}
        onCancel={() => setActive(null)}
        footer={null}
        width={640}
      >
        {active && (
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="id">
                <Text copyable>{active.id}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="name">{active.name}</Descriptions.Item>
              <Descriptions.Item label="status">
                <Badge status={(STATUS_COLORS[active.status] as 'success') ?? 'default'} text={active.status} />
              </Descriptions.Item>
              <Descriptions.Item label="language">{active.language}</Descriptions.Item>
              {active.confidence != null && (
                <Descriptions.Item label="confidence">
                  {(active.confidence * 100).toFixed(1)}%
                </Descriptions.Item>
              )}
              <Descriptions.Item label="tags">
                {(active.tags ?? []).join(', ') || '-'}
              </Descriptions.Item>
              <Descriptions.Item label="keywords">
                {(active.keywords ?? []).join(', ') || '-'}
              </Descriptions.Item>
              {active.parents?.length > 0 && (
                <Descriptions.Item label="parents">
                  {active.parents.map((p) => (
                    <Tag key={p}>{p}</Tag>
                  ))}
                </Descriptions.Item>
              )}
              {active.source?.uri && (
                <Descriptions.Item label="source.uri">
                  {active.source.uri}
                </Descriptions.Item>
              )}
            </Descriptions>
            <div>
              <Text type="secondary">关键问题</Text>
              <Input.TextArea value={active.critical_question} readOnly autoSize />
            </div>
            <div>
              <Text type="secondary">可信答案</Text>
              <Input.TextArea value={active.trusted_answer} readOnly autoSize />
            </div>
          </Space>
        )}
      </Modal>
    </Space>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return <AntTag style={{ marginBottom: 4 }}>{children}</AntTag>;
}

function errText(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const resp = (e as { response?: { data?: { detail?: string } } }).response;
    return resp?.data?.detail ?? '请求失败';
  }
  return (e as Error)?.message ?? String(e);
}

import { useCallback, useEffect, useState } from 'react';
import {
  Card,
  Table,
  Input,
  Select,
  Button,
  Space,
  Drawer,
  Form,
  message,
  Modal,
  Tag as AntTag,
  Typography,
  Pagination,
  Spin,
} from 'antd';
import { ReloadOutlined, DeleteOutlined, SaveOutlined } from '@ant-design/icons';
import { api } from '@/api';
import type { DocumentDetail, DocumentSummary } from '@/types';
import Markdown from '@/components/Markdown';

const { Title, Text, Paragraph } = Typography;

const PAGE_SIZE = 20;

export default function DocumentsPage() {
  const [items, setItems] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [editForm] = Form.useForm();
  const [editing, setEditing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [listRes, tagsRes] = await Promise.all([
        api.listDocuments({
          tag: tagFilter.length ? tagFilter.join(',') : undefined,
          q: query || undefined,
          limit: PAGE_SIZE,
          offset: (page - 1) * PAGE_SIZE,
        }),
        api.listTags(),
      ]);
      setItems(listRes.items);
      setTotal(listRes.total);
      setTags(tagsRes.tags);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
    }
  }, [tagFilter, query, page]);

  useEffect(() => {
    load();
  }, [load]);

  const openDetail = async (doc: DocumentSummary) => {
    setDetailLoading(true);
    setEditing(false);
    try {
      const d = await api.getDocument(doc.doc_id);
      setDetail(d);
      editForm.setFieldsValue({
        title: d.title ?? '',
        summary: d.summary ?? '',
        tags: d.tags,
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      setDetailLoading(false);
    }
  };

  const closeDetail = () => {
    setDetail(null);
    setEditing(false);
  };

  const saveDetail = async () => {
    if (!detail) return;
    const values = await editForm.validateFields();
    try {
      const updated = await api.updateDocument(detail.doc_id, {
        title: values.title,
        summary: values.summary,
        tags: values.tags,
      });
      setDetail(updated);
      setEditing(false);
      message.success('已保存修改');
      load();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const retag = async () => {
    if (!detail) return;
    try {
      const updated = await api.retagDocument(detail.doc_id, { replace: true });
      setDetail(updated);
      editForm.setFieldsValue({ ...editForm.getFieldsValue(), tags: updated.tags });
      message.success('已重新提取标签');
      load();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const remove = (doc: DocumentSummary) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文档「${doc.title ?? doc.doc_id}」吗？`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await api.deleteDocument(doc.doc_id);
          message.success('已删除');
          load();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const columns = [
    {
      title: '标题',
      dataIndex: 'title',
      render: (v: string, r: DocumentSummary) => {
        const fallback = r.source?.uri?.split('/').pop() || r.doc_id;
        return <a onClick={() => openDetail(r)}>{v || fallback}</a>;
      },
    },
    {
      title: '标签',
      dataIndex: 'tags',
      render: (v: string[]) =>
        v?.length ? v.map((t) => <AntTag key={t}>{t}</AntTag>) : <Text type="secondary">-</Text>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: '摘要',
      dataIndex: 'summary',
      ellipsis: true,
      render: (v: string | null) => v || <Text type="secondary">-</Text>,
    },
    {
      title: '操作',
      render: (_: unknown, r: DocumentSummary) => (
        <Button danger size="small" icon={<DeleteOutlined />} onClick={() => remove(r)}>
          删除
        </Button>
      ),
    },
  ];

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Title level={4} style={{ margin: 0 }}>
        文档管理
      </Title>

      <Card size="small">
        <Space wrap>
          <Select
            mode="multiple"
            allowClear
            placeholder="标签过滤（任意匹配）"
            style={{ minWidth: 240 }}
            value={tagFilter}
            onChange={(v) => {
              setTagFilter(v);
              setPage(1);
            }}
            options={tags.map((t) => ({ value: t, label: t }))}
          />
          <Input.Search
            placeholder="关键字搜索标题/正文"
            allowClear
            style={{ width: 240 }}
            onSearch={(v) => {
              setQuery(v);
              setPage(1);
            }}
          />
          <Button icon={<ReloadOutlined />} onClick={load}>
            刷新
          </Button>
        </Space>
      </Card>

      <Table
        rowKey="doc_id"
        columns={columns}
        dataSource={items}
        loading={loading}
        pagination={false}
        size="middle"
      />
      <Pagination
        current={page}
        pageSize={PAGE_SIZE}
        total={total}
        onChange={setPage}
        showTotal={(t) => `共 ${t} 条`}
        showSizeChanger={false}
      />

      <Drawer
        title="文档详情"
        width={640}
        open={!!detail}
        onClose={closeDetail}
        destroyOnClose
        extra={
          detail && (
            <Space>
              <Button size="small" onClick={() => setEditing((e) => !e)}>
                {editing ? '取消编辑' : '编辑'}
              </Button>
              <Button size="small" icon={<ReloadOutlined />} onClick={retag}>
                重新提取标签
              </Button>
            </Space>
          )
        }
      >
        {detailLoading ? (
          <div style={{ textAlign: 'center', padding: 48 }}>
            <Spin />
          </div>
        ) : detail ? (
          editing ? (
            <Form form={editForm} layout="vertical">
              <Form.Item label="标题" name="title">
                <Input />
              </Form.Item>
              <Form.Item label="标签" name="tags">
                <Select mode="tags" placeholder="输入标签" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item label="摘要" name="summary">
                <Input.TextArea rows={3} />
              </Form.Item>
              <Space>
                <Button type="primary" icon={<SaveOutlined />} onClick={saveDetail}>
                  保存修改
                </Button>
                <Button danger onClick={() => remove({ doc_id: detail.doc_id, title: detail.title } as DocumentSummary)}>
                  删除
                </Button>
              </Space>
            </Form>
          ) : (
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div>
                <Text type="secondary">doc_id：</Text>
                <Text copyable>{detail.doc_id}</Text>
              </div>
              <Paragraph>
                <Text strong>{detail.title || '(无标题)'}</Text>
              </Paragraph>
              <div>
                {detail.tags.map((t) => (
                  <AntTag key={t}>{t}</AntTag>
                ))}
              </div>
              {detail.summary && (
                <Card size="small" title="摘要">
                  <Text>{detail.summary}</Text>
                </Card>
              )}
              <Card size="small" title="正文" styles={{ body: { maxHeight: 420, overflow: 'auto' } }}>
                <Markdown>{detail.body_markdown}</Markdown>
              </Card>
            </Space>
          )
        ) : null}
      </Drawer>
    </Space>
  );
}

function errText(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const resp = (e as { response?: { data?: { detail?: string } } }).response;
    return resp?.data?.detail ?? '请求失败';
  }
  return (e as Error)?.message ?? String(e);
}

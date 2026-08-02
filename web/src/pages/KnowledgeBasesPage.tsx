import { useCallback, useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  message,
  Tag as AntTag,
  Typography,
  Statistic,
  Row,
  Col,
} from 'antd';
import {
  ReloadOutlined,
  DeleteOutlined,
  PlusOutlined,
  CheckOutlined,
} from '@ant-design/icons';
import { api } from '@/api';
import type { KnowledgeBaseSummary } from '@/types';

const { Title, Text } = Typography;

export default function KnowledgeBasesPage() {
  const [items, setItems] = useState<KnowledgeBaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form] = Form.useForm();
  const [modalOpen, setModalOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listKnowledgeBases({ limit: 1000 });
      setItems(res.items);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const openCreate = () => {
    form.resetFields();
    form.setFieldsValue({ language: 'zh', set_active: true });
    setModalOpen(true);
  };

  const submitCreate = async () => {
    const values = await form.validateFields();
    setCreating(true);
    try {
      await api.createKnowledgeBase({
        name: values.name,
        description: values.description || null,
        language: values.language || 'zh',
        tags: values.tags,
        set_active: Boolean(values.set_active),
      });
      message.success('知识库已创建');
      setModalOpen(false);
      load();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setCreating(false);
    }
  };

  const activate = async (kb: KnowledgeBaseSummary) => {
    try {
      await api.activateKnowledgeBase(kb.kb_id);
      message.success(`已切换到「${kb.name}」`);
      load();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const remove = (kb: KnowledgeBaseSummary) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除知识库「${kb.name}」吗？其下所有文档与索引将被移除。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await api.deleteKnowledgeBase(kb.kb_id);
          message.success('已删除');
          load();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const totalBlocks = items.reduce((s, k) => s + k.block_count, 0);
  const totalDocs = items.reduce((s, k) => s + k.document_count, 0);

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      render: (v: string, r: KnowledgeBaseSummary) => (
        <Space>
          <Text strong>{v}</Text>
          {r.active && <AntTag color="green">当前</AntTag>}
        </Space>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      ellipsis: true,
      render: (v: string | null) => v || <Text type="secondary">-</Text>,
    },
    {
      title: '语言',
      dataIndex: 'language',
      width: 80,
    },
    {
      title: 'Block',
      dataIndex: 'block_count',
      width: 80,
      render: (v: number) => <Text>{v}</Text>,
    },
    {
      title: '文档',
      dataIndex: 'document_count',
      width: 80,
      render: (v: number) => <Text>{v}</Text>,
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
      width: 170,
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: '操作',
      width: 200,
      render: (_: unknown, r: KnowledgeBaseSummary) => (
        <Space>
          <Button
            size="small"
            type={r.active ? 'primary' : 'default'}
            icon={<CheckOutlined />}
            disabled={r.active}
            onClick={() => activate(r)}
          >
            {r.active ? '已激活' : '激活'}
          </Button>
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => remove(r)}>
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Title level={4} style={{ margin: 0 }}>
          知识库管理
        </Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            新建知识库
          </Button>
        </Space>
      </Row>

      <Row gutter={16}>
        <Col>
          <Card size="small">
            <Statistic title="知识库总数" value={items.length} />
          </Card>
        </Col>
        <Col>
          <Card size="small">
            <Statistic title="Block 总数" value={totalBlocks} />
          </Card>
        </Col>
        <Col>
          <Card size="small">
            <Statistic title="文档总数" value={totalDocs} />
          </Card>
        </Col>
      </Row>

      <Table
        rowKey="kb_id"
        columns={columns}
        dataSource={items}
        loading={loading}
        pagination={false}
        size="middle"
        locale={{ emptyText: '暂无知识库' }}
      />

      <Modal
        title="新建知识库"
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={submitCreate}
        confirmLoading={creating}
        okText="创建"
        cancelText="取消"
        destroyOnHidden
      >
        <Form form={form} layout="vertical" initialValues={{ language: 'zh', set_active: true }}>
          <Form.Item
            label="名称"
            name="name"
            rules={[{ required: true, message: '请输入知识库名称' }]}
          >
            <Input placeholder="例如：产品文档库" />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={2} placeholder="可选" />
          </Form.Item>
          <Form.Item label="语言" name="language">
            <Select
              options={[
                { value: 'zh', label: 'zh (中文)' },
                { value: 'en', label: 'en (English)' },
                { value: 'ja', label: 'ja (日本語)' },
              ]}
            />
          </Form.Item>
          <Form.Item label="标签" name="tags">
            <Select mode="tags" placeholder="输入标签（可选）" tokenSeparators={[',']} />
          </Form.Item>
          <Form.Item label="创建后激活" name="set_active" valuePropName="checked">
            <Select
              options={[
                { value: true, label: '是（设为当前知识库）' },
                { value: false, label: '否' },
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
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

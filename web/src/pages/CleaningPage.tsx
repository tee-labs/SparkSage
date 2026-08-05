import { useCallback, useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  message,
  Tag as AntTag,
  Typography,
  Tooltip,
} from 'antd';
import {
  ReloadOutlined,
  DeleteOutlined,
  PlusOutlined,
  EditOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { api } from '@/api';
import type { CleaningRule, CleaningTestResponse } from '@/types';

const { Title, Text, Paragraph } = Typography;

const DEFAULT_CODE = 'def clean(text, source=None):\n    return text\n';

const emptyForm = {
  name: '',
  code: DEFAULT_CODE,
  pattern_kind: 'none',
  source_pattern: '',
  enabled: true,
  timeout: 5,
  max_input_chars: 1_000_000,
  max_output_chars: 2_000_000,
};

export default function CleaningPage() {
  const [items, setItems] = useState<CleaningRule[]>([]);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState<CleaningRule | null>(null);
  const [saving, setSaving] = useState(false);
  const [testOpen, setTestOpen] = useState(false);
  const [testCode, setTestCode] = useState(DEFAULT_CODE);
  const [testText, setTestText] = useState('');
  const [testSource, setTestSource] = useState('');
  const [testResult, setTestResult] = useState<CleaningTestResponse | null>(null);
  const [testing, setTesting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listCleaningRules({ limit: 1000 });
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
    setEditing(null);
    form.resetFields();
    form.setFieldsValue(emptyForm);
    setEditOpen(true);
  };

  const openEdit = (r: CleaningRule) => {
    setEditing(r);
    form.setFieldsValue({
      name: r.name,
      code: r.code,
      pattern_kind: r.pattern_kind,
      source_pattern: r.source_pattern ?? '',
      enabled: r.enabled,
      timeout: r.timeout,
      max_input_chars: r.max_input_chars,
      max_output_chars: r.max_output_chars,
    });
    setEditOpen(true);
  };

  const submit = async () => {
    const values = await form.validateFields();
    setSaving(true);
    try {
      const body = {
        name: values.name,
        code: values.code,
        pattern_kind: values.pattern_kind,
        source_pattern: values.pattern_kind === 'none' ? null : values.source_pattern || null,
        enabled: values.enabled,
        timeout: values.timeout,
        max_input_chars: values.max_input_chars,
        max_output_chars: values.max_output_chars,
      };
      if (editing) {
        await api.updateCleaningRule(editing.rule_id, body);
        message.success('规则已更新');
      } else {
        await api.createCleaningRule(body);
        message.success('规则已创建');
      }
      setEditOpen(false);
      load();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSaving(false);
    }
  };

  const toggleEnabled = async (r: CleaningRule, enabled: boolean) => {
    try {
      await api.updateCleaningRule(r.rule_id, { enabled });
      load();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const remove = (r: CleaningRule) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除清洗规则「${r.name}」吗？`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await api.deleteCleaningRule(r.rule_id);
          message.success('已删除');
          load();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const openTest = (r: CleaningRule | null) => {
    setTestCode(r?.code ?? DEFAULT_CODE);
    setTestText('');
    setTestSource('');
    setTestResult(null);
    setTestOpen(true);
  };

  const runTest = async () => {
    if (!testText.trim()) {
      message.warning('请输入测试文本');
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const res = await api.testCleaningRule({
        code: testCode,
        text: testText,
        source: testSource || null,
      });
      setTestResult(res);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setTesting(false);
    }
  };

  const patternLabel = (r: CleaningRule) => {
    if (r.pattern_kind === 'none' || !r.source_pattern) return <Text type="secondary">全局</Text>;
    return (
      <Space size={4}>
        <AntTag color="blue">{r.pattern_kind}</AntTag>
        <Text code>{r.source_pattern}</Text>
      </Space>
    );
  };

  const statusTag = (r: CleaningRule) => {
    if (!r.enabled) return <AntTag>已禁用</AntTag>;
    if (!r.compiled) {
      return (
        <Tooltip title={r.error || '编译失败'}>
          <AntTag color="error">编译失败</AntTag>
        </Tooltip>
      );
    }
    return <AntTag color="success">正常</AntTag>;
  };

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      render: (v: string) => <Text strong>{v}</Text>,
    },
    { title: '匹配来源', render: (_: unknown, r: CleaningRule) => patternLabel(r) },
    {
      title: '启用',
      dataIndex: 'enabled',
      width: 80,
      render: (v: boolean, r: CleaningRule) => (
        <Switch checked={v} onChange={(c) => toggleEnabled(r, c)} size="small" />
      ),
    },
    { title: '状态', width: 110, render: (_: unknown, r: CleaningRule) => statusTag(r) },
    {
      title: '超时(s)',
      dataIndex: 'timeout',
      width: 80,
      render: (v: number) => <Text>{v}</Text>,
    },
    {
      title: '操作',
      width: 220,
      render: (_: unknown, r: CleaningRule) => (
        <Space>
          <Button size="small" icon={<ExperimentOutlined />} onClick={() => openTest(r)}>
            测试
          </Button>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(r)}>
            编辑
          </Button>
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => remove(r)} />
        </Space>
      ),
    },
  ];

  const kind = Form.useWatch('pattern_kind', form);

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Space style={{ justifyContent: 'space-between', width: '100%' }}>
        <Title level={4} style={{ margin: 0 }}>
          清洗规则
        </Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
          <Button icon={<ExperimentOutlined />} onClick={() => openTest(null)}>
            测试脚本
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            新建规则
          </Button>
        </Space>
      </Space>
      <Text type="secondary">
        自定义沙箱化 Python 清洗脚本（定义 clean(text, source) 函数），在文档转换后、分块前运行。规则保存后立即生效，无需重启。支持按文件名/路径路由（glob 或 regex），也可设为全局。
      </Text>

      <Card size="small">
        <Table
          rowKey="rule_id"
          columns={columns}
          dataSource={items}
          loading={loading}
          pagination={false}
          size="middle"
          locale={{ emptyText: '暂无清洗规则' }}
        />
      </Card>

      <Modal
        title={editing ? '编辑规则' : '新建规则'}
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={submit}
        confirmLoading={saving}
        okText={editing ? '保存' : '创建'}
        cancelText="取消"
        destroyOnHidden
        width={720}
      >
        <Form form={form} layout="vertical" initialValues={emptyForm}>
          <Form.Item
            label="名称"
            name="name"
            rules={[{ required: true, message: '请输入规则名称' }]}
          >
            <Input placeholder="例如：去除水印" />
          </Form.Item>
          <Form.Item
            label="脚本代码"
            name="code"
            rules={[{ required: true, message: '请输入脚本代码' }]}
            extra="沙箱内可用：str / list / re 等。禁止 import / eval / open / dunder 访问。"
          >
            <Input.TextArea rows={8} style={{ fontFamily: 'monospace' }} />
          </Form.Item>
          <Form.Item label="来源匹配" extra="仅对匹配的文件名/路径生效（全局 = 所有文档）">
            <Space>
              <Form.Item name="pattern_kind" noStyle>
                <Select
                  style={{ width: 120 }}
                  options={[
                    { value: 'none', label: '全局' },
                    { value: 'glob', label: 'glob' },
                    { value: 'regex', label: 'regex' },
                  ]}
                />
              </Form.Item>
              {kind !== 'none' && (
                <Form.Item name="source_pattern" noStyle>
                  <Input placeholder="例如 *.pdf 或 ^reports/" style={{ width: 320 }} />
                </Form.Item>
              )}
            </Space>
          </Form.Item>
          <Form.Item label="启用" name="enabled" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Space size="large">
            <Form.Item label="超时(秒)" name="timeout">
              <InputNumber min={0.1} step={0.5} style={{ width: 110 }} />
            </Form.Item>
            <Form.Item label="最大输入字符" name="max_input_chars">
              <InputNumber min={1} style={{ width: 150 }} />
            </Form.Item>
            <Form.Item label="最大输出字符" name="max_output_chars">
              <InputNumber min={1} style={{ width: 150 }} />
            </Form.Item>
          </Space>
        </Form>
      </Modal>

      <Modal
        title="测试清洗脚本"
        open={testOpen}
        onCancel={() => setTestOpen(false)}
        footer={[
          <Button key="close" onClick={() => setTestOpen(false)}>
            关闭
          </Button>,
          <Button key="run" type="primary" loading={testing} onClick={runTest}>
            运行
          </Button>,
        ]}
        width={760}
        destroyOnHidden
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <div>
            <Text type="secondary">脚本代码</Text>
            <Input.TextArea
              rows={6}
              value={testCode}
              onChange={(e) => setTestCode(e.target.value)}
              style={{ fontFamily: 'monospace', marginTop: 4 }}
            />
          </div>
          <div>
            <Text type="secondary">来源（可选，文件名/路径）</Text>
            <Input value={testSource} onChange={(e) => setTestSource(e.target.value)} style={{ marginTop: 4 }} placeholder="例如 report.pdf" />
          </div>
          <div>
            <Text type="secondary">测试文本</Text>
            <Input.TextArea
              rows={4}
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              style={{ marginTop: 4 }}
              placeholder="粘贴一段需要清洗的文本…"
            />
          </div>
          {testResult && (
            <Card size="small" title={testResult.ok ? '输出结果' : '运行失败'}>
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Space>
                  <AntTag color={testResult.ok ? 'success' : 'error'}>
                    {testResult.ok ? '成功' : '失败'}
                  </AntTag>
                  <Text type="secondary">{testResult.elapsed_ms} ms</Text>
                </Space>
                {testResult.error && (
                  <Text type="danger" style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                    {testResult.error}
                  </Text>
                )}
                <Paragraph style={{ margin: 0 }}>
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0, maxHeight: 240, overflow: 'auto' }}>
                    {testResult.output}
                  </pre>
                </Paragraph>
              </Space>
            </Card>
          )}
        </Space>
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

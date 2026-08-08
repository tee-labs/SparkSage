import { useEffect, useState } from 'react';
import {
  Card,
  Form,
  Input,
  Button,
  Select,
  Switch,
  Space,
  Alert,
  message,
  Typography,
  Divider,
} from 'antd';
import { KeyOutlined } from '@ant-design/icons';
import { api } from '@/api';

const { Title, Text } = Typography;

const SENSITIVE = new Set(['SPARKSAGE_API_KEY', 'SPARKSAGE_EMBEDDING_API_KEY']);

export default function ConfigPage() {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ applied: string[]; message: string } | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.getConfig();
      const norm: Record<string, unknown> = { ...data.variables };
      const raw = norm.SPARKSAGE_TAGS_ZH;
      norm.SPARKSAGE_TAGS_ZH = ['1', 'true', 'yes', 'on'].includes(
        (typeof raw === 'string' ? raw : '').toLowerCase(),
      );
      form.setFieldsValue(norm);
      setResult(null);
    } catch (e) {
      message.error('加载配置失败：' + errText(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSave = async () => {
    const values = await form.validateFields();
    setSaving(true);
    try {
      const patch: Record<string, string> = {};
      Object.entries(values as Record<string, unknown>).forEach(([k, v]) => {
        if (v === undefined || v === null) return;
        if (k === 'SPARKSAGE_TAGS_ZH') {
          patch[k] = v ? 'true' : 'false';
          return;
        }
        if (SENSITIVE.has(k) && v === '****') return;
        patch[k] = String(v);
      });
      const res = await api.saveConfig(patch);
      setResult({ applied: res.applied, message: res.message });
      message.success('配置已保存');
      await load();
    } catch (e) {
      message.error('保存失败：' + errText(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Title level={4} style={{ margin: 0 }}>
        配置管理
      </Title>
      <Text type="secondary">编辑并保存 .env 配置。敏感字段以 **** 显示。</Text>

      {result?.message && (
        <Alert
          type="warning"
          showIcon
          message={result.message}
          description={
            result.applied.length
              ? `已写入：${result.applied.join(', ')}`
              : '没有字段被修改。'
          }
          closable
          onClose={() => setResult(null)}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        disabled={loading}
        style={{ maxWidth: 720 }}
      >
        <Card title="LLM 配置" size="small" loading={loading}>
          <Form.Item label="SPARKSAGE_API_KEY" name="SPARKSAGE_API_KEY">
            <Input.Password
              placeholder="sk-..."
              iconRender={(visible) =>
                visible ? <KeyOutlined /> : <KeyOutlined />
              }
            />
          </Form.Item>
          <Form.Item label="SPARKSAGE_BASE_URL" name="SPARKSAGE_BASE_URL">
            <Input placeholder="https://api.openai.com/v1" />
          </Form.Item>
          <Form.Item label="SPARKSAGE_MODEL" name="SPARKSAGE_MODEL">
            <Input placeholder="gpt-4o-mini" />
          </Form.Item>
          <Form.Item label="SPARKSAGE_LANGUAGE" name="SPARKSAGE_LANGUAGE">
            <Select
              options={[
                { value: 'en', label: 'English (en)' },
                { value: 'zh', label: '中文 (zh)' },
                { value: 'ja', label: '日本語 (ja)' },
              ]}
              allowClear
            />
          </Form.Item>
        </Card>

        <Card title="Embedding 配置" size="small" style={{ marginTop: 12 }} loading={loading}>
          <Form.Item label="SPARKSAGE_EMBEDDING_API_KEY" name="SPARKSAGE_EMBEDDING_API_KEY">
            <Input.Password placeholder="sk-..." />
          </Form.Item>
          <Form.Item label="SPARKSAGE_EMBEDDING_BASE_URL" name="SPARKSAGE_EMBEDDING_BASE_URL">
            <Input placeholder="https://api.openai.com/v1" />
          </Form.Item>
          <Form.Item label="SPARKSAGE_EMBEDDING_MODEL" name="SPARKSAGE_EMBEDDING_MODEL">
            <Input placeholder="text-embedding-3-small" />
          </Form.Item>
        </Card>

        <Card title="转换器配置" size="small" style={{ marginTop: 12 }} loading={loading}>
          <Form.Item
            label="SPARKSAGE_CONVERTER"
            name="SPARKSAGE_CONVERTER"
            extra="格式转换引擎：markitdown（默认，格式覆盖最广）或 anydoc（Firecrawl 本地 Rust 转换器，处理 office/PDF 更快更轻）"
          >
            <Select
              options={[
                { value: 'markitdown', label: 'markitdown' },
                { value: 'anydoc', label: 'anydoc (Firecrawl)' },
              ]}
              allowClear
            />
          </Form.Item>
        </Card>

        <Card title="存储配置" size="small" style={{ marginTop: 12 }} loading={loading}>
          <Form.Item
            label="SPARKSAGE_DOC_STORE"
            name="SPARKSAGE_DOC_STORE"
            extra="SQLite 文件路径，留空为内存存储"
          >
            <Input placeholder="./sparksage.docs.db" />
          </Form.Item>
          <Form.Item label="SPARKSAGE_DOC_STORE_TABLE" name="SPARKSAGE_DOC_STORE_TABLE">
            <Input placeholder="documents" />
          </Form.Item>
        </Card>

        <Card title="标签配置" size="small" style={{ marginTop: 12 }} loading={loading}>
          <Form.Item label="SPARKSAGE_AUTO_TAG_EXTRACTOR" name="SPARKSAGE_AUTO_TAG_EXTRACTOR">
            <Select
              options={[
                { value: 'rake', label: 'RAKE' },
                { value: 'tfidf', label: 'TF-IDF' },
                { value: 'textrank', label: 'TextRank' },
              ]}
              allowClear
            />
          </Form.Item>
          <Form.Item
            label="SPARKSAGE_AUTO_TAG_MIN_COHESION"
            name="SPARKSAGE_AUTO_TAG_MIN_COHESION"
            extra="CJK bigram 凝聚度阈值 (0-1)；过小会保留跨词噪声 bigram，off 关闭过滤"
          >
            <Input placeholder="0.34" />
          </Form.Item>
          <Form.Item label="SPARKSAGE_TAGS_ZH" name="SPARKSAGE_TAGS_ZH" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Card>

        <Card title="日志配置" size="small" style={{ marginTop: 12 }} loading={loading}>
          <Form.Item label="SPARKSAGE_LOG_LEVEL" name="SPARKSAGE_LOG_LEVEL">
            <Select
              options={[
                { value: 'DEBUG', label: 'DEBUG' },
                { value: 'INFO', label: 'INFO' },
                { value: 'WARNING', label: 'WARNING' },
                { value: 'ERROR', label: 'ERROR' },
                { value: 'CRITICAL', label: 'CRITICAL' },
              ]}
              allowClear
            />
          </Form.Item>
        </Card>

        <Divider />
        <Space>
          <Button type="primary" onClick={onSave} loading={saving}>
            保存配置
          </Button>
          <Button onClick={load} disabled={loading || saving}>
            重新加载
          </Button>
        </Space>
      </Form>
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

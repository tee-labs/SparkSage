import { useState } from 'react';
import {
  Card,
  Upload,
  Tabs,
  Form,
  Select,
  InputNumber,
  Switch,
  Button,
  Space,
  message,
  Empty,
  Collapse,
  Timeline,
  Typography,
  Tag as AntTag,
} from 'antd';
import { InboxOutlined, DownloadOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { api } from '@/api';
import type { ConvertResponse, GenerateResponse, IdeaBlock } from '@/types';
import Markdown from '@/components/Markdown';

const { Dragger } = Upload;
const { Text } = Typography;

interface LogEntry {
  step: string;
  time: string;
}

export default function IngestPage() {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [form] = Form.useForm();
  const [mode, setMode] = useState<'convert' | 'generate'>('convert');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [convertResult, setConvertResult] = useState<ConvertResponse | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateResponse | null>(null);

  const pushLog = (step: string) =>
    setLogs((prev) => [...prev, { step, time: new Date().toLocaleTimeString() }]);

  const onTabChange = (key: string) => {
    setMode(key as 'convert' | 'generate');
    setConvertResult(null);
    setGenerateResult(null);
    setLogs([]);
  };

  const run = async () => {
    const target = files.find((f) => f.originFileObj);
    if (!target?.originFileObj) {
      message.warning('请先上传文件');
      return;
    }
    const values = await form.validateFields();
    const file = target.originFileObj as File;
    setLoading(true);
    setLogs([]);
    try {
      if (mode === 'convert') {
        pushLog('开始转换');
        const res = await api.convert(file, Boolean(values.clean));
        pushLog('转换完成');
        setConvertResult(res);
      } else {
        pushLog('开始生成 IdeaBlock');
        const useKb = Boolean(values.kb_ingest);
        const opts = {
          clean: Boolean(values.clean),
          max_blocks: values.max_blocks ?? null,
          language: values.language,
          tags: values.tags,
          auto_tag: Boolean(values.auto_tag),
          top_k: values.top_k,
        };
        const res = useKb
          ? await api.kbIngest(file, opts)
          : await api.generate(file, opts);
        pushLog(`生成完成：${res.blocks.length} 个 block`);
        setGenerateResult(res);
      }
    } catch (e) {
      message.error(errText(e));
      pushLog('处理失败');
    } finally {
      setLoading(false);
    }
  };

  const exportJson = () => {
    const data = generateResult?.blocks ?? [];
    if (!data.length) {
      message.warning('没有可导出的 block');
      return;
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ideablocks.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Typography.Title level={4} style={{ margin: 0 }}>
        文档上传
      </Typography.Title>

      <Card size="small">
        <Dragger
          multiple
          beforeUpload={() => false}
          fileList={files}
          onChange={({ fileList }) => setFiles(fileList)}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
          <p className="ant-upload-hint">支持多文件上传</p>
        </Dragger>
      </Card>

      <Card title="参数" size="small">
        <Form form={form} layout="inline" initialValues={{ clean: true, auto_tag: true, top_k: 8, kb_ingest: false }}>
          <Form.Item label="language" name="language">
            <Select
              allowClear
              style={{ width: 120 }}
              options={[
                { value: 'en', label: 'en' },
                { value: 'zh', label: 'zh' },
                { value: 'ja', label: 'ja' },
              ]}
            />
          </Form.Item>
          <Form.Item label="max_blocks" name="max_blocks">
            <InputNumber min={1} placeholder="不限" style={{ width: 110 }} />
          </Form.Item>
          <Form.Item label="clean" name="clean" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item label="auto_tag" name="auto_tag" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item label="top_k" name="top_k">
            <InputNumber min={1} style={{ width: 90 }} />
          </Form.Item>
          <Form.Item label="入库" name="kb_ingest" valuePropName="checked" tooltip="勾选则写入知识库索引">
            <Switch />
          </Form.Item>
        </Form>
      </Card>

      <Tabs
        activeKey={mode}
        onChange={onTabChange}
        items={[
          { key: 'convert', label: '转换预览' },
          { key: 'generate', label: '完整生成' },
        ]}
      />

      <Space>
        <Button type="primary" onClick={run} loading={loading}>
          开始（{mode === 'convert' ? '转换' : '生成'}）
        </Button>
        {mode === 'generate' && (
          <Button icon={<DownloadOutlined />} onClick={exportJson} disabled={!generateResult?.blocks.length}>
            导出 JSON
          </Button>
        )}
      </Space>

      {logs.length > 0 && (
        <Card title="操作日志" size="small">
          <Timeline
            items={logs.map((l) => ({
              children: (
                <span>
                  <Text strong>{l.step}</Text> <Text type="secondary">{l.time}</Text>
                </span>
              ),
            }))}
          />
        </Card>
      )}

      {mode === 'convert' && convertResult && (
        <Card title="转换结果" size="small">
          <Markdown>{convertResult.markdown}</Markdown>
        </Card>
      )}

      {mode === 'generate' && generateResult && (
        <Card title={`生成结果（${generateResult.blocks.length} 个 block）`} size="small">
          {generateResult.blocks.length === 0 ? (
            <Empty description="没有生成 block" />
          ) : (
            <Collapse
              items={generateResult.blocks.map((b: IdeaBlock, i: number) => ({
                key: b.id ?? i,
                label: (
                  <Space>
                    <Text strong>{b.name}</Text>
                    {(b.tags ?? []).map((t) => (
                      <AntTag key={t} color="blue">
                        {t}
                      </AntTag>
                    ))}
                  </Space>
                ),
                children: (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text type="secondary">关键问题：</Text>
                      <Text>{b.critical_question}</Text>
                    </div>
                    <div>
                      <Text type="secondary">可信答案：</Text>
                      <Markdown>{b.trusted_answer}</Markdown>
                    </div>
                    <pre
                      style={{
                        background: '#f6f8fa',
                        padding: 8,
                        borderRadius: 6,
                        fontSize: 12,
                        maxHeight: 240,
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify(b, null, 2)}
                    </pre>
                  </Space>
                ),
              }))}
            />
          )}
        </Card>
      )}
    </Space>
  );
}

function errText(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const resp = (e as { response?: { data?: { detail?: string }; status?: number } }).response;
    if (resp?.status === 503) return '生成未配置（需要设置 SPARKSAGE_API_KEY）';
    return resp?.data?.detail ?? '请求失败';
  }
  return (e as Error)?.message ?? String(e);
}

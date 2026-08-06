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
  Progress,
} from 'antd';
import { InboxOutlined, DownloadOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { api } from '@/api';
import type { ConvertResponse, IdeaBlock, IngestJobSnapshot } from '@/types';
import Markdown from '@/components/Markdown';
import KbSelector from '@/components/KbSelector';
import { useKnowledgeBases } from '@/components/useKnowledgeBases';

const { Dragger } = Upload;
const { Text } = Typography;

interface LogEntry {
  step: string;
  time: string;
}

interface FileJobState {
  filename: string;
  status: string;
  phase: string;
  percent: number;
  error?: string | null;
  blockCount?: number;
}

const POLL_INTERVAL_MS = 5000;
const TERMINAL = new Set(['success', 'failed', 'cancelled']);

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function IngestPage() {
  const { kbs, loading: kbLoading, selectedKbId, setSelectedKbId } = useKnowledgeBases();
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [form] = Form.useForm();
  const [mode, setMode] = useState<'convert' | 'generate'>('convert');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [convertResult, setConvertResult] = useState<ConvertResponse | null>(null);
  const [generateResult, setGenerateResult] = useState<{ blocks: IdeaBlock[] } | null>(null);
  const [fileJobs, setFileJobs] = useState<FileJobState[]>([]);

  const pushLog = (step: string) =>
    setLogs((prev) => [...prev, { step, time: new Date().toLocaleTimeString() }]);

  const upsertJob = (state: FileJobState) =>
    setFileJobs((prev) => {
      const idx = prev.findIndex((p) => p.filename === state.filename);
      if (idx < 0) return [...prev, state];
      const next = prev.slice();
      next[idx] = state;
      return next;
    });

  const onTabChange = (key: string) => {
    setMode(key as 'convert' | 'generate');
    setConvertResult(null);
    setGenerateResult(null);
    setLogs([]);
    setFileJobs([]);
  };

  async function pollIngestJob(
    jobId: string,
    filename: string,
  ): Promise<IngestJobSnapshot> {
    for (;;) {
      const snap = await api.getIngestJob(jobId);
      upsertJob({
        filename,
        status: snap.status,
        phase: snap.phase,
        percent: snap.percent,
        error: snap.error,
        blockCount: snap.block_count,
      });
      if (TERMINAL.has(snap.status)) return snap;
      await delay(POLL_INTERVAL_MS);
    }
  }

  const run = async () => {
    const targets = files
      .map((f) => f.originFileObj)
      .filter((f): f is NonNullable<typeof f> => f != null);
    if (!targets.length) {
      message.warning('请先上传文件');
      return;
    }
    const values = await form.validateFields();
    const clean = Boolean(values.clean);
    setLoading(true);
    setLogs([]);
    setFileJobs([]);
    try {
      if (mode === 'convert') {
        const parts: string[] = [];
        let last: ConvertResponse | null = null;
        for (const file of targets) {
          pushLog(`开始转换：${file.name}`);
          const res = await api.convert(file, clean);
          pushLog(`转换完成：${file.name}`);
          parts.push(`### ${file.name}\n\n${res.markdown}`);
          last = res;
        }
        setConvertResult({
          ...(last as ConvertResponse),
          markdown: parts.join('\n\n'),
        });
        pushLog('全部转换完成');
      } else {
        const useKb = Boolean(values.kb_ingest);
        const opts = {
          clean,
          max_blocks: values.max_blocks ?? null,
          language: values.language,
          tags: values.tags,
          auto_tag: Boolean(values.auto_tag),
          top_k: values.top_k,
          kb_id: selectedKbId,
        };
        const allBlocks: IdeaBlock[] = [];
        let okCount = 0;
        let failCount = 0;
        for (const file of targets) {
          if (useKb) {
            try {
              pushLog(`提交：${file.name}`);
              const { job_id } = await api.kbIngestAsync(file, opts);
              const snap = await pollIngestJob(job_id, file.name);
              if (snap.status === 'success') {
                okCount += 1;
                pushLog(`完成：${file.name}（${snap.block_count} 个 block）`);
                if (snap.result?.blocks) allBlocks.push(...snap.result.blocks);
              } else if (snap.status === 'cancelled') {
                failCount += 1;
                pushLog(`已取消：${file.name}`);
              } else {
                failCount += 1;
                pushLog(`失败：${file.name} - ${snap.error ?? '未知错误'}`);
              }
            } catch (e) {
              failCount += 1;
              pushLog(`失败：${file.name} - ${errText(e)}`);
            }
          } else {
            try {
              pushLog(`开始生成 IdeaBlock：${file.name}`);
              const res = await api.generate(file, opts);
              okCount += 1;
              pushLog(`生成完成：${file.name}（${res.blocks.length} 个 block）`);
              allBlocks.push(...res.blocks);
            } catch (e) {
              failCount += 1;
              pushLog(`失败：${file.name} - ${errText(e)}`);
            }
          }
        }
        setGenerateResult({ blocks: allBlocks });
        if (failCount === 0) {
          pushLog(`全部完成：共 ${allBlocks.length} 个 block`);
          message.success(`${okCount} 个文件处理完成`);
        } else if (okCount === 0) {
          pushLog('全部失败');
          message.error('全部文件处理失败');
        } else {
          pushLog(`部分完成：成功 ${okCount}，失败 ${failCount}，共 ${allBlocks.length} 个 block`);
          message.warning(`成功 ${okCount} 个，失败 ${failCount} 个`);
        }
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
        <Form form={form} layout="inline" initialValues={{ clean: true, auto_tag: true, top_k: 8, kb_ingest: false, language: 'zh' }}>
          <Form.Item label="知识库">
            <KbSelector
              value={selectedKbId}
              onChange={setSelectedKbId}
              kbs={kbs}
              loading={kbLoading}
              placeholder="选择知识库"
            />
          </Form.Item>
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
          <Form.Item label="入库" name="kb_ingest" valuePropName="checked" tooltip="勾选则写入知识库索引（异步任务，可在下方查看进度）">
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

      {fileJobs.length > 0 && (
        <Card title="任务进度" size="small">
          {fileJobs.map((j) => (
            <div key={j.filename} style={{ marginBottom: 8 }}>
              <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                <Text strong>{j.filename}</Text>
                <Text type="secondary">
                  {j.status === 'success'
                    ? `完成（${j.blockCount ?? 0} 个 block）`
                    : j.status === 'failed'
                      ? `失败：${j.error ?? '未知错误'}`
                      : j.status === 'cancelled'
                        ? '已取消'
                        : `${j.phase}（${Math.round(j.percent * 100)}%）`}
                </Text>
              </Space>
              <Progress
                percent={Math.round(j.percent * 100)}
                status={
                  j.status === 'failed'
                    ? 'exception'
                    : j.status === 'success'
                      ? 'success'
                      : 'active'
                }
                size="small"
              />
            </div>
          ))}
        </Card>
      )}

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

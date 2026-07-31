import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Row,
  Col,
  Card,
  Statistic,
  Progress,
  Timeline,
  Badge,
  Typography,
  Space,
  Empty,
  Spin,
  Button,
  Collapse,
} from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { api } from '@/api';
import type { FeedbackRecord, FeedbackStats } from '@/types';

const { Title, Text, Paragraph } = Typography;

const RATING_COLOR: Record<string, 'success' | 'error' | 'processing'> = {
  positive: 'success',
  negative: 'error',
  corrected: 'processing',
};

const RATING_LABEL: Record<string, string> = {
  positive: '👍 有帮助',
  negative: '👎 无帮助',
  corrected: '✏️ 纠正',
};

export default function FeedbackPage() {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [records, setRecords] = useState<FeedbackRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, r] = await Promise.all([
        api.feedbackStats(),
        api.feedbackRecords({ limit: 50 }),
      ]);
      setStats(s);
      setRecords(r.items);
    } catch {
      // silent on background refresh
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    timer.current = setInterval(load, 30000);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [load]);

  const approvalPct = stats ? Math.round(stats.approval * 100) : 0;

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Space style={{ justifyContent: 'space-between', width: '100%' }}>
        <Title level={4} style={{ margin: 0 }}>
          反馈统计
        </Title>
        <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
          立即刷新
        </Button>
      </Space>
      <Text type="secondary">每 30 秒自动刷新</Text>

      <Row gutter={16}>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic title="Total" value={stats?.total ?? 0} loading={!stats} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic
              title="Positive"
              value={stats?.positive ?? 0}
              valueStyle={{ color: '#3f8600' }}
              loading={!stats}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic
              title="Negative"
              value={stats?.negative ?? 0}
              valueStyle={{ color: '#cf1322' }}
              loading={!stats}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic
              title="Corrected"
              value={stats?.corrected ?? 0}
              valueStyle={{ color: '#1677ff' }}
              loading={!stats}
            />
          </Card>
        </Col>
      </Row>

      <Card size="small" title="满意度">
        {stats ? (
          <Progress percent={approvalPct} status={approvalPct >= 70 ? 'success' : 'active'} />
        ) : (
          <Spin />
        )}
        {stats && (
          <Text type="secondary">认可率 {approvalPct}%（{stats.positive}/{stats.total}）</Text>
        )}
      </Card>

      <Card size="small" title="最近反馈">
        {!stats ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin />
          </div>
        ) : records.length === 0 ? (
          <Empty description="还没有反馈记录" />
        ) : (
          <Timeline
            items={records.map((r) => ({
              color: RATING_COLOR[r.rating] === 'success' ? 'green' : RATING_COLOR[r.rating] === 'error' ? 'red' : 'blue',
              children: (
                <Collapse
                  size="small"
                  ghost
                  items={[
                    {
                      key: r.feedback_id,
                      label: (
                        <Space direction="vertical" size={0} style={{ width: '100%' }}>
                          <Space>
                            <Badge status={RATING_COLOR[r.rating]} text={RATING_LABEL[r.rating]} />
                            <Text type="secondary">
                              {new Date(r.created_at).toLocaleString()}
                            </Text>
                          </Space>
                          <Paragraph ellipsis={{ rows: 1 }} style={{ margin: 0 }}>
                            <Text type="secondary">Q：</Text>
                            {r.query}
                          </Paragraph>
                          <Paragraph ellipsis={{ rows: 1 }} style={{ margin: 0 }}>
                            <Text type="secondary">A：</Text>
                            {r.answer_text || '-'}
                          </Paragraph>
                        </Space>
                      ),
                      children: (
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <div>
                            <Text type="secondary">完整问题：</Text>
                            <Paragraph>{r.query}</Paragraph>
                          </div>
                          <div>
                            <Text type="secondary">完整答案：</Text>
                            <Paragraph>{r.answer_text || '-'}</Paragraph>
                          </div>
                          {r.correction && (
                            <div>
                              <Text type="secondary">用户纠正：</Text>
                              <Paragraph>{r.correction}</Paragraph>
                            </div>
                          )}
                          {r.block_ids?.length > 0 && (
                            <Text type="secondary">block_ids：{r.block_ids.join(', ')}</Text>
                          )}
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            }))}
          />
        )}
      </Card>
    </Space>
  );
}

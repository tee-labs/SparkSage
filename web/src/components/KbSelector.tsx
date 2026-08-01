import { Select, Tag } from 'antd';
import type { KnowledgeBaseSummary } from '@/types';

interface KbSelectorProps {
  value?: string | null;
  onChange: (value: string | null) => void;
  kbs: KnowledgeBaseSummary[];
  loading?: boolean;
  style?: React.CSSProperties;
  placeholder?: string;
}

export default function KbSelector({
  value,
  onChange,
  kbs,
  loading,
  style,
  placeholder = '选择知识库',
}: KbSelectorProps) {
  return (
    <Select<string>
      allowClear
      showSearch
      optionFilterProp="label"
      placeholder={placeholder}
      style={{ minWidth: 220, ...style }}
      value={value ?? undefined}
      loading={loading}
      onChange={(v) => onChange(v ?? null)}
      options={kbs.map((k) => ({
        value: k.kb_id,
        label: `${k.name}${k.active ? ' ·' : ''}`,
      }))}
      optionRender={(option) => {
        const kb = kbs.find((k) => k.kb_id === option.value);
        return (
          <span>
            {option.label}
            {kb?.active && (
              <Tag color="green" style={{ marginInlineStart: 6 }}>
                当前
              </Tag>
            )}
          </span>
        );
      }}
    />
  );
}

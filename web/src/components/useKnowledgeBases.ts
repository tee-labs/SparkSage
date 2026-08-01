import { useCallback, useEffect, useState } from 'react';
import { api } from '@/api';
import type { KnowledgeBaseSummary } from '@/types';

export interface UseKnowledgeBasesResult {
  kbs: KnowledgeBaseSummary[];
  activeKb: KnowledgeBaseSummary | undefined;
  loading: boolean;
  selectedKbId: string | null;
  setSelectedKbId: (id: string | null) => void;
  reload: () => Promise<void>;
}

export function useKnowledgeBases(): UseKnowledgeBasesResult {
  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listKnowledgeBases({ limit: 1000 });
      setKbs(res.items);
      setSelectedKbId((prev) => {
        if (prev && res.items.some((k) => k.kb_id === prev)) return prev;
        const active = res.items.find((k) => k.active);
        return active ? active.kb_id : res.items[0]?.kb_id ?? null;
      });
    } catch {
      setKbs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const activeKb = kbs.find((k) => k.active);

  return { kbs, activeKb, loading, selectedKbId, setSelectedKbId, reload };
}

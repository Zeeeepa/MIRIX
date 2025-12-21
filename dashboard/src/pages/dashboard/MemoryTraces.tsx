import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import apiClient from '@/api/client';
import { useWorkspace } from '@/contexts/WorkspaceContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { 
  RefreshCcw, 
  ChevronDown, 
  ChevronUp, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  AlertCircle,
  Brain,
  Database,
  Book,
  Wrench,
  Layers,
  Cpu,
  ArrowRight
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';

interface MemoryQueueTrace {
  id: string;
  status: string;
  queued_at?: string;
  started_at?: string;
  completed_at?: string;
  success?: boolean;
  error_message?: string;
  user_id?: string;
  agent_id?: string;
  message_count?: number;
  triggered_memory_types?: string[];
  meta_agent_output?: string;
  memory_update_counts?: Record<string, Record<string, number>>;
}

interface AssistantMessage {
  timestamp: string;
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: string[];
}

interface MemoryAgentTrace {
  id: string;
  agent_id?: string;
  agent_name?: string;
  agent_type?: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  success?: boolean;
  error_message?: string;
  triggered_memory_types?: string[];
  assistant_messages?: AssistantMessage[];
  memory_update_counts?: Record<string, Record<string, number>>;
  parent_trace_id?: string | null;
}

interface MemoryAgentToolCall {
  id: string;
  tool_call_id?: string;
  function_name: string;
  function_args?: Record<string, unknown>;
  llm_call_id?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  cached_tokens?: number;
  total_tokens?: number;
  credit_cost?: number;
  status: string;
  started_at?: string;
  completed_at?: string;
  success?: boolean;
  response_text?: string;
  error_message?: string;
}

interface MemoryAgentTraceWithTools {
  agent_trace: MemoryAgentTrace;
  tool_calls: MemoryAgentToolCall[];
}

interface MemoryQueueTraceDetailResponse {
  trace: MemoryQueueTrace;
  agent_traces: MemoryAgentTraceWithTools[];
}

interface LiveTraceSummary {
  toolCalls: number;
  managerRuns: number;
}

interface LiveTraceStatus {
  status: string;
  success?: boolean;
}


const formatTimestamp = (value?: string) => {
  if (!value) return 'N/A';
  const ms = parseTimestampMs(value);
  if (!ms || Number.isNaN(ms)) return 'N/A';
  return new Date(ms).toLocaleString();
};

const formatDuration = (start?: string, end?: string) => {
  if (!start || !end) return 'N/A';
  const startMs = parseTimestampMs(start);
  const endMs = parseTimestampMs(end);
  if (!startMs || !endMs || Number.isNaN(startMs) || Number.isNaN(endMs)) {
    return 'N/A';
  }
  const ms = endMs - startMs;
  if (ms < 0) return 'N/A';
  if (ms < 1000) return `${ms}ms`;
  const seconds = Math.round(ms / 100) / 10;
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round((seconds % 60) * 10) / 10;
  return `${minutes}m ${remainder}s`;
};

const compactText = (value?: string | null, maxLength = 200) => {
  if (!value) return 'N/A';
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength)}...`;
};

const StatusBadge = ({ status, success }: { status?: string; success?: boolean }) => {
  if (status === 'failed' || success === false) {
    return (
      <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 gap-1 px-2 py-0.5">
        <XCircle className="w-3 h-3" />
        {status}
      </Badge>
    );
  }
  if (status === 'completed' || success === true) {
    return (
      <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20 gap-1 px-2 py-0.5">
        <CheckCircle2 className="w-3 h-3" />
        {status}
      </Badge>
    );
  }
  if (status === 'processing' || status === 'queued') {
    return (
      <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20 gap-1 px-2 py-0.5 animate-pulse">
        <Clock className="w-3 h-3" />
        {status}
      </Badge>
    );
  }
  return (
    <Badge variant="secondary" className="px-2 py-0.5">
      {status}
    </Badge>
  );
};

const MEMORY_TYPE_CONFIG: Record<string, { style: string; icon: React.ReactNode; color: string }> = {
  core: { 
    style: 'border-sky-500/30 bg-sky-500/5 text-sky-400', 
    icon: <Cpu className="w-3.5 h-3.5" />,
    color: 'text-sky-400'
  },
  episodic: { 
    style: 'border-amber-500/30 bg-amber-500/5 text-amber-400', 
    icon: <Brain className="w-3.5 h-3.5" />,
    color: 'text-amber-400'
  },
  semantic: { 
    style: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400', 
    icon: <Book className="w-3.5 h-3.5" />,
    color: 'text-emerald-400'
  },
  procedural: { 
    style: 'border-indigo-500/30 bg-indigo-500/5 text-indigo-400', 
    icon: <Wrench className="w-3.5 h-3.5" />,
    color: 'text-indigo-400'
  },
  resource: { 
    style: 'border-fuchsia-500/30 bg-fuchsia-500/5 text-fuchsia-400', 
    icon: <Layers className="w-3.5 h-3.5" />,
    color: 'text-fuchsia-400'
  },
  knowledge: { 
    style: 'border-rose-500/30 bg-rose-500/5 text-rose-400', 
    icon: <Database className="w-3.5 h-3.5" />,
    color: 'text-rose-400'
  },
};

const getMemoryTypeConfig = (memoryType?: string) =>
  MEMORY_TYPE_CONFIG[memoryType ?? ''] ?? {
    style: 'border-slate-500/30 bg-slate-500/5 text-slate-400',
    icon: <AlertCircle className="w-3.5 h-3.5" />,
    color: 'text-slate-400'
  };

const formatMemoryTypeLabel = (value: string) => value.replace(/_/g, ' ');

const inferMemoryType = (value?: string) => {
  if (!value) return undefined;
  const lower = value.toLowerCase();
  if (lower.includes('knowledge')) return 'knowledge';
  if (lower.includes('episodic')) return 'episodic';
  if (lower.includes('semantic')) return 'semantic';
  if (lower.includes('procedural')) return 'procedural';
  if (lower.includes('resource')) return 'resource';
  if (lower.includes('core')) return 'core';
  return undefined;
};

const normalizeUserToken = (value: string) => value.replace(/_/g, '-');

const renderCounts = (counts?: Record<string, Record<string, number>>, className?: string) => {
  if (!counts || Object.keys(counts).length === 0) {
    return <div className={cn("text-xs text-muted-foreground italic px-1", className)}>No updates recorded.</div>;
  }
  return (
    <div className={cn("flex flex-row flex-wrap items-center gap-2 pb-1", className)}>
      {Object.entries(counts).map(([memoryType, operations]) => {
        const config = getMemoryTypeConfig(memoryType);
        return (
            <div
            key={memoryType}
            className={cn(
              "flex flex-col items-start gap-1.5 rounded-md border px-2 py-1.5 shrink-0 min-w-[90px]",
              config.style
            )}
          >
            <div className="flex items-center gap-1.5 font-bold text-[10px] whitespace-nowrap border-b border-white/5 w-full pb-1">
              {config.icon}
              <span className="capitalize">{formatMemoryTypeLabel(memoryType)}</span>
            </div>
            <div className="flex flex-wrap gap-1 items-center">
              {Object.entries(operations).length > 0 ? (
                Object.entries(operations).map(([op, count]) => (
                  <span
                    key={op}
                    className="text-[9px] bg-white/10 px-1.5 py-0.5 rounded border border-white/5 font-medium whitespace-nowrap"
                  >
                    {op}: {count}
                  </span>
                ))
              ) : (
                <span className="text-[9px] opacity-50 italic">None</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const TRIGGER_TOOL_NAMES = new Set([
  'trigger_memory_update',
  'trigger_memory_update_with_instruction',
]);

const isTriggerToolCall = (toolCall: MemoryAgentToolCall) =>
  TRIGGER_TOOL_NAMES.has(toolCall.function_name);

const parseTimestampMs = (value?: string) => {
  if (!value) return Number.NaN;
  const trimmed = value.trim();
  const microsecondsNormalized = trimmed.replace(/\.(\d{3})\d+/, '.$1');
  const normalized = microsecondsNormalized.includes('T')
    ? microsecondsNormalized
    : microsecondsNormalized.replace(' ', 'T');
  const hasTimezone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(normalized);
  if (hasTimezone) {
    const tzMs = Date.parse(normalized);
    if (!Number.isNaN(tzMs)) return tzMs;
  }
  const utcMs = Date.parse(`${normalized}Z`);
  if (!Number.isNaN(utcMs)) return utcMs;
  const localMs = Date.parse(normalized);
  if (!Number.isNaN(localMs)) return localMs;
  const rawMs = Date.parse(microsecondsNormalized);
  return Number.isNaN(rawMs) ? Number.NaN : rawMs;
};

const extractLatestActivityMs = (detail: MemoryQueueTraceDetailResponse) => {
  const timestamps: number[] = [];
  const pushTime = (value?: string) => {
    const ms = parseTimestampMs(value);
    if (!Number.isNaN(ms)) {
      timestamps.push(ms);
    }
  };

  pushTime(detail.trace.queued_at);
  pushTime(detail.trace.started_at);
  pushTime(detail.trace.completed_at);

  detail.agent_traces.forEach((trace) => {
    pushTime(trace.agent_trace.started_at);
    pushTime(trace.agent_trace.completed_at);
    trace.tool_calls.forEach((call) => {
      pushTime(call.started_at);
      pushTime(call.completed_at);
    });
  });

  return timestamps.length > 0 ? Math.max(...timestamps) : Number.NaN;
};

const buildTraceSignature = (detail: MemoryQueueTraceDetailResponse) => {
  const toolCallSignature = detail.agent_traces
    .flatMap((trace) =>
      trace.tool_calls.map(
        (call) => `${call.id}:${call.status}:${call.success ?? 'unknown'}`
      )
    )
    .join('|');
  const agentSignature = detail.agent_traces
    .map(
      (trace) =>
        `${trace.agent_trace.id}:${trace.agent_trace.status}:${trace.agent_trace.success ?? 'unknown'}`
    )
    .join('|');
  const updateSignature = detail.agent_traces
    .map((trace) => JSON.stringify(trace.agent_trace.memory_update_counts ?? {}))
    .join('|');
  return [
    detail.trace.status,
    detail.trace.success ?? 'unknown',
    detail.trace.error_message ?? '',
    detail.agent_traces.length,
    toolCallSignature,
    agentSignature,
    updateSignature,
  ].join('::');
};

const resolveActivityMs = (primary?: number, fallback?: number) => {
  if (!primary || Number.isNaN(primary)) return fallback;
  if (!fallback || Number.isNaN(fallback)) return primary;
  const now = Date.now();
  const primaryDelta = Math.abs(now - primary);
  const fallbackDelta = Math.abs(now - fallback);
  return primaryDelta <= fallbackDelta ? primary : fallback;
};

const aggregateUpdateCounts = (traces: MemoryAgentTraceWithTools[]) => {
  const totals: Record<string, Record<string, number>> = {};
  traces.forEach(({ agent_trace }) => {
    const counts = agent_trace.memory_update_counts;
    if (!counts) return;
    Object.entries(counts).forEach(([memoryType, operations]) => {
      if (!totals[memoryType]) {
        totals[memoryType] = {};
      }
      Object.entries(operations).forEach(([op, count]) => {
        totals[memoryType][op] = (totals[memoryType][op] ?? 0) + count;
      });
    });
  });
  return Object.keys(totals).length > 0 ? totals : undefined;
};

const buildLiveSummary = (detail: MemoryQueueTraceDetailResponse): LiveTraceSummary => {
  const toolCalls = detail.agent_traces.reduce(
    (sum, trace) => sum + trace.tool_calls.length,
    0
  );
  const managerRuns = detail.agent_traces.filter(
    (trace) => !!trace.agent_trace.parent_trace_id
  ).length;
  return { toolCalls, managerRuns };
};

export const MemoryTraces: React.FC = () => {
  const { selectedUser, users, setSelectedUser, isLoading: usersLoading } = useWorkspace();
  const [searchParams, setSearchParams] = useSearchParams();
  const [traces, setTraces] = useState<MemoryQueueTrace[]>([]);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [detail, setDetail] = useState<MemoryQueueTraceDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedToolCalls, setExpandedToolCalls] = useState<Record<string, boolean>>({});
  const [expandedToolCallErrors, setExpandedToolCallErrors] = useState<Record<string, boolean>>({});
  const [expandedMemoryUpdates, setExpandedMemoryUpdates] = useState<Record<string, boolean>>({});
  const [expandedTraceErrors, setExpandedTraceErrors] = useState<Record<string, boolean>>({});
  const [liveTraceSummaries, setLiveTraceSummaries] = useState<Record<string, LiveTraceSummary>>({});
  const [liveTraceActivity, setLiveTraceActivity] = useState<Record<string, number>>({});
  const [liveTraceStatus, setLiveTraceStatus] = useState<Record<string, LiveTraceStatus>>({});
  const liveTraceSignaturesRef = React.useRef<Record<string, string>>({});
  const [liveTraceChangeMs, setLiveTraceChangeMs] = useState<Record<string, number>>({});
  const waitLogRef = React.useRef<Record<string, number>>({});
  const userParamAppliedRef = React.useRef(false);
  const lastUserParamRef = React.useRef<string | null>(null);
  const userIdParam = searchParams.get('user_id');
  const searchParamsString = searchParams.toString();
  const matchingUserFromParam = useMemo(() => {
    if (!userIdParam || users.length === 0) return null;
    const normalizedParam = normalizeUserToken(userIdParam);
    return (
      users.find((user) => user.id === userIdParam || user.name === userIdParam) ||
      users.find(
        (user) =>
          normalizeUserToken(user.id) === normalizedParam ||
          normalizeUserToken(user.name) === normalizedParam
      ) ||
      null
    );
  }, [userIdParam, users]);
  const userParamValid = !!matchingUserFromParam;

  useEffect(() => {
    if (usersLoading || !userIdParam || users.length === 0) return;
    if (!matchingUserFromParam) return;
    const userParamChanged = userIdParam !== lastUserParamRef.current;
    if (userParamChanged || !userParamAppliedRef.current) {
      if (selectedUser?.id !== matchingUserFromParam.id) {
        setSelectedUser(matchingUserFromParam);
      }
      userParamAppliedRef.current = true;
    }
    lastUserParamRef.current = userIdParam;
  }, [usersLoading, userIdParam, users, matchingUserFromParam, selectedUser?.id, setSelectedUser]);

  useEffect(() => {
    if (!selectedUser) return;
    const deferUserParamUpdate = !!userIdParam && userParamValid && !userParamAppliedRef.current;
    const nextUserParam = deferUserParamUpdate ? userIdParam : selectedUser.id;
    const params = new URLSearchParams(searchParamsString);
    params.set('user_id', nextUserParam);
    if (params.toString() !== searchParamsString) {
      setSearchParams(params, { replace: true });
    }
  }, [selectedUser?.id, userIdParam, userParamValid, searchParamsString, setSearchParams]);

  const fetchTraces = useCallback(async () => {
    if (!selectedUser) return;
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get('/memory/queue-traces', {
        params: { user_id: selectedUser.id, limit: 50 },
      });
      const data = response.data as MemoryQueueTrace[];
      setTraces(data);
      
      // Keep selection only if it still exists in the refreshed data, otherwise condense all
      setSelectedTraceId(prev => {
        if (prev && data.some((trace) => trace.id === prev)) {
          return prev;
        }
        return null;
      });
    } catch (err) {
      console.error('Failed to load queue traces', err);
      setError('Failed to load queue traces. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [selectedUser]); // Removed selectedTraceId from dependencies

  const fetchTraceDetail = useCallback(async (traceId: string, options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;
    if (!silent) {
      setDetailLoading(true);
    }
    try {
      const response = await apiClient.get(`/memory/queue-traces/${traceId}`);
      setDetail(response.data);
    } catch (err) {
      console.error('Failed to load trace detail', err);
      setError('Failed to load trace detail. Please try again.');
    } finally {
      if (!silent) {
        setDetailLoading(false);
      }
    }
  }, []);

  const fetchLiveSummaries = useCallback(async (traceIds: string[]) => {
    if (traceIds.length === 0) return;
    const responseTimestamp = Date.now();
    const responses = await Promise.all(
      traceIds.map(async (traceId) => {
        try {
          const response = await apiClient.get(`/memory/queue-traces/${traceId}`);
          return { traceId, detail: response.data as MemoryQueueTraceDetailResponse };
        } catch (error) {
          console.error('Failed to load live trace summary', error);
          return null;
        }
      })
    );
    setLiveTraceSummaries((prev) => {
      const next = { ...prev };
      responses.forEach((result) => {
        if (!result) return;
        next[result.traceId] = buildLiveSummary(result.detail);
      });
      return next;
    });
    const previousSignatures = liveTraceSignaturesRef.current;
    const nextSignatures = { ...previousSignatures };
    const changed: string[] = [];
    responses.forEach((result) => {
      if (!result) return;
      const signature = buildTraceSignature(result.detail);
      const previousSignature = previousSignatures[result.traceId];
      if (previousSignature && previousSignature !== signature) {
        changed.push(result.traceId);
      }
      nextSignatures[result.traceId] = signature;
    });
    liveTraceSignaturesRef.current = nextSignatures;
    if (changed.length > 0) {
      setLiveTraceChangeMs((prevChanges) => {
        const nextChanges = { ...prevChanges };
        changed.forEach((traceId) => {
          nextChanges[traceId] = responseTimestamp;
        });
        return nextChanges;
      });
    }
    setLiveTraceActivity((prev) => {
      const next = { ...prev };
      responses.forEach((result) => {
        if (!result) return;
        const latestActivityMs = extractLatestActivityMs(result.detail);
        if (!Number.isNaN(latestActivityMs)) {
          next[result.traceId] = latestActivityMs;
        }
      });
      return next;
    });
    setLiveTraceStatus((prev) => {
      const next = { ...prev };
      responses.forEach((result) => {
        if (!result) return;
        next[result.traceId] = {
          status: result.detail.trace.status,
          success: result.detail.trace.success,
        };
      });
      return next;
    });
  }, []);

  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  useEffect(() => {
    if (selectedTraceId) {
      fetchTraceDetail(selectedTraceId);
    } else {
      setDetail(null);
    }
  }, [selectedTraceId, fetchTraceDetail]);

  useEffect(() => {
    if (!detail) return;
    const detailSignature = buildTraceSignature(detail);
    const previousSignature = liveTraceSignaturesRef.current[detail.trace.id];
    if (previousSignature !== detailSignature) {
      if (previousSignature) {
        setLiveTraceChangeMs((prevChanges) => ({
          ...prevChanges,
          [detail.trace.id]: Date.now(),
        }));
      }
      liveTraceSignaturesRef.current = {
        ...liveTraceSignaturesRef.current,
        [detail.trace.id]: detailSignature,
      };
    }
    const latestActivityMs = extractLatestActivityMs(detail);
    if (!Number.isNaN(latestActivityMs)) {
      setLiveTraceSummaries((prev) => ({
        ...prev,
        [detail.trace.id]: buildLiveSummary(detail),
      }));
      setLiveTraceActivity((prev) => ({
        ...prev,
        [detail.trace.id]: latestActivityMs,
      }));
    }
    setLiveTraceSummaries((prev) => ({
      ...prev,
      [detail.trace.id]: buildLiveSummary(detail),
    }));
    setLiveTraceStatus((prev) => ({
      ...prev,
      [detail.trace.id]: {
        status: detail.trace.status,
        success: detail.trace.success,
      },
    }));
  }, [detail]);

  useEffect(() => {
    const processingTraceIds = traces
      .filter((trace) => {
        const liveStatus = liveTraceStatus[trace.id]?.status ?? trace.status;
        return ['queued', 'processing'].includes(liveStatus);
      })
      .map((trace) => trace.id);
    const detailStatus = detail?.trace?.status;
    const detailProcessing = detailStatus && ['queued', 'processing'].includes(detailStatus);
    if (processingTraceIds.length === 0 && !detailProcessing) return;

    const intervalId = window.setInterval(() => {
      if (document.hidden) return;
      const nowMs = Date.now();
      const traceById = new Map(traces.map((trace) => [trace.id, trace]));
      processingTraceIds.forEach((traceId) => {
        const trace = traceById.get(traceId);
        const fallbackActivityMs = parseTimestampMs(
          trace?.started_at ?? trace?.queued_at
        );
        const lastActivityMs =
          liveTraceChangeMs[traceId] ??
          resolveActivityMs(liveTraceActivity[traceId], fallbackActivityMs);
        if (!lastActivityMs || Number.isNaN(lastActivityMs)) {
          return;
        }
        const waitSeconds = Math.floor((nowMs - lastActivityMs) / 1000);
        const lastLogged = waitLogRef.current[traceId];
        if (waitSeconds < 0) {
          if (lastLogged !== waitSeconds) {
            waitLogRef.current[traceId] = waitSeconds;
            console.log(
              `[MemoryTraces] Trace ${traceId} activity timestamp is ${Math.abs(
                waitSeconds
              )}s in the future (queued_at=${trace?.queued_at ?? 'N/A'}, started_at=${trace?.started_at ?? 'N/A'}).`
            );
          }
          return;
        }
        if (lastLogged === undefined || waitSeconds - lastLogged >= 10) {
          waitLogRef.current[traceId] = waitSeconds;
          console.log(
            `[MemoryTraces] Trace ${traceId} waiting ${waitSeconds}s (queued_at=${trace?.queued_at ?? 'N/A'}, started_at=${trace?.started_at ?? 'N/A'}).`
          );
        }
      });
      if (selectedTraceId) {
        fetchTraceDetail(selectedTraceId, { silent: true });
      }
      const idsToPoll = processingTraceIds.filter((traceId) => traceId !== selectedTraceId);
      fetchLiveSummaries(idsToPoll);
    }, 3000);

    return () => window.clearInterval(intervalId);
  }, [
    traces,
    detail?.trace?.status,
    fetchTraceDetail,
    fetchLiveSummaries,
    selectedTraceId,
    liveTraceStatus,
  ]);

  const toggleToolCall = (toolCallId: string) => {
    setExpandedToolCalls((prev) => ({
      ...prev,
      [toolCallId]: !prev[toolCallId],
    }));
  };

  const toggleToolCallError = (toolCallId: string) => {
    setExpandedToolCallErrors((prev) => ({
      ...prev,
      [toolCallId]: !prev[toolCallId],
    }));
  };

  const toggleMemoryUpdates = (traceId: string) => {
    setExpandedMemoryUpdates((prev) => ({
      ...prev,
      [traceId]: !prev[traceId],
    }));
  };

  const toggleTraceError = (traceId: string) => {
    setExpandedTraceErrors((prev) => ({
      ...prev,
      [traceId]: !prev[traceId],
    }));
  };

  const renderToolCall = (toolCall: MemoryAgentToolCall) => {
    const expanded = expandedToolCalls[toolCall.id] ?? false;
    const isLlmRequest = toolCall.function_name === 'llm_request';
    const errorMessage = toolCall.error_message;
    const hasError = !!errorMessage;
    const previewExpanded = expandedToolCallErrors[toolCall.id] ?? false;
    const previewText =
      toolCall.status === 'failed' && errorMessage
        ? errorMessage
        : toolCall.response_text || errorMessage;
    const responseLabel = hasError && !toolCall.response_text ? 'Error' : 'Response';
    const usageParts: string[] = [];
    if (toolCall.prompt_tokens !== undefined && toolCall.prompt_tokens !== null) {
      usageParts.push(`prompt ${toolCall.prompt_tokens}`);
    }
    if (toolCall.completion_tokens !== undefined && toolCall.completion_tokens !== null) {
      usageParts.push(`completion ${toolCall.completion_tokens}`);
    }
    if (toolCall.cached_tokens !== undefined && toolCall.cached_tokens !== null) {
      usageParts.push(`cached ${toolCall.cached_tokens}`);
    }
    if (toolCall.total_tokens !== undefined && toolCall.total_tokens !== null) {
      usageParts.push(`total ${toolCall.total_tokens}`);
    }
    const usageSummary = usageParts.length > 0 ? `Tokens: ${usageParts.join(' · ')}` : null;
    const costSummary =
      toolCall.credit_cost !== undefined && toolCall.credit_cost !== null
        ? `Cost: $${toolCall.credit_cost.toFixed(6)}`
        : null;
    return (
      <div
        key={toolCall.id}
        className={cn(
          "rounded-lg border bg-card/50 overflow-hidden shadow-sm",
          isLlmRequest && "border-primary/40 bg-primary/5"
        )}
      >
        <div 
          className="flex items-center justify-between p-3 cursor-pointer hover:bg-muted/30 transition-colors"
          onClick={() => toggleToolCall(toolCall.id)}
        >
          <div className="flex flex-col gap-0.5">
            <div className="font-semibold text-xs tracking-tight flex items-center gap-2">
              <span>{isLlmRequest ? 'LLM Request' : toolCall.function_name}</span>
              {isLlmRequest && (
                <Badge variant="secondary" className="text-[9px] uppercase tracking-widest">
                  LLM
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Clock className="w-3 h-3" />
              {formatDuration(toolCall.started_at, toolCall.completed_at)}
            </div>
            {(usageSummary || costSummary) && (
              <div className="text-[10px] text-muted-foreground">
                {[usageSummary, costSummary].filter(Boolean).join(' · ')}
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={toolCall.status} success={toolCall.success} />
            {expanded ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
          </div>
        </div>
        
        {expanded && (
          <div className="border-t bg-muted/10 p-4 space-y-4">
            {(usageSummary || costSummary) && (
              <div>
                <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1.5 flex items-center gap-1.5">
                  <div className="w-1 h-3 bg-primary/40 rounded-full" />
                  Usage
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {[usageSummary, costSummary].filter(Boolean).join(' · ') || 'N/A'}
                </div>
              </div>
            )}
            <div>
              <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1.5 flex items-center gap-1.5">
                <div className="w-1 h-3 bg-primary/40 rounded-full" />
                Arguments
              </div>
              <pre className="text-[11px] font-mono bg-black/20 p-3 rounded-md overflow-x-auto border border-white/5">
                {toolCall.function_args ? JSON.stringify(toolCall.function_args, null, 2) : 'N/A'}
              </pre>
            </div>
            <div>
              <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1.5 flex items-center gap-1.5">
                <div className="w-1 h-3 bg-primary/40 rounded-full" />
                {responseLabel}
              </div>
              <pre className="text-[11px] font-mono bg-black/20 p-3 rounded-md overflow-x-auto border border-white/5 whitespace-pre-wrap">
                {toolCall.response_text ? toolCall.response_text : toolCall.error_message || 'N/A'}
              </pre>
            </div>
          </div>
        )}
        {!expanded && previewText && (
          <div className="px-3 pb-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] text-muted-foreground line-clamp-1 bg-black/10 px-2 py-1 rounded border border-white/5 italic">
                {compactText(previewText, 100)}
              </div>
              {hasError && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-[10px] text-red-400 hover:text-red-300"
                  onClick={() => toggleToolCallError(toolCall.id)}
                >
                  {previewExpanded ? 'Hide Error' : 'Show Error'}
                </Button>
              )}
            </div>
            {hasError && previewExpanded && (
              <pre className="text-[11px] font-mono bg-red-500/5 p-3 rounded-md overflow-x-auto border border-red-500/10 whitespace-pre-wrap text-red-200/90">
                {errorMessage}
              </pre>
            )}
          </div>
        )}
      </div>
    );
  };

  const traceGroups = useMemo(() => {
    const agentTraces = detail?.agent_traces ?? [];
    const metaTraces: MemoryAgentTraceWithTools[] = [];
    const metaIds = new Set<string>();

    for (const trace of agentTraces) {
      if (!trace.agent_trace.parent_trace_id) {
        metaTraces.push(trace);
        metaIds.add(trace.agent_trace.id);
      }
    }

    const childTracesByParent = new Map<string, MemoryAgentTraceWithTools[]>();
    const orphanTraces: MemoryAgentTraceWithTools[] = [];

    for (const trace of agentTraces) {
      const parentId = trace.agent_trace.parent_trace_id;
      if (!parentId) {
        continue;
      }
      if (!metaIds.has(parentId)) {
        orphanTraces.push(trace);
        continue;
      }
      const bucket = childTracesByParent.get(parentId) ?? [];
      bucket.push(trace);
      childTracesByParent.set(parentId, bucket);
    }

    return { metaTraces, childTracesByParent, orphanTraces };
  }, [detail]);

  const renderMemoryAgentTraceContent = (traceWithTools: MemoryAgentTraceWithTools) => {
    const { agent_trace, tool_calls } = traceWithTools;

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 bg-muted/20 p-3 rounded-lg border border-white/5">
          <div className="space-y-0.5 overflow-hidden">
            <div className="text-[10px] uppercase font-bold text-muted-foreground">Type</div>
            <div className="text-xs font-medium truncate" title={agent_trace.agent_type}>{agent_trace.agent_type || 'N/A'}</div>
          </div>
          <div className="space-y-0.5">
            <div className="text-[10px] uppercase font-bold text-muted-foreground">Status</div>
            <div className="flex">
              <StatusBadge status={agent_trace.status} success={agent_trace.success} />
            </div>
          </div>
          <div className="space-y-0.5">
            <div className="text-[10px] uppercase font-bold text-muted-foreground">Duration</div>
            <div className="text-xs font-medium flex items-center gap-1.5">
              <Clock className="w-3 h-3" />
              {formatDuration(agent_trace.started_at, agent_trace.completed_at)}
            </div>
          </div>
          <div className="space-y-0.5 overflow-hidden">
            <div className="text-[10px] uppercase font-bold text-muted-foreground">Managers</div>
            <div className="text-xs font-medium truncate" title={(() => {
                  const triggered = new Set(agent_trace.triggered_memory_types || []);
                  Object.keys(agent_trace.memory_update_counts || {}).forEach(t => triggered.add(t));
                  return Array.from(triggered).join(', ');
                })()}>
              {(() => {
                const triggered = new Set(agent_trace.triggered_memory_types || []);
                Object.keys(agent_trace.memory_update_counts || {}).forEach(t => triggered.add(t));
                return triggered.size > 0 ? Array.from(triggered).join(', ') : 'N/A';
              })()}
            </div>
          </div>
        </div>

        {agent_trace.error_message && (
          <div className="rounded-md border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-500 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            {agent_trace.error_message}
          </div>
        )}

        <div className="space-y-2">
          <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5 px-1">
            <ArrowRight className="w-3 h-3 text-primary/60" />
            Updates Summary
          </div>
          {renderCounts(agent_trace.memory_update_counts, "flex-wrap")}
        </div>

        {agent_trace.assistant_messages && agent_trace.assistant_messages.length > 0 && (
          <div className="space-y-2">
            <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5 px-1">
              <ArrowRight className="w-3 h-3 text-primary/60" />
              Assistant Output
            </div>
            <div className="grid gap-2">
              {agent_trace.assistant_messages.map((message, index) => (
                <div key={`${agent_trace.id}-${index}`} className="rounded-lg border bg-muted/5 p-3 space-y-1">
                  <div className="text-[10px] text-muted-foreground flex items-center gap-1.5">
                    <Clock className="w-3 h-3" />
                    {formatTimestamp(message.timestamp)}
                  </div>
                  <div className="text-xs leading-relaxed whitespace-pre-wrap">
                    {message.content || message.reasoning_content || 'N/A'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5 px-1">
            <ArrowRight className="w-3 h-3 text-primary/60" />
            Tool Calls ({tool_calls.length})
          </div>
          <div className="grid gap-2">
            {tool_calls.length === 0 ? (
              <div className="text-xs text-muted-foreground italic px-1">No tool calls recorded.</div>
            ) : (
              tool_calls.map((toolCall) => renderToolCall(toolCall))
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderMemoryUpdatesBlock = (
    groupKey: string,
    updateCounts: Record<string, Record<string, number>> | undefined,
    childTraces: MemoryAgentTraceWithTools[]
  ) => {
    const expanded = expandedMemoryUpdates[groupKey] ?? false;
    return (
      <div className="rounded-xl border bg-primary/5 p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <div className="font-bold text-sm tracking-tight">Parallel Memory Updates</div>
            <Badge variant="secondary" className="text-[10px] h-5">
              {childTraces.length} Manager{childTraces.length !== 1 ? 's' : ''}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => toggleMemoryUpdates(groupKey)}
            className="h-8 text-xs gap-1.5"
          >
            {expanded ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" />
                Hide Details
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" />
                Show Details
              </>
            )}
          </Button>
        </div>
        
        <div className="space-y-3">
          <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5">
            <ArrowRight className="w-3 h-3 text-primary/60" />
            Combined Updates
          </div>
          {renderCounts(updateCounts, "flex-wrap")}
        </div>

        {expanded && (
          <div className="pt-2 animate-in fade-in slide-in-from-top-2 duration-300">
            <Separator className="mb-4 bg-primary/10" />
            <div className="text-[10px] uppercase font-bold text-muted-foreground mb-4 flex items-center gap-1.5">
              <ArrowRight className="w-3 h-3 text-primary/60" />
              Manager Run Details
            </div>
            {childTraces.length === 0 ? (
              <div className="text-xs text-muted-foreground italic">
                No memory manager runs recorded.
              </div>
            ) : (
              <div className="flex flex-row gap-4 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-primary/20 scrollbar-track-transparent">
                {childTraces.map((childTrace) => {
                  const memoryType = inferMemoryType(
                    childTrace.agent_trace.agent_type ?? childTrace.agent_trace.agent_name
                  );
                  const config = getMemoryTypeConfig(memoryType);
                  return (
                    <div
                      key={childTrace.agent_trace.id}
                      className={cn(
                        "flex-none w-[340px] rounded-xl border-l-4 p-4 space-y-4 shadow-lg bg-card/50",
                        config.style
                      )}
                    >
                      <div className="flex items-center gap-2 border-b border-white/5 pb-2 overflow-hidden">
                        <div className={cn("p-1.5 rounded-lg bg-white/5 shrink-0", config.color)}>
                          {config.icon}
                        </div>
                        <div className="font-bold text-sm truncate" title={childTrace.agent_trace.agent_name}>
                          {childTrace.agent_trace.agent_name || 'Memory agent'}
                        </div>
                      </div>
                      {renderMemoryAgentTraceContent(childTrace)}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderMetaToolCalls = (
    toolCalls: MemoryAgentToolCall[],
    agentTrace: MemoryAgentTrace,
    childTraces: MemoryAgentTraceWithTools[]
  ) => {
    if (toolCalls.length === 0) {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5 px-1">
              <ArrowRight className="w-3 h-3 text-primary/60" />
              Tool Calls (0)
            </div>
            <div className="text-xs text-muted-foreground italic px-1">No tool calls recorded.</div>
          </div>
        </div>
      );
    }

    const triggerCalls = toolCalls.filter(isTriggerToolCall);
    const triggerStarts = triggerCalls.map((call) => ({
      call,
      startMs: parseTimestampMs(call.started_at),
    }));
    const groupedByTrigger = new Map<string, MemoryAgentTraceWithTools[]>();
    triggerCalls.forEach((call) => groupedByTrigger.set(call.id, []));
    const unassignedTraces: MemoryAgentTraceWithTools[] = [];

    childTraces.forEach((childTrace) => {
      if (triggerCalls.length === 0) {
        unassignedTraces.push(childTrace);
        return;
      }
      const firstTrigger = triggerStarts[0];
      if (!firstTrigger) {
        unassignedTraces.push(childTrace);
        return;
      }
      const childStart = parseTimestampMs(childTrace.agent_trace.started_at);
      let matchedTrigger = firstTrigger;
      if (!Number.isNaN(childStart)) {
        triggerStarts.forEach((trigger) => {
          if (!Number.isNaN(trigger.startMs) && trigger.startMs <= childStart) {
            matchedTrigger = trigger;
          }
        });
      }
      const group = groupedByTrigger.get(matchedTrigger.call.id);
      if (group) {
        group.push(childTrace);
      } else {
        unassignedTraces.push(childTrace);
      }
    });

    const orderedBlocks = toolCalls.reduce<React.ReactNode[]>((acc, toolCall) => {
      acc.push(renderToolCall(toolCall));
      if (isTriggerToolCall(toolCall)) {
        const groupedTraces = groupedByTrigger.get(toolCall.id) ?? [];
        const updateCounts = aggregateUpdateCounts(groupedTraces);
        acc.push(
          renderMemoryUpdatesBlock(`${agentTrace.id}-${toolCall.id}`, updateCounts, groupedTraces)
        );
      }
      return acc;
    }, []);

    if (triggerCalls.length > 0 && unassignedTraces.length > 0) {
      const updateCounts = aggregateUpdateCounts(unassignedTraces);
      orderedBlocks.push(
        renderMemoryUpdatesBlock(`${agentTrace.id}-unassigned`, updateCounts, unassignedTraces)
      );
    }

    return (
      <div className="space-y-3">
        <div className="text-[10px] uppercase font-bold text-muted-foreground flex items-center gap-1.5 px-1">
          <ArrowRight className="w-3 h-3 text-primary/60" />
          Execution Steps
        </div>
        <div className="space-y-3">{orderedBlocks}</div>
      </div>
    );
  };

  if (!selectedUser) {
    return (
      <div className="text-sm text-muted-foreground">
        Select a user to view memory traces.
      </div>
    );
  }

  const detailErrorMessage = detail?.trace?.error_message ?? null;

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Memory Traces
          </h2>
          <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
            Inspect queue requests, memory updates, and parallel execution traces for <span className="font-semibold text-primary">{selectedUser.name}</span>.
          </p>
        </div>
        <Button 
          onClick={fetchTraces} 
          disabled={loading}
          className="shadow-lg shadow-primary/20 transition-all hover:shadow-primary/30"
        >
          <RefreshCcw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} />
          Refresh Traces
        </Button>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-500 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          {error}
        </div>
      )}

      <div className="grid gap-6">
        {loading && traces.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 border rounded-2xl bg-muted/5 border-dashed">
            <RefreshCcw className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
            <p className="text-muted-foreground font-medium text-sm">Loading traces...</p>
          </div>
        ) : traces.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 border rounded-2xl bg-muted/5 border-dashed">
            <AlertCircle className="h-8 w-8 text-muted-foreground mb-4" />
            <p className="text-muted-foreground font-medium text-sm">No traces found for this user.</p>
          </div>
        ) : (
          traces.map((trace) => {
            const isSelected = trace.id === selectedTraceId;
            const liveStatus = liveTraceStatus[trace.id];
            const effectiveStatus = liveStatus?.status ?? trace.status;
            const effectiveSuccess = liveStatus?.success ?? trace.success;
            const displayStatus = effectiveStatus;
            const displaySuccess = effectiveSuccess;
            const liveSummary = liveTraceSummaries[trace.id];
            const canExpand =
              effectiveStatus === 'completed' ||
              effectiveStatus === 'failed' ||
              (!!liveSummary && (liveSummary.toolCalls > 0 || liveSummary.managerRuns > 0));
            const showLiveSummary =
              ['queued', 'processing'].includes(effectiveStatus) && liveSummary;
            
            return (
              <Card 
                key={trace.id} 
                className={cn(
                  "overflow-hidden transition-all duration-300 border-2",
                  isSelected ? "ring-2 ring-primary/20 border-primary/40 shadow-xl" : "border-transparent hover:border-muted-foreground/20 hover:shadow-md"
                )}
              >
                <div 
                  className={cn(
                    "p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors",
                    canExpand ? "cursor-pointer" : "cursor-default",
                    isSelected ? "bg-primary/5" : canExpand ? "hover:bg-muted/30" : ""
                  )}
                  onClick={() => {
                    if (!canExpand) return;
                    if (isSelected) {
                      setSelectedTraceId(null);
                    } else {
                      setSelectedTraceId(trace.id);
                    }
                  }}
                >
                  <div className="flex flex-col gap-3 min-w-0 flex-grow">
                    <div className="flex items-center gap-4 shrink-0">
                      <div className={cn(
                        "p-3 rounded-xl shrink-0",
                        displayStatus === 'completed' ? "bg-emerald-500/10 text-emerald-500" : 
                        displayStatus === 'failed' ? "bg-red-500/10 text-red-500" : "bg-blue-500/10 text-blue-500"
                      )}>
                        {displayStatus === 'completed' ? <CheckCircle2 className="w-6 h-6" /> : 
                         displayStatus === 'failed' ? <XCircle className="w-6 h-6" /> : <Clock className="w-6 h-6" />}
                      </div>
                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-lg tracking-tight whitespace-nowrap">
                            {formatTimestamp(trace.queued_at)}
                          </span>
                          <StatusBadge status={displayStatus} success={displaySuccess} />
                        </div>
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground font-medium overflow-hidden">
                          <span className="flex items-center gap-1.5 shrink-0">
                            <Clock className="w-3.5 h-3.5" />
                            {formatDuration(trace.started_at, trace.completed_at)}
                          </span>
                          <span className="flex items-center gap-1.5 shrink-0">
                            <Layers className="w-3.5 h-3.5" />
                            {trace.message_count ?? 0} Message(s)
                          </span>
                          <span className="flex items-center gap-1.5 shrink-0 bg-black/10 px-1.5 py-0.5 rounded border border-white/5">
                            <div className="w-1 h-1 bg-emerald-500 rounded-full" />
                            <code className="font-mono text-[10px] text-muted-foreground/60 tracking-tight">
                              {trace.id}
                            </code>
                          </span>
                          {showLiveSummary && (
                            <>
                              <span className="flex items-center gap-1.5 shrink-0">
                                <Wrench className="w-3.5 h-3.5" />
                                {liveSummary.toolCalls} Tool Call(s)
                              </span>
                              <span className="flex items-center gap-1.5 shrink-0">
                                <Layers className="w-3.5 h-3.5" />
                                {liveSummary.managerRuns} Manager Run(s)
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    {trace.memory_update_counts && Object.keys(trace.memory_update_counts).length > 0 && (
                      <div className="pl-[72px]">
                        {renderCounts(trace.memory_update_counts, "flex-wrap justify-start gap-1.5")}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-2 ml-auto shrink-0">
                    {canExpand && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className={cn(
                          "rounded-full h-10 w-10 shrink-0 transition-all duration-200",
                          isSelected ? "bg-primary/20 text-primary" : "hover:bg-muted-foreground/10"
                        )}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (isSelected) {
                            setSelectedTraceId(null);
                          } else {
                            setSelectedTraceId(trace.id);
                          }
                        }}
                      >
                        {isSelected ? <ChevronUp className="w-6 h-6" /> : <ChevronDown className="w-6 h-6" />}
                      </Button>
                    )}
                  </div>
                </div>

                {isSelected && (
                  <div className="border-t bg-card animate-in slide-in-from-top-4 duration-500">
                    <CardContent className="p-6 space-y-8">
                      {detailLoading ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-3">
                          <RefreshCcw className="h-6 w-6 animate-spin text-primary" />
                          <p className="text-sm text-muted-foreground font-medium">Fetching execution details...</p>
                        </div>
                      ) : !detail ? (
                        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                          <AlertCircle className="h-6 w-6 mb-2" />
                          <p className="text-sm">Failed to load execution detail.</p>
                        </div>
                      ) : (
                        <div className="space-y-6">
                          {detailErrorMessage && (
                            <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 space-y-3">
                              <div className="flex items-center justify-between gap-3">
                                <div className="flex items-center gap-2 text-sm font-semibold text-red-500">
                                  <AlertCircle className="w-4 h-4" />
                                  Queue Error
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7 text-xs"
                                  onClick={() => toggleTraceError(detail.trace.id)}
                                >
                                  {expandedTraceErrors[detail.trace.id] ? 'Hide Error' : 'Show Error'}
                                </Button>
                              </div>
                              {expandedTraceErrors[detail.trace.id] ? (
                                <pre className="text-xs text-red-200/90 whitespace-pre-wrap bg-black/20 rounded-md p-3 border border-red-500/10">
                                  {detailErrorMessage}
                                </pre>
                              ) : (
                                <div className="text-xs text-red-200/80 italic">
                                  {compactText(detailErrorMessage, 160)}
                                </div>
                              )}
                            </div>
                          )}
                          <div className="space-y-4">
                            <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest flex items-center gap-2">
                              <div className="w-2 h-2 bg-primary rounded-full shadow-sm shadow-primary/40" />
                              Execution Flow
                            </div>
                            
                            <div className="space-y-4">
                              {detail.agent_traces && detail.agent_traces.length > 0 ? (
                                <>
                                  {traceGroups.metaTraces.map(({ agent_trace, tool_calls }) => {
                                    const childTraces = traceGroups.childTracesByParent.get(agent_trace.id) ?? [];
                                    return (
                                      <div key={agent_trace.id} className="space-y-4">
                                        <div className="rounded-xl border bg-muted/5 p-5 space-y-4">
                                          <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                              <div className="p-2 rounded-lg bg-primary/10 text-primary">
                                                <Cpu className="w-5 h-5" />
                                              </div>
                                              <div className="font-bold text-base">
                                                {agent_trace.agent_name || 'Meta Memory Agent'}
                                              </div>
                                            </div>
                                            <StatusBadge status={agent_trace.status} success={agent_trace.success} />
                                          </div>
                                          
                                          <div className="grid grid-cols-2 gap-x-4 gap-y-3 bg-muted/20 p-3 rounded-lg border border-white/5">
                                            <div className="space-y-0.5 overflow-hidden">
                                              <div className="text-[10px] uppercase font-bold text-muted-foreground">Type</div>
                                              <div className="text-xs font-medium truncate" title={agent_trace.agent_type}>{agent_trace.agent_type || 'N/A'}</div>
                                            </div>
                                            <div className="space-y-0.5">
                                              <div className="text-[10px] uppercase font-bold text-muted-foreground">Status</div>
                                              <div className="flex">
                                                <StatusBadge status={agent_trace.status} success={agent_trace.success} />
                                              </div>
                                            </div>
                                            <div className="space-y-0.5">
                                              <div className="text-[10px] uppercase font-bold text-muted-foreground">Duration</div>
                                              <div className="text-xs font-medium flex items-center gap-1.5">
                                                <Clock className="w-3 h-3" />
                                                {formatDuration(agent_trace.started_at, agent_trace.completed_at)}
                                              </div>
                                            </div>
                                            <div className="space-y-0.5 overflow-hidden">
                                              <div>Managers</div>
                                              <div className="text-foreground">
                                                {(() => {
                                                  const triggered = new Set(agent_trace.triggered_memory_types || []);
                                                  Object.keys(agent_trace.memory_update_counts || {}).forEach(t => triggered.add(t));
                                                  return triggered.size;
                                                })()} Triggered
                                              </div>
                                            </div>
                                          </div>

                                          {agent_trace.assistant_messages && agent_trace.assistant_messages.length > 0 && (
                                            <div className="bg-background/50 rounded-lg p-3 border border-white/5 space-y-2">
                                              <div className="text-[10px] uppercase font-bold text-muted-foreground">Assistant Output</div>
                                              {agent_trace.assistant_messages.map((message, idx) => (
                                                <div key={idx} className="text-sm leading-relaxed italic text-foreground/80">
                                                  "{message.content || message.reasoning_content}"
                                                </div>
                                              ))}
                                            </div>
                                          )}

                                          {renderMetaToolCalls(tool_calls, agent_trace, childTraces)}
                                        </div>
                                      </div>
                                    );
                                  })}
                                  
                                  {traceGroups.orphanTraces.map((traceWithTools) => (
                                    <div key={traceWithTools.agent_trace.id} className="rounded-xl border bg-muted/5 p-5">
                                      <div className="flex items-center gap-3 mb-4">
                                        <div className="p-2 rounded-lg bg-primary/10 text-primary">
                                          <Wrench className="w-5 h-5" />
                                        </div>
                                        <div className="font-bold text-base">
                                          {traceWithTools.agent_trace.agent_name || 'Memory Agent'}
                                        </div>
                                      </div>
                                      {renderMemoryAgentTraceContent(traceWithTools)}
                                    </div>
                                  ))}
                                </>
                              ) : (
                                <div className="py-12 text-center border border-dashed rounded-xl text-muted-foreground">
                                  No agent traces recorded.
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </div>
                )}
              </Card>
            );
          })
        )}
      </div>
    </div>
  );
};

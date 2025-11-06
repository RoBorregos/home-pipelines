import React, { useEffect, useState } from 'react';
import { ChevronDown, Play, Trash } from 'lucide-react';
import { ws, ws2 } from '../../pages/RunPage';

const CellRunner = ({ tag }) => {
    // Keep websocket handlers as-is (visual-only changes requested).
    ws2.onmessage = (msg) => {
        // message content is not altered; we only append for display
        console.log("LOGPPP:", msg.data);
        setLogs((s) => [...s, msg.data]);
        if(msg.data.includes("Finished")){
            setRunning(false);
        }
    };

    const wsOnOpen =
    ws.onopen = (tag) => {
        ws.send(JSON.stringify({
        action: "run",
        tags: [tag]
        }));
    };

  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);

  const handleRun = () => {
    setRunning(true);
    setOpen(true);
    // wsOnOpen in repo is implemented as a helper that sends a run message with the tag
    try {
      if (typeof wsOnOpen === 'function') wsOnOpen(tag);
      else if (typeof ws.onopen === 'function') ws.onopen(tag);
    } catch (e) {
      setLogs((s) => [...s, `Error sending run request: ${e.message}`]);
      setRunning(false);
    }
  };

  const handleClear = () => setLogs([]);

  // Small helper to compute visual level from log text (purely visual)
  const detectLevel = (text) => {
    if (/error|failed|exception/i.test(text)) return 'error';
    if (/warn|warning/i.test(text)) return 'warn';
    return 'info';
  };

  // Render logs as a single scrollable block (user requested single block, not a box per log)
  const handleCopyAll = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'));
    } catch (e) {
      // ignore clipboard errors (UX only)
    }
  };

  const LogBlock = () => {
    return (
      // Make the log block more opaque/visible per user request
      <div className="bg-slate-800/80 rounded p-3 font-mono text-xs text-slate-100 ring-1 ring-slate-700">
        {logs.map((text, i) => {
          const level = detectLevel(text);
          const dot = level === 'error' ? 'bg-red-400' : level === 'warn' ? 'bg-yellow-400' : 'bg-slate-400';
          return (
            <div key={i} className="flex items-start gap-2 py-1">
              <div className={`w-2 h-2 rounded-full mt-1 ${dot} flex-shrink-0`} />
              <div className="whitespace-pre-wrap break-words text-slate-100">{text}</div>
            </div>
          )
        })}
      </div>
    );
  };

  return (
  <div className="bg-slate-800/90 rounded-lg p-4 shadow-md border border-slate-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 bg-slate-700/30 p-2 rounded-md">
          <button onClick={() => setOpen(!open)} aria-label={open ? 'Cerrar' : 'Abrir'} className="p-1 rounded text-slate-200 hover:text-white">
            <ChevronDown className={`${open ? 'rotate-180' : ''} transition-transform`} />
          </button>
          <div>
            <div className="text-lg font-semibold text-white flex items-center gap-2">
              {tag}
            </div>
            <div className="text-sm text-slate-300">Run cells tagged "{tag}"</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 mr-2">
            {running ? (
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-sm text-green-300">Running</span>
              </div>
            ) : (
              <div className="text-sm text-slate-400">Idle</div>
            )}
          </div>

          <button
            onClick={handleRun}
            disabled={running}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-md disabled:opacity-60"
          >
            <Play size={16} /> Run
          </button>
          <button onClick={handleClear} className="flex items-center gap-2 px-3 py-2 rounded-md bg-gray-700 text-gray-200 hover:bg-gray-600">
            <Trash size={14} /> Clear
          </button>
          <button onClick={handleCopyAll} className="ml-2 text-sm px-3 py-2 rounded-md bg-slate-700 hover:bg-slate-600 text-slate-100">Copy all</button>
        </div>
      </div>

      {open && (
        <div className="mt-3 border-t border-slate-700 pt-3 max-h-56 overflow-auto pr-2">
          {logs.length === 0 ? (
            <div className="text-sm text-slate-400 p-4 bg-slate-800/80 rounded">No logs yet. Run this cell tag to see output.</div>
          ) : (
            <LogBlock />
          )}
        </div>
      )}
    </div>
  );
};

export default CellRunner;

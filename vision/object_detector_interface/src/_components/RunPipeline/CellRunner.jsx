import React, { useEffect, useState, useRef } from 'react';
import { ChevronDown, Play, Square, SquareArrowDown, Trash } from 'lucide-react';
import { ws, ws2 } from '../../pages/RunPage';
import ManuallyCheck from './ManuallyCheck';

const CellRunner = ({ tag }) => {
  // All
  const manuallyRef = useRef(null);

  ws2.onmessage = (msg) => {
    console.log("LOG:", msg.data);
    setLogs((s) => [...s, msg.data]);
    if(msg.data.includes("Finished")){
      setRunning(false);
      if (tag === "manually_check" && manuallyRef.current?.sendDeletedImages) {
        manuallyRef.current.sendDeletedImages();
      }
    }
  };

    const wsRun = (tag) => {
        ws.send(JSON.stringify({
          action: "run",
          tags: [tag]
        }));
    };

    const wsStop = (tag) => {
        ws2.send(JSON.stringify({
          action: "stop",
          tags: [tag]
        }));
    };

  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const logsEndRef = useRef(null);

  useEffect(() => {
    const lref = logsEndRef.current;
    if (!lref) return;
    const isBottom = lref.scrollHeight - lref.scrollTop - lref.clientHeight <= 1000;
    if (isBottom) {
      lref.scrollTop = lref.scrollHeight;
    } 
  }, [logs]);

  const handleRun = () => {
    setRunning(true);
    setOpen(true);
    try {
      wsRun(tag);
    } catch (e) {
      setLogs((s) => [...s, `Error sending run request: ${e.message}`]);
      setRunning(false);
    }
  };

  const handleStop = () => {
    try {
      wsStop(tag);
      setRunning(false);
    } catch (e) {
      setLogs((s) => [...s, `Error sending stop request: ${e.message}`]);
    }
  };


  const handleClear = () => setLogs([]);

  // Small helper to compute visual level from log text (purely visual)
  const detectLevel = (text) => {
    if (/error|failed|exception/i.test(text)) return 'error';
    if (/warn|warning/i.test(text)) return 'warn';
    return 'info';
  };

  const handleCopyAll = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'));
    } catch (e) {
    }
  };

  const LogBlock = () => {
    return (
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
  <div className="w-full bg-slate-800/90 rounded-lg p-4 shadow-md border border-slate-700">
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
          <button onClick={handleStop} className="flex items-center gap-2 px-3 py-2 rounded-md bg-red-600 hover:bg-red-700 text-white">
            <Square size={14} /> Stop
          </button>
          <button onClick={handleCopyAll} className="ml-2 text-sm px-3 py-2 rounded-md bg-slate-700 hover:bg-slate-600 text-slate-100">Copy all</button>
        </div>
      </div>

      {open && (
        <div ref={logsEndRef} className="mt-3 border-t border-slate-700 pt-3 max-h-56 overflow-auto pr-2">
          {logs.length === 0 ? (
            <div className="text-sm text-slate-400 p-4 bg-slate-800/80 rounded">No logs yet. Run this cell tag to see output.</div>
          ) : (
            <LogBlock />
          )}
        </div>
      )}
      {tag === "manually_check" && open && (<ManuallyCheck ref={manuallyRef} />)}
    </div>
  );
};

export default CellRunner;

import React, { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { ws } from "../../pages/RunPage";

const ManuallyCheck = forwardRef((props, ref) => {
  const [deleted, setDeleted] = useState([]);
  const [images, setImages] = useState([]);

  useEffect(() => {
    const handler = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log("WebSocket message received:");
        if (msg.type === "batch") {
          setImages(msg.images.map(img => ({ name: img.filename, src: `data:image/png;base64,${img.data}` })));
        }
      } catch (e) {
        console.error("Failed to parse WS message", e);
      }
    };

    if (ws && ws.addEventListener) {
      ws.addEventListener('message', handler);
      return () => ws.removeEventListener('message', handler);
    }

    if (ws) {
      ws.onmessage = handler;
      return () => { ws.onmessage = null; };
    }
  }, []);

  const sendDeletedImages = () => {
    if (!ws) return;
    ws.send(JSON.stringify({
      action: "delete",
      images: deleted
    }));
    console.log("Deleted images sent");
  };

  useImperativeHandle(ref, () => ({
    sendDeletedImages
  }), [deleted]);

  const handleDelete = (checked, src) => {
    console.log("Handle delete:", checked, src);
    if (checked) {
      setDeleted((prev) => [...prev, src]);
    } else {
      setDeleted((prev) => prev.filter(item => item !== src));
    }
  };

  const finishDelete = () => {
    if (!ws) return;
    ws.send(JSON.stringify({ action: "finish" }));
    console.log("Finish delete sent");
  };

  const nextBatch = () => {
    console.log("Next batch");
    if (!ws) return;
    ws.send(JSON.stringify({ action: "next" }));
  };

    const visibleImages = images.slice(0, 10);

    return (
        <div className="mt-4 w-full bg-slate-800/80 p-4 rounded-md border border-slate-600">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-white">Processed Images</h3>
            <div className="text-sm text-slate-300">Select images to delete</div>
          </div>

          <div className="grid grid-cols-5 grid-rows-2 gap-4">
            {Array.from({ length: 10 }).map((_, idx) => {
              const img = visibleImages[idx];
              return (
                <div key={idx} className="relative bg-slate-900/50 rounded-md border border-slate-700 p-2 flex flex-col items-center justify-between">
                  {/* Image area with fixed size */}
                  <div className="w-40 h-28 bg-slate-800 flex items-center justify-center overflow-hidden rounded">
                    {img ? (
                      <img src={img.src} alt={img.name || `Processed ${idx+1}`} className="w-full h-full object-contain" />
                    ) : (
                      <div className="text-slate-500 text-xs">No image</div>
                    )}
                  </div>

                  <div className="w-full mt-2 flex items-center justify-between gap-2">
                    <div className="text-xs text-slate-200 truncate">{img ? img.name : ''}</div>
                    <label className="flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        className="w-4 h-4"
                        disabled={!img}
                        onChange={(e) => img && handleDelete(e.target.checked, img.name)}
                      />
                      <span className="text-slate-300">Del</span>
                    </label>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-4 flex gap-3">
            <button onClick={nextBatch} className="flex-1 px-3 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white">Next Batch</button>
            <button onClick={finishDelete} className="flex-1 px-3 py-2 rounded-md bg-red-600 hover:bg-red-700 text-white">Delete Selected Images</button>
          </div>
        </div>
    );

});

export default ManuallyCheck;
import React, { useState } from 'react';
import { ArrowLeft, Play } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import CellRunner from '../_components/RunPipeline/CellRunner';

//export const ws = new WebSocket("ws://localhost:8000/ws");
let ws;
export const ws2 = new WebSocket("ws://localhost:9000/ws");

function connectWebSocket() {
  ws = new WebSocket("ws://localhost:8000/ws");

  ws.onopen = () => {
    console.log("✅ WebSocket conectado");
  };

  ws.onclose = () => {
    console.log("⚠️ WebSocket cerrado. Reintentando en 2s...");
    ws = null;
    setTimeout(() => {
      connectWebSocket();
    }, 2000);
  };
}

connectWebSocket();
export { ws }; 

const cells = [
  'setup',
  'downloadModels',
  'crop',
  'resize',
  'segment',
  'crop_precessed',
  'manually_check'
]
const RunPage = ({ onRunComplete }) => {

  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [showNotebook, setShowNotebook] = useState(false);

  const steps = ['Data Loaded', 'Resolution Adjustment', 'Silhouette Extraction', 'Background Generation', 'Training'];

  return (
    <div className="mt-12 p-8">
      <img src="../../../object_detector/DS_res/Cat/141.png" alt="Logo" width={150} height={150} className="mb-8" />
      <button 
        onClick={() => navigate('/')} 
        className="text-purple-300 hover:text-white mb-6 flex items-center gap-2"
      >
        <ArrowLeft size={20} /> Back to Dashboard
      </button>

      <h1 className="text-4xl font-bold text-white mb-8">Run New Pipeline</h1>

          <div className="mt-6">
            <h2 className="text-xl font-semibold text-white mb-4">Cell Executors</h2>
            <div className="grid grid-cols-1 gap-4">
              {cells.map((c) => (
                <CellRunner key={c} tag={c} ws={ws} ws2={ws2} />
              ))}
            </div>
          </div>

    </div>
  );
};

export default RunPage;

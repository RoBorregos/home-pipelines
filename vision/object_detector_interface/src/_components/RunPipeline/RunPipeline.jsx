import React, { useState } from 'react';
import { ArrowLeft, Play } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import UploadDataset from './UploadDataset';
import PipelineParams from './PipelineParams';
import ProgressTracker from './ProgressTracker';

const RunPipeline = ({ onRunComplete }) => {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = ['Data Loaded', 'Resolution Adjustment', 'Silhouette Extraction', 'Background Generation', 'Training'];

  const simulateRun = () => {
    setIsRunning(true);
    setCurrentStep(0);
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(interval);
          setIsRunning(false);
          onRunComplete();
          return prev;
        }
        return prev + 1;
      });
    }, 2000);
  };

  return (
    <div className="mt-12 p-8">
      <button 
        onClick={() => navigate('/')} 
        className="text-purple-300 hover:text-white mb-6 flex items-center gap-2"
      >
        <ArrowLeft size={20} /> Back to Dashboard
      </button>

      <h1 className="text-4xl font-bold text-white mb-8">Run New Pipeline</h1>

      {!isRunning ? (
        <div className="space-y-6 max-w-4xl">
          <UploadDataset uploadedFile={uploadedFile} setUploadedFile={setUploadedFile} />
          <PipelineParams />
          
          <button 
            onClick={simulateRun}
            disabled={!uploadedFile}
            className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white py-4 rounded-xl font-semibold text-lg shadow-xl transform hover:scale-105 transition flex items-center justify-center gap-3"
          >
            <Play size={24} />
            Start Run
          </button>
        </div>
      ) : (
        <div className="max-w-4xl">
          <ProgressTracker 
            currentStep={currentStep} 
            steps={steps} 
            onCancel={() => setIsRunning(false)} 
          />
        </div>
      )}
    </div>
  );
};

export default RunPipeline;
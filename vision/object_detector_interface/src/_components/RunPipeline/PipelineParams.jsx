import React from 'react';

const PipelineParams = () => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20">
      <h2 className="text-2xl font-semibold text-white mb-4">Pipeline Parameters</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-purple-200 mb-2">Target Resolution</label>
          <input 
            type="text" 
            defaultValue="640x480" 
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white" 
          />
        </div>
        <div>
          <label className="block text-purple-200 mb-2">Batch Size</label>
          <input 
            type="number" 
            defaultValue="32" 
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white" 
          />
        </div>
        <div>
          <label className="block text-purple-200 mb-2">Epochs</label>
          <input 
            type="number" 
            defaultValue="50" 
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white" 
          />
        </div>
        <div>
          <label className="block text-purple-200 mb-2">Learning Rate</label>
          <input 
            type="text" 
            defaultValue="0.001" 
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white" 
          />
        </div>
        <div>
          <label className="block text-purple-200 mb-2">Noise Level</label>
          <select className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white">
            <option>Low</option>
            <option>Medium</option>
            <option>High</option>
          </select>
        </div>
        <div>
          <label className="block text-purple-200 mb-2">Augmentation</label>
          <select className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-4 py-2 text-white">
            <option>Enabled</option>
            <option>Disabled</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default PipelineParams;
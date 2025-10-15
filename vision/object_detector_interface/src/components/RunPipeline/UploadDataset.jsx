import React from 'react';
import { Upload, CheckCircle } from 'lucide-react';

const UploadDataset = ({ uploadedFile, setUploadedFile }) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-3">
        <Upload size={24} />
        Upload Dataset
      </h2>
      <input 
        type="file" 
        onChange={(e) => setUploadedFile(e.target.files[0])}
        className="block w-full text-purple-200 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700 file:cursor-pointer"
        accept=".zip,.tar,.tar.gz"
      />
      {uploadedFile && (
        <p className="text-green-400 mt-3 flex items-center gap-2">
          <CheckCircle size={16} /> {uploadedFile.name}
        </p>
      )}
    </div>
  );
};

export default UploadDataset;
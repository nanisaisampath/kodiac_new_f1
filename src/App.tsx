import React, { useEffect, useState } from 'react';
import { FileExplorer } from './components/FileExplorer/FileExplorer';
import { DicomViewer } from './components/DicomViewer/DicomViewer';
import { Sidebar } from './components/Sidebar/Sidebar';
import { S3CredentialsModal } from './components/Modals/S3CredentialsModal';
import { LayoutControls } from './components/Layout/LayoutControls';
import { useAppStore } from './store/appStore';
import { api } from './services/api';
import { AlertTriangle, CheckCircle, Wifi, WifiOff } from 'lucide-react';

function App() {
  const { viewports, settings, setCacheStatus } = useAppStore();
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [s3Status, setS3Status] = useState<'checking' | 'configured' | 'needs_credentials'>('checking');

  useEffect(() => {
    console.log('🚀 App starting up, checking backend and S3...');
    
    // Check backend connectivity and S3 status
    const checkBackendAndS3 = async () => {
      try {
        console.log('🔍 Checking backend connectivity...');
        
        // Check cache status to verify backend connectivity
        const cacheStatus = await api.getCacheStatus();
        setCacheStatus(cacheStatus);
        setBackendStatus('connected');
        console.log('✅ Backend connected successfully');

        // Check S3 configuration
        try {
          console.log('🔍 Checking S3 configuration...');
          const s3StatusResponse = await api.getS3Status();
          setS3Status(s3StatusResponse.configured ? 'configured' : 'needs_credentials');
          console.log('📊 S3 status:', s3StatusResponse);
        } catch (error) {
          console.warn('⚠️ S3 status check failed:', error);
          setS3Status('needs_credentials');
        }
      } catch (error) {
        console.warn('❌ Backend not available:', error);
        setBackendStatus('disconnected');
        setS3Status('checking');
        // Set mock cache status for demo
        setCacheStatus({
          total_files: 0,
          cached_files: 0,
          cache_size_mb: 0,
          hit_rate: 0
        });
      }
    };

    checkBackendAndS3();

    // Refresh status every 30 seconds
    const interval = setInterval(() => {
      console.log('🔄 Refreshing backend and S3 status...');
      checkBackendAndS3();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [setCacheStatus]);

  const viewport1 = viewports.find(vp => vp.id === 1)!;
  const viewport2 = viewports.find(vp => vp.id === 2)!;

  const getStatusBanner = () => {
    if (backendStatus === 'disconnected') {
      return (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 flex items-center gap-2 text-red-800">
          <WifiOff className="w-4 h-4" />
          <span className="text-sm">
            Backend server not available. Please start the FastAPI backend on localhost:8000.
          </span>
        </div>
      );
    }

    if (backendStatus === 'connected' && s3Status === 'needs_credentials') {
      return (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 flex items-center gap-2 text-amber-800">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">
            Backend connected but S3 credentials needed. Click "S3 Credentials" in the sidebar to configure.
          </span>
        </div>
      );
    }

    if (backendStatus === 'connected' && s3Status === 'configured') {
      return (
        <div className="bg-green-50 border-b border-green-200 px-4 py-2 flex items-center gap-2 text-green-800">
          <CheckCircle className="w-4 h-4" />
          <span className="text-sm">
            System ready - Backend and S3 configured successfully.
          </span>
        </div>
      );
    }

    return null;
  };

  console.log('🎨 Rendering App with status:', { backendStatus, s3Status });

  return (
    <div className="h-screen bg-gray-100 overflow-hidden">
      <Sidebar />
      
      <div className={`h-full transition-all duration-300 ${
        settings.sidebarExpanded ? 'ml-64' : 'ml-16'
      }`}>
        {/* Status Banner */}
        {getStatusBanner()}

        <div className="h-full flex">
          {/* Left Panel - File Explorer */}
          <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
            <FileExplorer />
          </div>

          {/* Main Content Area */}
          <div className="flex-1 p-4 flex flex-col gap-4">
            {/* Layout Controls */}
            <LayoutControls />

            {/* Viewports */}
            <div className={`flex-1 flex gap-4 ${
              settings.layoutMode === 'stacked' ? 'flex-col' : 'flex-row'
            }`}>
              <DicomViewer viewport={viewport1} />
              <DicomViewer viewport={viewport2} />
            </div>
          </div>
        </div>
      </div>

      <S3CredentialsModal />
    </div>
  );
}

export default App;
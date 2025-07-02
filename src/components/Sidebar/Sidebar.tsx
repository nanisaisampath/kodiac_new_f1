import React from 'react';
import { 
  Upload, 
  FileText, 
  Settings, 
  Database, 
  BarChart3,
  ChevronRight,
  Menu
} from 'lucide-react';
import { useAppStore } from '../../store/appStore';

export function Sidebar() {
  const { settings, updateSettings, setShowS3Modal, cacheStatus } = useAppStore();

  const toggleSidebar = () => {
    updateSettings({ sidebarExpanded: !settings.sidebarExpanded });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Handle file upload via API
      console.log('Uploading file:', file.name);
    }
  };

  const sidebarItems = [
    {
      icon: Upload,
      label: 'E2E to DICOM',
      action: () => document.getElementById('file-upload')?.click(),
    },
    {
      icon: FileText,
      label: 'Metadata Extractor',
      action: () => console.log('Extract metadata'),
    },
    {
      icon: Database,
      label: 'S3 Credentials',
      action: () => setShowS3Modal(true),
    },
    {
      icon: Settings,
      label: 'Settings',
      action: () => console.log('Open settings'),
    },
  ];

  return (
    <>
      {/* Sidebar Toggle Button */}
      <button
        onClick={toggleSidebar}
        className="fixed left-4 top-4 z-50 p-2 bg-white border border-gray-300 rounded-lg shadow-md hover:bg-gray-50 transition-colors"
      >
        <Menu className="w-4 h-4" />
      </button>

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full bg-gray-800 text-white shadow-lg transition-all duration-300 z-40 ${
          settings.sidebarExpanded ? 'w-64' : 'w-16'
        }`}
        onMouseEnter={() => updateSettings({ sidebarExpanded: true })}
        onMouseLeave={() => updateSettings({ sidebarExpanded: false })}
      >
        <div className="pt-16 p-4">
          {/* Logo/Title */}
          <div className="mb-8">
            {settings.sidebarExpanded ? (
              <h2 className="text-lg font-bold text-white">DICOM Viewer</h2>
            ) : (
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                <FileText className="w-4 h-4" />
              </div>
            )}
          </div>

          {/* Navigation Items */}
          <nav className="space-y-2">
            {sidebarItems.map((item, index) => (
              <button
                key={index}
                onClick={item.action}
                className="w-full flex items-center p-3 rounded-lg hover:bg-gray-700 transition-colors group"
              >
                <item.icon className="w-5 h-5 text-gray-300 group-hover:text-white" />
                {settings.sidebarExpanded && (
                  <>
                    <span className="ml-3 text-sm font-medium">{item.label}</span>
                    <ChevronRight className="w-4 h-4 ml-auto text-gray-400 group-hover:text-white" />
                  </>
                )}
              </button>
            ))}
          </nav>

          {/* Cache Status */}
          {settings.sidebarExpanded && cacheStatus && (
            <div className="mt-8 p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center mb-2">
                <BarChart3 className="w-4 h-4 text-blue-400" />
                <span className="ml-2 text-sm font-medium">Cache Status</span>
              </div>
              
              <div className="space-y-1 text-xs text-gray-300">
                <div className="flex justify-between">
                  <span>Size:</span>
                  <span>{cacheStatus.cache_size_mb.toFixed(1)}MB</span>
                </div>
                <div className="flex justify-between">
                  <span>Hit Rate:</span>
                  <span>{(cacheStatus.hit_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Files:</span>
                  <span>{cacheStatus.cached_files}/{cacheStatus.total_files}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        id="file-upload"
        type="file"
        accept=".e2e,.dcm,.dicom,.fds,.fda"
        onChange={handleFileUpload}
        className="hidden"
      />
    </>
  );
}
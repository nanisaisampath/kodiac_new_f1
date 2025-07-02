import React from 'react';
import { LayoutGrid, Layers, Link, Unlink } from 'lucide-react';
import { useAppStore } from '../../store/appStore';

export function LayoutControls() {
  const { settings, updateSettings } = useAppStore();

  const toggleLayout = () => {
    const newLayout = settings.layoutMode === 'side-by-side' ? 'stacked' : 'side-by-side';
    updateSettings({ layoutMode: newLayout });
  };

  const toggleBindSliders = () => {
    updateSettings({ bindSliders: !settings.bindSliders });
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
      <div className="flex items-center gap-4">
        {/* Layout Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Layout:</span>
          <button
            onClick={toggleLayout}
            className={`flex items-center gap-2 px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              settings.layoutMode === 'side-by-side'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            title="Switch to side-by-side layout"
          >
            <LayoutGrid className="w-4 h-4" />
            Side-by-Side
          </button>
          
          <button
            onClick={toggleLayout}
            className={`flex items-center gap-2 px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              settings.layoutMode === 'stacked'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            title="Switch to stacked layout"
          >
            <Layers className="w-4 h-4" />
            Stacked
          </button>
        </div>

        {/* Bind Sliders Toggle */}
        <div className="flex items-center gap-2 ml-4 pl-4 border-l border-gray-200">
          <span className="text-sm font-medium text-gray-700">Sync Frames:</span>
          <button
            onClick={toggleBindSliders}
            className={`flex items-center gap-2 px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              settings.bindSliders
                ? 'bg-green-100 text-green-700'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            title={settings.bindSliders ? 'Unbind frame sliders' : 'Bind frame sliders'}
          >
            {settings.bindSliders ? (
              <>
                <Link className="w-4 h-4" />
                Linked
              </>
            ) : (
              <>
                <Unlink className="w-4 h-4" />
                Independent
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
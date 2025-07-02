import { create } from 'zustand';
import type { ViewportState, AppSettings, CacheStatus } from '../types';
import { api } from '../services/api';

interface AppStore {
  // Viewports
  viewports: ViewportState[];
  updateViewport: (id: number, updates: Partial<ViewportState>) => void;
  loadFileInViewport: (viewportId: number, filePath: string) => void;
  
  // Settings
  settings: AppSettings;
  updateSettings: (updates: Partial<AppSettings>) => void;
  
  // Cache
  cacheStatus: CacheStatus | null;
  setCacheStatus: (status: CacheStatus) => void;
  
  // UI State
  showS3Modal: boolean;
  setShowS3Modal: (show: boolean) => void;
  selectedFile: string | null;
  setSelectedFile: (filePath: string | null) => void;
  contextMenu: { x: number; y: number; filePath: string } | null;
  setContextMenu: (menu: { x: number; y: number; filePath: string } | null) => void;
}

export const useAppStore = create<AppStore>((set, get) => ({
  // Initial viewport states
  viewports: [
    {
      id: 1,
      filePath: null,
      frames: [],
      currentFrame: 0,
      zoom: 1,
      loading: false,
      error: null,
      metadata: null,
      crcKey: null,
      loadTime: null,
      processedFileKey: null,
      isOct: false,
    },
    {
      id: 2,
      filePath: null,
      frames: [],
      currentFrame: 0,
      zoom: 1,
      loading: false,
      error: null,
      metadata: null,
      crcKey: null,
      loadTime: null,
      processedFileKey: null,
      isOct: false,
    },
  ],

  updateViewport: (id, updates) =>
    set((state) => ({
      viewports: state.viewports.map((vp) =>
        vp.id === id ? { ...vp, ...updates } : vp
      ),
    })),

  loadFileInViewport: async (viewportId, filePath) => {
    const { updateViewport } = get();
    
    updateViewport(viewportId, {
      filePath,
      loading: true,
      error: null,
      frames: [],
      currentFrame: 0,
      zoom: 1,
      metadata: null,
      crcKey: null,
      loadTime: null,
      processedFileKey: null,
      isOct: false,
    });

    try {
      // Process the file through the backend
      const processResponse = await api.downloadDicomFromS3(filePath);
      
      // Store the processed file key for future frame requests
      updateViewport(viewportId, {
        processedFileKey: processResponse.dicom_file_path,
        loading: false,
      });
      
    } catch (error) {
      updateViewport(viewportId, {
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to process file',
      });
    }
  },

  // Default settings
  settings: {
    layoutMode: 'side-by-side',
    bindSliders: false,
    sidebarExpanded: false,
  },

  updateSettings: (updates) =>
    set((state) => ({
      settings: { ...state.settings, ...updates },
    })),

  // Cache status
  cacheStatus: null,
  setCacheStatus: (status) => set({ cacheStatus: status }),

  // UI state
  showS3Modal: false,
  setShowS3Modal: (show) => set({ showS3Modal: show }),
  
  selectedFile: null,
  setSelectedFile: (filePath) => set({ selectedFile: filePath }),
  
  contextMenu: null,
  setContextMenu: (menu) => set({ contextMenu: menu }),
}));
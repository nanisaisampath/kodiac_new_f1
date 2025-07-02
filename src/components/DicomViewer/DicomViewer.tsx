import React, { useEffect, useState, useRef } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Loader, X } from 'lucide-react';
import { api } from '../../services/api';
import { useAppStore } from '../../store/appStore';
import type { ViewportState } from '../../types';

interface DicomViewerProps {
  viewport: ViewportState;
}

export function DicomViewer({ viewport }: DicomViewerProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const { updateViewport, settings } = useAppStore();

  useEffect(() => {
    if (viewport.filePath && !viewport.loading) {
      loadDicomFrames();
    }
  }, [viewport.filePath]);

  useEffect(() => {
    if (viewport.frames.length > 0 && !loadingFrame) {
      loadCurrentFrame();
    }
  }, [viewport.currentFrame, viewport.frames]);

  const loadDicomFrames = async () => {
    if (!viewport.filePath) return;

    try {
      updateViewport(viewport.id, { loading: true, error: null });
      const startTime = Date.now();

      // Get CRC for caching
      const crcResponse = await api.getFileCrc(viewport.filePath);
      
      // Check if it's an OCT file and try flattened version first
      const isOCT = viewport.filePath.toLowerCase().includes('oct') || 
                   viewport.filePath.toLowerCase().includes('.e2e') ||
                   viewport.filePath.toLowerCase().includes('.fda');
      
      if (isOCT) {
        try {
          const flattenedBlob = await api.getFlattenedDicom(viewport.filePath);
          const url = URL.createObjectURL(flattenedBlob);
          
          updateViewport(viewport.id, {
            frames: [{ index: 0, url, cached: true }],
            currentFrame: 0,
            loading: false,
            crcKey: crcResponse.crc,
            loadTime: Date.now() - startTime,
            isOct: true,
            metadata: {
              frameCount: 1,
              width: 0,
              height: 0,
              imageType: 'OCT Flattened',
            },
          });
          return;
        } catch (error) {
          console.warn('Failed to load flattened OCT, falling back to regular frames');
        }
      }

      // Load regular DICOM frames
      const framesResponse = await api.getDicomFrames(viewport.filePath);
      
      const frames = framesResponse.frames.map(index => ({
        index,
        url: '',
        cached: false,
      }));

      updateViewport(viewport.id, {
        frames,
        currentFrame: 0,
        loading: false,
        crcKey: crcResponse.crc,
        loadTime: Date.now() - startTime,
        isOct: false,
        metadata: {
          frameCount: frames.length,
          width: 0,
          height: 0,
          imageType: 'DICOM',
        },
      });
    } catch (error) {
      updateViewport(viewport.id, {
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load DICOM',
      });
    }
  };

  const loadCurrentFrame = async () => {
    if (!viewport.filePath || viewport.frames.length === 0) return;

    const currentFrameData = viewport.frames[viewport.currentFrame];
    if (currentFrameData.url) {
      setImageUrl(currentFrameData.url);
      return;
    }

    try {
      setLoadingFrame(true);
      
      // Use the processed file key if available, otherwise use the original path
      const fileKey = viewport.processedFileKey || viewport.filePath;
      
      const blob = await api.getDicomFrame(
        fileKey,
        currentFrameData.index,
        viewport.crcKey || undefined
      );
      
      const url = URL.createObjectURL(blob);
      setImageUrl(url);
      
      // Update frame with cached URL
      const updatedFrames = [...viewport.frames];
      updatedFrames[viewport.currentFrame] = { ...currentFrameData, url, cached: true };
      updateViewport(viewport.id, { frames: updatedFrames });
    } catch (error) {
      updateViewport(viewport.id, {
        error: error instanceof Error ? error.message : 'Failed to load frame',
      });
    } finally {
      setLoadingFrame(false);
    }
  };

  const handleZoom = (factor: number) => {
    const newZoom = Math.max(0.1, Math.min(5, viewport.zoom * factor));
    updateViewport(viewport.id, { zoom: newZoom });
  };

  const handleResetZoom = () => {
    updateViewport(viewport.id, { zoom: 1 });
  };

  const handleFrameChange = (frame: number) => {
    updateViewport(viewport.id, { currentFrame: frame });
    
    // If sliders are bound, update other viewport too
    if (settings.bindSliders) {
      const otherViewportId = viewport.id === 1 ? 2 : 1;
      updateViewport(otherViewportId, { currentFrame: frame });
    }
  };

  const handleClose = () => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    updateViewport(viewport.id, {
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
    });
    setImageUrl(null);
  };

  if (!viewport.filePath) {
    return (
      <div className="flex-1 bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-2">📁</div>
          <p>Right-click a DICOM file to load it here</p>
          <p className="text-sm mt-1">Viewport {viewport.id}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-white border border-gray-300 rounded-lg overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-gray-100 px-4 py-2 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-gray-800 truncate">
              Viewport {viewport.id} {viewport.isOct && '(OCT)'}
            </h4>
            <p className="text-xs text-gray-600 truncate">
              {viewport.filePath.split('/').pop()}
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            {viewport.loadTime && (
              <span className="text-xs text-green-600">
                {(viewport.loadTime / 1000).toFixed(1)}s
              </span>
            )}
            
            <button
              onClick={handleClose}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleZoom(1.2)}
              className="p-1 hover:bg-gray-200 rounded"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => handleZoom(0.8)}
              className="p-1 hover:bg-gray-200 rounded"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            
            <button
              onClick={handleResetZoom}
              className="p-1 hover:bg-gray-200 rounded"
              title="Reset Zoom"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            
            <span className="text-sm text-gray-600 ml-2">
              {Math.round(viewport.zoom * 100)}%
            </span>
          </div>
          
          {viewport.metadata && (
            <div className="text-xs text-gray-600">
              {viewport.metadata.imageType} • {viewport.metadata.frameCount} frames
            </div>
          )}
        </div>
        
        {/* Frame Slider */}
        {viewport.frames.length > 1 && (
          <div className="mt-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-600 w-12">
                Frame {viewport.currentFrame + 1}
              </span>
              
              <input
                type="range"
                min={0}
                max={viewport.frames.length - 1}
                value={viewport.currentFrame}
                onChange={(e) => handleFrameChange(parseInt(e.target.value))}
                className="flex-1"
              />
              
              <span className="text-xs text-gray-600 w-12 text-right">
                of {viewport.frames.length}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Image Display */}
      <div className="flex-1 overflow-hidden relative bg-black">
        {viewport.loading && (
          <div className="absolute inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-10">
            <div className="text-center text-white">
              <Loader className="w-8 h-8 animate-spin mx-auto mb-2" />
              <p>Loading DICOM...</p>
            </div>
          </div>
        )}
        
        {loadingFrame && (
          <div className="absolute top-2 right-2 z-10">
            <Loader className="w-4 h-4 animate-spin text-white" />
          </div>
        )}
        
        {viewport.error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-red-600 bg-white p-4 rounded-lg">
              <p className="font-medium">Error loading image</p>
              <p className="text-sm mt-1">{viewport.error}</p>
            </div>
          </div>
        )}
        
        {imageUrl && !viewport.loading && !viewport.error && (
          <div className="w-full h-full flex items-center justify-center overflow-auto">
            <img
              ref={imageRef}
              src={imageUrl}
              alt={`DICOM Frame ${viewport.currentFrame + 1}`}
              className="max-w-none"
              style={{
                transform: `scale(${viewport.zoom})`,
                transformOrigin: 'center',
              }}
              onLoad={() => {
                if (imageRef.current && viewport.metadata) {
                  updateViewport(viewport.id, {
                    metadata: {
                      ...viewport.metadata,
                      width: imageRef.current.naturalWidth,
                      height: imageRef.current.naturalHeight,
                    },
                  });
                }
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
import React, { useEffect, useRef } from 'react';
import { useAppStore } from '../../store/appStore';

export function ContextMenu() {
  const menuRef = useRef<HTMLDivElement>(null);
  const { contextMenu, setContextMenu, loadFileInViewport } = useAppStore();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setContextMenu(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [setContextMenu]);

  if (!contextMenu) return null;

  const handleLoadInViewport = (viewportId: number) => {
    if (contextMenu.filePath) {
      loadFileInViewport(viewportId, contextMenu.filePath);
    }
    setContextMenu(null);
  };

  return (
    <div
      ref={menuRef}
      className="fixed bg-white border border-gray-200 rounded-md shadow-lg py-1 z-50"
      style={{
        left: contextMenu.x,
        top: contextMenu.y,
      }}
    >
      <button
        onClick={() => handleLoadInViewport(1)}
        className="block w-full text-left px-4 py-2 text-sm hover:bg-blue-50 hover:text-blue-700"
      >
        Load in Viewport 1
      </button>
      <button
        onClick={() => handleLoadInViewport(2)}
        className="block w-full text-left px-4 py-2 text-sm hover:bg-blue-50 hover:text-blue-700"
      >
        Load in Viewport 2
      </button>
    </div>
  );
}